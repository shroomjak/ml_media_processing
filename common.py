"""Общие подготовка кадра, декодирование и обработка файла для двух Python backend.
Модель: YOLO11-seg, два класса, 32 коэффициента маски, вход 640×640.
"""
import time
from pathlib import Path
import cv2
import numpy as np

# Параметры одинаковы для HEF и ONNX: их изменение меняет итоговую разметку.
SIZE = 640
CONF, IOU, MASK = 0.25, 0.45, 0.5
cv2.setNumThreads(1)


def sigmoid(x):
    # Ограничение аргумента предотвращает переполнение exp для больших значений.
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def prepare(image):
    """uint8 серое/BGR изображение → RGB 640×640; сохранить геометрию."""
    if image.dtype != np.uint8:
        raise ValueError('Нужно 8-битное изображение')
    # Серый пиксель копируется в три канала: модель ожидает RGB, не grayscale.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Нужно 1 или 3 канала')
    h, w = image.shape[:2]
    # Letterbox сохраняет пропорции; для 1000×1000 поля не понадобятся.
    # Геометрию возвращаем, чтобы после инференса восстановить исходный размер.
    scale = min(SIZE / w, SIZE / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (SIZE - nw) // 2, (SIZE - nh) // 2
    # Фон 114 должен совпадать с подготовкой обучающих/калибровочных данных.
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return image, rgb, (left, top, nw, nh, w, h)


def nms(boxes, scores, classes, coeff):
    """Отбор максимум 300 кандидатов и NMS отдельно для каждого класса."""
    # Самые уверенные кандидаты обрабатываются первыми; 300 ограничивает цену NMS.
    order = np.argsort(scores)[::-1][:300]
    kept = []
    while len(order) and len(kept) < 50:
        i = order[0]
        kept.append(i)
        rest = order[1:]
        if not len(rest):
            break
        # IoU = площадь пересечения / площадь объединения двух рамок.
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        wh = np.maximum(rb - lt, 0)
        intersection = wh[:, 0] * wh[:, 1]
        area_i = np.prod(np.maximum(boxes[i, 2:] - boxes[i, :2], 0))
        area_r = np.prod(np.maximum(boxes[rest, 2:] - boxes[rest, :2], 0), axis=1)
        overlap = intersection / (area_i + area_r - intersection + 1e-7)
        # Подавляем перекрывающиеся рамки только своего класса; коэффициенты идут вместе с рамкой.
        order = rest[(classes[rest] != classes[i]) | (overlap <= IOU)]
    return boxes[kept], scores[kept], classes[kept], coeff[kept]


def raw_heads(outputs):
    """Декодировать 10 де-квантизированных выходов HEF в рамки и маски."""
    boxes, scores, classes, coeffs = [], [], [], []
    # Три масштаба сетки предсказывают объекты разного размера.
    for size in (20, 40, 80):
        stride = SIZE // size
        # 64 = четыре стороны рамки × 16 логитов распределения DFL.
        # reshape здесь меняет представление массива для декодера, а не размер картинки.
        raw = outputs[(size, size, 64)].reshape(-1, 4, 16)
        cls = outputs[(size, size, 2)].reshape(-1, 2)  # sigmoid уже в HEF
        coeff = outputs[(size, size, 32)].reshape(-1, 32)  # без sigmoid
        # Упрощение: выбираем один лучший класс на ячейку и отбрасываем низкую уверенность.
        ids = cls.argmax(axis=1)
        values = cls[np.arange(len(cls)), ids]
        keep = np.flatnonzero(values >= CONF)
        if not len(keep):
            continue
        # DFL: ожидание по 16 бинам для каждой из 4 сторон.
        # Стабильный softmax: вычитание максимума защищает exp от переполнения.
        x = np.clip(raw[keep], -50, 50)
        prob = np.exp(x - x.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)
        dist = (prob * np.arange(16, dtype=np.float32)).sum(axis=-1) * stride
        # Индекс ячейки → её центр в пикселях входа; расстояния превращаем в xyxy.
        cx = (keep % size + 0.5) * stride
        cy = (keep // size + 0.5) * stride
        xyxy = np.stack((cx-dist[:, 0], cy-dist[:, 1], cx+dist[:, 2], cy+dist[:, 3]), axis=1)
        boxes.append(np.clip(xyxy, 0, SIZE))
        scores.append(values[keep]); classes.append(ids[keep]); coeffs.append(coeff[keep])
    proto = outputs[(160, 160, 32)].reshape(160, 160, 32)
    if not boxes:
        return []
    return masks(np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes), np.concatenate(coeffs), proto)


def onnx_heads(outputs):
    """Декодировать обычный экспорт Ultralytics: [1,38,8400], [1,32,160,160]."""
    arrays = [np.asarray(x) for x in outputs]
    pred = next((x for x in arrays if x.shape == (1, 38, 8400)), None)
    proto = next((x for x in arrays if x.shape == (1, 32, 160, 160)), None)
    if len(arrays) != 2 or pred is None or proto is None:
        raise ValueError('Ожидаются выходы ONNX [1,38,8400] и [1,32,160,160]')
    # ONNX уже выдаёт xywh в пикселях и оценки классов: DFL повторно не нужен.
    # После транспонирования: одна строка = один из 8400 кандидатов.
    pred = pred[0].T
    ids = pred[:, 4:6].argmax(axis=1)
    scores = pred[np.arange(len(pred)), 4 + ids]
    keep = np.flatnonzero(scores >= CONF)
    if not len(keep):
        return []
    # Перевод из (центр x, центр y, ширина, высота) в (x1, y1, x2, y2).
    xywh = pred[keep, :4]
    boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2,
                            xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    return masks(np.clip(boxes, 0, SIZE), scores[keep], ids[keep], pred[keep, 6:38], proto[0].transpose(1, 2, 0))


def masks(boxes, scores, classes, coeff, proto):
    # Сначала NMS: вычислять полные маски для отброшенных кандидатов не нужно.
    boxes, scores, classes, coeff = nms(boxes, scores, classes, coeff)
    # Линейная комбинация 32 прототипов создаёт маску каждого объекта.
    return [(box, float(score), int(cls), sigmoid(proto @ c))
            for box, score, cls, c in zip(boxes, scores, classes, coeff)]


def draw(image, objects, geo):
    left, top, nw, nh, w, h = geo
    result = image.copy()
    for box, score, cls, mask in objects:
        # Из 160×160 возвращаемся к 640×640, снимаем поля letterbox и восстанавливаем размер кадра.
        mask = cv2.resize(mask, (SIZE, SIZE))[top:top+nh, left:left+nw]
        mask = cv2.resize(mask, (w, h))
        x1 = int(np.clip(np.floor((box[0] - left) * w / nw), 0, w))
        y1 = int(np.clip(np.floor((box[1] - top) * h / nh), 0, h))
        x2 = int(np.clip(np.ceil((box[2] - left) * w / nw), 0, w))
        y2 = int(np.clip(np.ceil((box[3] - top) * h / nh), 0, h))
        if x2 <= x1 or y2 <= y1:
            continue
        color = np.array(((37*cls+67)%256, (17*cls+149)%256, (97*cls+83)%256), np.uint8)
        roi = result[y1:y2, x1:x2]
        # Отображаем пороговую маску только внутри рамки с прозрачностью 40%.
        active = mask[y1:y2, x1:x2] > MASK
        roi[active] = (0.6 * roi[active] + 0.4 * color).astype(np.uint8)
        c = tuple(int(x) for x in color)
        cv2.rectangle(result, (x1, y1), (min(x2, w-1), min(y2, h-1)), c, 2)
        cv2.putText(result, f'id:{cls} {score:.2f}', (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
    return result


def process_file(input_path, output_path, backend):
    """backend: prepare(rgb), infer(input), decode(output). Время — мс/кадр."""
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError('Вход и выход должны быть разными файлами')
    # Фото читаем один раз; прочие файлы рассматриваем как видео.
    image_mode = Path(input_path).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    cap = None if image_mode else cv2.VideoCapture(input_path)
    if cap is not None and not cap.isOpened():
        raise RuntimeError('Не удалось открыть видео')
    writer = None
    # Каждый список содержит время одного этапа для каждого успешно обработанного кадра.
    timings = {k: [] for k in ('read', 'preprocess', 'infer', 'postprocess', 'draw', 'write')}
    try:
        while True:
            start = time.perf_counter()
            frame = cv2.imread(input_path, cv2.IMREAD_UNCHANGED) if image_mode else cap.read()[1]
            if frame is None:
                if image_mode:
                    raise RuntimeError('Не удалось прочитать изображение')
                break
            timings['read'].append((time.perf_counter()-start)*1000)
            start = time.perf_counter()
            frame, rgb, geo = prepare(frame)
            data = backend.prepare(rgb)
            timings['preprocess'].append((time.perf_counter()-start)*1000)
            # Только вызов runtime: у Hailo это ещё и передача/чтение данных, не чистое время NPU.
            start = time.perf_counter(); result = backend.infer(data)
            timings['infer'].append((time.perf_counter()-start)*1000)
            start = time.perf_counter(); objects = backend.decode(result)
            timings['postprocess'].append((time.perf_counter()-start)*1000)
            start = time.perf_counter(); painted = draw(frame, objects, geo)
            timings['draw'].append((time.perf_counter()-start)*1000)
            start = time.perf_counter()
            if image_mode:
                if not cv2.imwrite(output_path, painted):
                    raise RuntimeError('Не удалось записать изображение')
            else:
                # Создаём writer по первому кадру: известны исходные размеры и FPS.
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps <= 0:
                        raise RuntimeError('Не удалось определить FPS видео')
                    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (painted.shape[1], painted.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Не удалось создать видео')
                writer.write(painted)
            timings['write'].append((time.perf_counter()-start)*1000)
            if image_mode:
                break
        if not timings['read']:
            raise RuntimeError('Нет кадров')
    finally:
        # Закрываем видео и дописываем контейнер даже при ошибке обработки.
        if cap is not None: cap.release()
        if writer is not None: writer.release()
    print('Время на кадр, мс: среднее +- СКО (все кадры)')
    # np.std по умолчанию делит на N: это СКО всей измеренной последовательности.
    for name, values in timings.items():
        print(f'{name:12s}: {np.mean(values):8.2f} +- {np.std(values):.2f}')
    print('Кадров:', len(timings['read']), '| Результат:', output_path)
