# ml_media_processing

Инференс сегментации на Hailo (C++ + HailoRT + OpenCV): фото или видео на входе,
файл того же формата на выходе — с масками, рамками и классами.

Два самостоятельных бэкенда, по одному на архитектуру головы. Общего «универсального»
декодера нет сознательно: параметры модели переносятся вручную из `.alls` и из `--inspect`.

| Бэкенд | Модель | Выходы HEF | Особенность |
|---|---|---|---|
| `yolo26s_seg.cpp` | YOLO26s-seg | 3×(box 4, cls NC, coeff 32) + protos | без DFL (`reg_max = 1`) |
| `yolo11s_seg.cpp` | YOLO11s-seg / YOLOv8s-seg | 3×(box 64, cls NC, coeff 32) + protos | DFL 16 бинов |

NMS в HEF не используется: для instance segmentation Hailo DFC его не поддерживает,
NMS выполняется на CPU в `common.hpp::select_nms()`.

## Сборка

```bash
cmake -S . -B build && cmake --build build -j
```

Зависимости: OpenCV 4, HailoRT (4.x/5.x), C++17.

## Запуск

```bash
# формы и имена выходов HEF — без платы
./build/yolo26s_seg --inspect yolo26s-seg.hef

# фото
./build/yolo26s_seg yolo26s-seg.hef photo.jpg out.jpg
# видео (кодек выбирается по расширению: .mp4 → mp4v, .avi → MJPG)
./build/yolo26s_seg yolo26s-seg.hef in.mp4 out.mp4
```

## Файлы

- `common.hpp` — letterbox 640×640 (pad 114, BGR→RGB), NMS, маски, отрисовка, а также
  `decode()` для головы с DFL (используется только `yolo11s_seg.cpp`).
- `hailo_setup.hpp` — обвязка HailoRT: устройство → HEF → сеть → потоки → `activate()`.
- `media.hpp` — общий цикл по кадрам и замеры read/preprocess/infer/postprocess/draw/write.
- `alls/` — model scripts: базовый для yolo26s-seg и эталон с `nms_postprocess` для ревизии.
- `docs/REVIEW.md` — разбор исходного `hailo.cpp` (встроенный NMS) и список исправлений.
- `docs/YOLO26S_SEG.md` — как пересобрать `.alls`, как узнать размерности и что менять в коде.
- `tools/onnx_end_nodes.py` — печатает end nodes головы для парсинга ONNX в DFC.
