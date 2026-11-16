# Ревизия `hailo.cpp`: обработка с NMS, скомпилированным в HEF

Проверялось по исходникам HailoRT (`hailo/hailort.hpp`, `hailort.h`, `vstream.hpp`,
`network_group.hpp`, ветка `master`, v4/v5) и компиляцией `g++ -std=c++17 -fsyntax-only`
против этих заголовков. Платы и DFC в среде не было, поэтому HEF не запускался:
утверждения ниже следуют из API и документации, а не из прогона на железе.

## Главный вывод

В HailoRT **нет** API, включающего NMS на этапе выполнения. NMS целиком задаётся при
компиляции модели в `.alls` (`nms_postprocess(...)`) и «зашивается» в HEF. В рантайме
можно только подкрутить пороги уже скомпилированного NMS:

```cpp
output_vstream.set_nms_score_threshold(0.3f);
output_vstream.set_nms_iou_threshold(0.5f);
output_vstream.set_nms_max_proposals_per_class(100);
```

Отсюда два неизбежных следствия для исходного кода:

1. Вызова `set_postprocessing_config(...)` не существует — файл не собирался.
2. Команды `nms_processing` в DFC тоже нет; правильное имя — `nms_postprocess`, и она
   применяется к модели **до** компиляции, а не в C++.
3. Для instance segmentation (YOLOv8-seg, YOLO11-seg, YOLO26-seg) NMS в HEF не
   поддерживается вообще: в model zoo `yolov8s_seg.alls` состоит только из `normalization`
   и `change_output_activation`, без `nms_postprocess`. Значит HEF для seg всегда отдаёт
   сырые конвы, и NMS обязан быть на хосте.

## Дефекты исходного `hailo.cpp`

### Не компилируется

| # | Место | Проблема |
|---|---|---|
| 1 | `config.value().set_postprocessing_config(nms_config_path)` | Метода нет у `ConfigureNetworkParams`/`NetworkGroupsParamsMap`. NMS задаётся в `.alls`. |
| 2 | `postprocess()`: `outputs[0].read()` | У `OutputVStream` есть только `read(MemoryView buffer)`. Перегрузки без аргумента нет. |

### Не работает в рантайме

| # | Место | Проблема | Последствие |
|---|---|---|---|
| 3 | конструктор | Нет `group->activate()` | `write`/`read` возвращают `HAILO_STREAM_NOT_ACTIVATED`; `ActivatedNetworkGroup` не хранится |
| 4 | конструктор | `if (inputs.size() != 1 \|\| outputs.size() != 10)` | HEF с NMS имеет 1 выход — требование несовместимо с заявленной целью |
| 5 | цикл по выходам | `if (outputs.size() < 1 \|\| outputs.size() > 2) throw` | проверяется размер вектора, а не число детекций; при 10 выходах условие всегда истинно → конструктор падает всегда |
| 6 | `postprocess()` | повторное чтение уже прочитанных в `infer()` потоков | второй кадр/дедлок, данные из `buffers` игнорируются |
| 7 | формат выхода | `user_buffer_format.order = NHWC` применён и к NMS-выходу | у NMS свой порядок (`HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS`); NHWC ломает разбор |
| 8 | разбор | `det_size = 6`, `data[i*6+6+c]` | чтение за границей записи на последней детекции; коэффициенты масок лежат не там |
| 9 | маски | `make_masks(std::vector<Det>{obj}, proto)` | маска пишется во временный вектор, `obj.mask` остаётся пустой |
| 10 | `ids`, `Key` | ключ `{height, features}` без `width` | у NMS-выходов формы вырождены → коллизии ключей |
| 11 | `main` | `HailoModel model(argv[1])` | параметр `nms_config_path` не передаётся: вся NMS-ветка мертва |

### Неверная модель формата

Документированный формат выхода с CPU-NMS (`HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS`):

```
для каждого класса: float32 count, затем count × hailo_bbox_float32_t{y_min, x_min, y_max, x_max, score}
```

Класс определяется позицией блока, отдельного поля класса нет. Разбор «6 float на
детекцию» (`x1, y1, x2, y2, score, cls`) не соответствует ни этому формату, ни
`HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE` (`uint16 count` + `hailo_detection_t{y_min, x_min,
y_max, x_max, score, class_id}`). Второй порог — нормализация координат: боксы приходят
в `[0, 1]` либо в пикселях, это надо проверять по данным, а не угадывать.

## Что сделано вместо этого

NMS-бэкенд удалён: для seg он неприменим, а для детекции задача не ставилась.
Разбор сохранён здесь и в `alls/yolov8s_detection_nms.alls` — это эталон того, как
NMS выглядит правильно (скопировано с `hailo_model_zoo/cfg/alls/generic/yolov8s.alls`).

Код разделён по архитектуре головы, без общего декодера:

- `yolo26s_seg.cpp` — YOLO26s-seg: 4 канала бокса (нет DFL), sigmoid на хосте.
- `yolo11s_seg.cpp` — YOLO11s-seg: 64 канала бокса (DFL), `decode()` из `common.hpp`.
- `hailo_setup.hpp` — общая обвязка HailoRT: конфигурация, потоки, обязательная
  активация, параллельное чтение выходов.
- `alls/yolo26s-seg.alls` — базовый model script для seg, с комментариями по слоям.

Попутно исправлено в общем коде:

- `annotate()` падал на пустой маске (`cv::Mat::at` по пустой матрице) — теперь детекция
  без масок просто не заливается.
- `make_masks()` принимает указатель на прототипы: копия 160×160×32 (3 МБ) на каждый кадр
  больше не нужна; добавлена проверка размера буфера и числа коэффициентов.
- `media.hpp` выбирает кодек по расширению выхода: `.avi` → MJPG, иначе mp4v
  (раньше mp4v писался и в `.avi`).
