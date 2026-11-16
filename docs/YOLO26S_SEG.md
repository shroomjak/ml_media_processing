# yolo26s-seg на Hailo: `.alls`, размерности и правка скриптов

## 1. Что лежит в HEF

NMS для сегментации Hailo в HEF не компилирует, поэтому HEF отдаёт сырые тензоры головы —
10 выходов при входе 640×640. NMS при этом не отменяется: он выполняется на CPU по
боксам (как и в Ultralytics), после него для оставшихся объектов собираются маски.

| Выход | Форма (h×w×c) | Роль |
|---|---|---|
| 1–3 | 80×80×4, 80×80×NC, 80×80×32 | страйд 8: бокс, классы, коэффициенты масок |
| 4–6 | 40×40×4, 40×40×NC, 40×40×32 | страйд 16 |
| 7–9 | 20×20×4, 20×20×NC, 20×20×32 | страйд 32 |
| 10 | 160×160×32 | прототипы масок |

`NC` — число классов (COCO = 80), 32 — число прототипов. Четыре канала бокса — не
опечатка: в YOLO26 `reg_max = 1`, DFL-слоя нет, вместо 64 бинов приходят четыре
расстояния `ltrb` от центра ячейки. Поэтому `common.hpp::decode()` (16 бинов) здесь
неприменим — он остался для `yolo11s_seg.cpp`.

Порядок выходов в HEF не гарантирован, поэтому `yolo26s_seg.cpp` привязывается к ролям по
размерностям (4 / NC / 32), а не по позиции.

Порядок обработки кадра: декод всех масштабов → `select_nms()` по боксам каждого класса
(`NMS_IOU`, `NMS_MAX` в `common.hpp`) → `make_masks()` только для оставшихся объектов.
Маски после NMS — это дешевле (32×160×160 на каждый объект) и, главное, они не
попадают в подавленные дубли.

## 2. Шаг 1: узнать фактические размерности

```bash
./build/yolo26s_seg --inspect yolo26s-seg.hef
# IN  input  640x640x3
# OUT ... 80x80x4 / 80x80x80 / 80x80x32 ...
```

Без платы, только чтение файла. Если выходов не 10 или формы не совпали с таблицей,
дальше идти нет смысла: модель собрана с другими end nodes.

## 3. Шаг 2: end nodes и пересборка `.alls`

End nodes задаются **при парсинге ONNX**, а не в `.alls`:

```python
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8")   # или hailo8l
hn, npz = runner.translate_onnx_model(
    "yolo26s-seg.onnx", "yolo26s_seg",
    end_node_names=[...],                 # 9 головных конвов + ветвь прототипов
    net_input_shapes={"images": [1, 3, 640, 640]},
)
runner.save_har("yolo26s_seg.har")
```

Список узлов печатает `tools/onnx_end_nodes.py` (нужен только `onnx`):

```bash
python3 tools/onnx_end_nodes.py yolo26s-seg.onnx
```

Он выдаёт готовый `end_node_names = [...]` и напоминает про ветвь прототипов — без неё
в HEF не будет тензора 160×160×32 и маски собрать не из чего.

Дальше оптимизация и компиляция:

```python
runner = ClientRunner(har="yolo26s_seg.har")
runner.load_model_script("alls/yolo26s-seg.alls")
runner.optimize(calib_dataset)        # uint8 RGB 0..255, letterbox pad 114, без /255
hef = runner.compile()
open("yolo26s-seg.hef", "wb").write(hef)
```

`alls/yolo26s-seg.alls` содержит только `normalization([0],[255])`, `model_optimization_flavor`
и `performance_param`. Команда `nms_postprocess` там отсутствует и добавлять её нельзя:
для seg DFC завершится ошибкой.

### sigmoid: в HEF или на CPU

Классовые ветви выдают логиты. Два взаимозаменяемых варианта:

1. **sigmoid на CPU** (по умолчанию, `SIGMOID_IN_HEF = false` в `yolo26s_seg.cpp`):
   ничего не добавлять в `.alls`. Плюс: выходной слой остаётся с полным диапазоном,
   минус: `NC` sigmoid на ячейку на CPU.
2. **sigmoid в HEF**: добавить в `.alls` три строки
   `change_output_activation(<cls_conv>, sigmoid)` и выставить `SIGMOID_IN_HEF = true`.
   Имена конвов берутся из списка слоёв HAR (после оптимизации):

   ```python
   for layer in runner.get_layer_names():   # или har['layers'] через h5py
       print(layer)
   ```

   Если включать sigmoid не хочется, но выход «упирается» в диапазон, есть третий путь:
   `quantization_param([conv_a, conv_b, conv_c], force_range_out=[0.0, 1.0])`.

## 4. Шаг 3: что менять в коде

В начале `yolo26s_seg.cpp` четыре константы — их значения берутся из `.alls` и `--inspect`:

```cpp
constexpr int NC = 80;                 // классов в модели
constexpr int NM = 32;                 // прототипов масок
constexpr int PROTO = 160;             // сторона карты прототипов: 640 / 4
constexpr bool SIGMOID_IN_HEF = false; // true, если в .alls есть change_output_activation
```

Остальное (letterbox, NMS, маски, отрисовка, запись файла) общее и править не нужно.

Если модель обучена на другом разрешении, менять надо не только `PROTO`: `S` в
`common.hpp` (letterbox и сетки) и страйды `S / head.size` в `yolo26s_seg.cpp`.

## 5. Шаг 4: сборка и запуск

```bash
cmake -S . -B build && cmake --build build -j
./build/yolo26s_seg yolo26s-seg.hef photo.jpg out.jpg
./build/yolo26s_seg yolo26s-seg.hef in.mp4 out.mp4
```

## 6. Типовые ошибки

| Симптом | Причина | Что делать |
|---|---|---|
| `must have 10 outputs`: `HEF contains an NMS output` | в `.alls` попал `nms_postprocess`; NMS заменил боковые и классовые ветви, а ветви масок и прототипов при этом потеряны | убрать `nms_postprocess` из `.alls`, NMS делать на хосте (см. `docs/REVIEW.md`) |
| `must have 10 outputs` без упоминания NMS | HEF собран с другими end nodes (например, один общий выход головы) | `--inspect`; пересобрать с 9 конвами + прототипы |
| `no 4-channel box output` / боксов нет | это HEF YOLOv8/YOLO11 (64 канала, DFL) | использовать `yolo11s_seg.cpp` |
| детекций нет совсем | sigmoid не там: логиты с порогом 0.25 | сверить `SIGMOID_IN_HEF` с `.alls` |
| боксы «в углу» / крошечные | порядок `ltrb` не совпал (модель отдаёт `cx,cy,w,h`) | пересобрать HEF без декода в графе или поменять формулу в `postprocess()` |
| маски пустые, рамки есть | нет выхода 160×160×32 | добавить ветвь прототипов в end nodes |
| `NC == NM` и роли перепутаны | 32 класса совпали с 32 прототипами | задать роли вручную по именам выходов в конструкторе |
| `HAILO_STREAM_NOT_ACTIVATED` | нет `activate()` | в этом репозитории активация внутри `Session`, ошибка означает другую сборку |

## 7. Чем `yolo26s_seg.cpp` отличается от `yolo11s_seg.cpp`

| | YOLO26s-seg | YOLO11s-seg |
|---|---|---|
| Каналы бокса | 4 (нет DFL) | 64 (DFL, 16 бинов) |
| Декод бокса | `(anchor ± d) * stride` | softmax по бинам + математическое ожидание |
| sigmoid | на CPU (по умолчанию) | в HEF (`change_output_activation`) |
| Декодер | внутри `yolo26s_seg.cpp` | `common.hpp::decode()` |

Общей «универсальной» ветки нет специально: разница в формате головы — это разные
формулы, и попытка угадать её по данным даёт молча неверные боксы вместо понятной ошибки.
