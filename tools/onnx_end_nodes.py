#!/usr/bin/env python3
"""Печатает end nodes головы YOLO-модели для DFC (hailo_sdk_client).

Зачем: при парсинге ONNX в HAR нужно явно указать, на каких узлах заканчивается граф.
Для YOLO26-seg это 9 головных конвов (по 3 на масштаб: бокс, классы, коэффициенты масок)
плюс ветвь прототипов масок. Имена зависят от экспорта, поэтому их лучше не угадывать.

Зависимость: onnx (`pip install onnx`). Ultralytics не нужен.

Пример:
    python3 tools/onnx_end_nodes.py yolo26s-seg.onnx
    python3 tools/onnx_end_nodes.py yolo26s-seg.onnx --stem /model.23
"""

import argparse
import sys

import onnx
from onnx import shape_inference


def channel_count(node, initializers):
    """Число выходных каналов свёртки: первый размер ядра весов."""
    if len(node.input) < 2:
        return None
    weight = initializers.get(node.input[1])
    if weight is None or len(weight.dims) < 4:
        return None
    return int(weight.dims[0])


def tensor_shapes(model):
    shapes = {}
    for value in list(model.graph.value_info) + list(model.graph.output):
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if dims:
            shapes[value.name] = dims
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="ONNX-файл модели")
    parser.add_argument("--stem", default="/model.23",
                        help="префикс головы в графе (у YOLO26 обычно /model.23)")
    parser.add_argument("--branch", default="cv2,cv3,cv4,proto",
                        help="подстроки имён ветвей: бокс, классы, коэффициенты масок, прототипы")
    args = parser.parse_args()

    model = onnx.load(args.model)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # инференс форм не критичен, каналы берём из весов
        print(f"shape inference failed (продолжаем): {exc}", file=sys.stderr)

    initializers = {t.name: t for t in model.graph.initializer}
    shapes = tensor_shapes(model)

    print("Входы сети:")
    for value in model.graph.input:
        dims = [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
        print(f"  {value.name}: {dims}")
    print("Выходы сети (как их видит onnx):")
    for value in model.graph.output:
        print(f"  {value.name}: {shapes.get(value.name)}")

    branches = [b.strip() for b in args.branch.split(",") if b.strip()]
    heads = []
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        name = node.output[0]
        if not name.startswith(args.stem):
            continue
        if not any(f"/{b}." in name for b in branches):
            continue
        heads.append((name, channel_count(node, initializers), shapes.get(name)))

    if not heads:
        print(f"\nНичего не найдено под префиксом {args.stem}. "
              f"Проверьте имена: возможно, голова называется иначе (см. список Conv выше).",
              file=sys.stderr)
        for node in model.graph.node:
            if node.op_type == "Conv" and any(f"/{b}." in node.output[0] for b in branches):
                print(f"  кандидат: {node.output[0]} каналы={channel_count(node, initializers)}")
        return 1

    print("\nГоловные конвы:")
    for name, channels, shape in heads:
        print(f"  {name}  channels={channels}  out_shape={shape}")

    # DFC сопоставляет выходы по именам, но читать список удобнее по ветвям и масштабам.
    def sort_key(name):
        parts = [p for p in name.replace(args.stem, "").split("/") if p]
        branch = parts[0].split(".")[0] if parts else ""
        try:
            scale = int(parts[0].split(".")[1])
        except (IndexError, ValueError):
            scale = 0
        return (branch, scale)

    heads.sort(key=lambda item: sort_key(item[0]))
    print("\nГотовый фрагмент для скрипта парсинга:")
    print("end_node_names = [")
    for name, _, _ in heads:
        print(f'    "{name}",')
    print("]")
    if not any("/proto" in name or "proto" in name.split("/")[-2:][0] for name, _, _ in heads):
        print("\nВетвь прототипов масок не найдена: без неё в HEF не будет тензора 160x160x32")
        print("и маски собрать не из чего. Добавьте её имя в список end nodes вручную.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
