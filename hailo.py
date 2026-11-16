"""Запуск: python hailo.py model.hef input.jpg output.jpg"""
import sys
from contextlib import ExitStack
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                            InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)
from common import process_file, raw_heads

class HailoModel:
    def __init__(self, model_path, stack):
        # HEF — уже скомпилированная модель; здесь мы не обучаем и не компилируем её.
        hef = HEF(model_path)
        inputs = hef.get_input_vstream_infos()
        if len(inputs) != 1 or tuple(inputs[0].shape) != (640, 640, 3):
            raise ValueError('Ожидается вход HEF 640×640×3')
        self.input_name = inputs[0].name
        # Сопоставление по форме верно только для оговорённой 2-классовой модели.
        expected = {(s, s, c) for s in (20, 40, 80) for c in (2, 32, 64)} | {(160, 160, 32)}
        infos = hef.get_output_vstream_infos()
        # Индексы потоков не считаем стабильными: ищем их по ожидаемой форме.
        # Совпадение формы не гарантирует семантику — сверяйте с вашим графом.
        self.names = {tuple(x.shape): x.name for x in infos}
        if len(infos) != 10 or set(self.names) != expected:
            raise ValueError('Нужны 10 выходов двухклассовой YOLO11-seg')
        for shape, name in self.names.items():
            print('OUTPUT', name, shape)
        # ExitStack сохраняет устройство, потоки и активацию до конца обработки файла.
        device = stack.enter_context(VDevice())
        config = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        group = device.configure(hef, config)[0]
        inputs = InputVStreamParams.make(group, format_type=FormatType.UINT8)
        # HailoRT вернёт де-квантизированные float32; вручную scale/zero-point не применяем.
        outputs = OutputVStreamParams.make(group, format_type=FormatType.FLOAT32)
        self.pipeline = stack.enter_context(InferVStreams(group, inputs, outputs))
        stack.enter_context(group.activate(group.create_params()))

    def prepare(self, rgb):
        return rgb[None, ...]  # Добавить batch=1: NHWC, 0…255; деление на 255 находится в HEF.

    def infer(self, data):
        return self.pipeline.infer({self.input_name: data})

    def decode(self, result):
        # Привести словарь имён HailoRT к словарю форм, который ожидает общий декодер.
        return raw_heads({shape: result[name] for shape, name in self.names.items()})

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Использование: python hailo.py model.hef input output')
    with ExitStack() as stack:
        process_file(sys.argv[2], sys.argv[3], HailoModel(sys.argv[1], stack))
