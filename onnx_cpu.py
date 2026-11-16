"""Запуск: python onnx_cpu.py model.onnx input.jpg output.jpg"""
import sys
import numpy as np
import onnxruntime as ort
from common import process_file, onnx_heads

NUM_THREADS = -1

class OnnxModel:
    def __init__(self, model_path):
        # Явно используем CPU и один поток: сравнение с C++ станет воспроизводимее.
        options = ort.SessionOptions()
        options.intra_op_num_threads = NUM_THREADS  # для сравнения; изменяйте в обоих языках
        options.inter_op_num_threads = NUM_THREADS
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
        # Не пытаемся угадывать формат входа другой модели.
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or inputs[0].type != 'tensor(float)':
            raise ValueError('Ожидается вход ONNX float32 [1,3,640,640]')
        self.input_name = inputs[0].name

    def prepare(self, rgb):
        # ONNX ожидает float RGB NCHW 0…1, HEF — uint8 NHWC 0…255.
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255

    def infer(self, data):
        # None запрашивает все выходы модели; декодер далее проверяет их формы.
        return self.session.run(None, {self.input_name: data})

    def decode(self, result):
        return onnx_heads(result)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Использование: python onnx_cpu.py model.onnx input output')
    print(f'Число потоков: {NUM_THREADS}')
    process_file(sys.argv[2], sys.argv[3], OnnxModel(sys.argv[1]))
