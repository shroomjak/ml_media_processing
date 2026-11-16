#include "media.hpp"
#include <hailo/hailort.hpp>
#include <set>
#include <thread>
using namespace hailort;

template <typename T> void check(const T &result, const char *message) {
    if (!result)
        throw std::runtime_error(std::string(message) + ": " + std::to_string(result.status()));
}

// Владеет устройством, потоками и выходными буферами на всё время обработки.
class HailoModel {
    std::unique_ptr<VDevice> device;
    std::shared_ptr<ConfiguredNetworkGroup> group;
    std::vector<InputVStream> inputs;
    std::vector<OutputVStream> outputs;
    std::map<Key, size_t> ids;
    std::vector<std::vector<float>> buffers;
    cv::Mat rgb;

  public:
    explicit HailoModel(const char *path) {
        // Открываем Hailo и настраиваем его скомпилированным HEF.
        auto dev = VDevice::create();
        check(dev, "VDevice");
        device = dev.release();
        auto hef = Hef::create(path);
        check(hef, "HEF");
        auto config = device->create_configure_params(hef.value());
        check(config, "config");
        auto groups = device->configure(hef.value(), config.value());
        check(groups, "configure");
        if (groups->size() != 1)
            throw std::runtime_error("Expected one network group");
        group = groups->at(0);

        // На вход подаём uint8 RGB; HailoRT возвращает выходы как де-квантизированный float32.
        auto in_params = group->make_input_vstream_params({}, HAILO_FORMAT_TYPE_UINT8,
                                                          HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                          HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        auto out_params = group->make_output_vstream_params({}, HAILO_FORMAT_TYPE_FLOAT32,
                                                            HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                            HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        check(in_params, "input params");
        check(out_params, "output params");
        // Фиксируем порядок HWC в пользовательских буферах: декодер обращается именно так.
        for (auto &item : *in_params)
            item.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        for (auto &item : *out_params)
            item.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        auto in = VStreamsBuilder::create_input_vstreams(*group, *in_params);
        check(in, "input streams");
        inputs = in.release();
        auto out = VStreamsBuilder::create_output_vstreams(*group, *out_params);
        check(out, "output streams");
        outputs = out.release();
        if (inputs.size() != 1 || outputs.size() != 10)
            throw std::runtime_error("Expected 1 input and 10 outputs");
        auto input_shape = inputs[0].get_info().shape;
        if (input_shape.height != 640 || input_shape.width != 640 || input_shape.features != 3 ||
            inputs[0].get_frame_size() != 640 * 640 * 3)
            throw std::runtime_error("Unexpected input shape");

        // Сопоставляем потоки по форме, а не позиции; по одной форме нельзя доказать семантику.
        // Имена и роли должны быть проверены по графу и логам именно вашего HEF.
        std::set<Key> expected;
        for (int size : {20, 40, 80})
            for (int channels : {2, 32, 64})
                expected.insert({size, channels});
        expected.insert({160, 32});
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto shape = outputs[i].get_info().shape;
            Key key{int(shape.height), int(shape.features)};
            std::cout << "OUTPUT " << outputs[i].name() << " " << shape.height << "x" << shape.width
                      << "x" << shape.features << '\n';
            if (shape.height != shape.width || !expected.count(key) || ids.count(key) ||
                outputs[i].get_frame_size() !=
                    size_t(shape.height) * shape.width * shape.features * sizeof(float))
                throw std::runtime_error("Unexpected or duplicate output shape");
            ids[key] = i;
            buffers.emplace_back(outputs[i].get_frame_size() / sizeof(float));
        }
    }

    void prepare_input(const cv::Mat &image) {
        rgb = image;
    }

    void infer() {
        // Начинаем читать все выходы параллельно с записью кадра: последовательная работа
        // со многими потоками может заблокировать конвейер. Буферы остаются живы до join.
        // Каждый вызов HailoRT возвращает статус: ошибки нельзя игнорировать.
        std::vector<hailo_status> statuses(outputs.size(), HAILO_UNINITIALIZED);
        std::vector<std::thread> readers;
        for (size_t i = 0; i < outputs.size(); ++i)
            readers.emplace_back([&, i] {
                statuses[i] = outputs[i].read(
                    MemoryView(buffers[i].data(), buffers[i].size() * sizeof(float)));
            });
        auto status = inputs[0].write(MemoryView(rgb.data, rgb.total() * rgb.elemSize()));
        for (auto &thread : readers)
            thread.join();
        if (status != HAILO_SUCCESS)
            throw std::runtime_error("Hailo input write failed: " + std::to_string(status));
        for (auto s : statuses)
            if (s != HAILO_SUCCESS)
                throw std::runtime_error("Hailo output read failed: " + std::to_string(s));
    }

    std::vector<Det> postprocess() {
        // Копируем выходы в словарь, который использует общий декодер; это входит в замер.
        std::map<Key, std::vector<float>> tensors;
        for (auto &item : ids)
            tensors.emplace(item.first, buffers[item.second]);
        auto objects = decode(tensors); // sigmoid уже выполнен в HEF
        make_masks(objects, tensors.at({160, 32}));
        return objects;
    }
};

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: hailo_seg model.hef input output\n";
            return 2;
        }
        // Одно число потоков OpenCV во всех вариантах упрощает сравнение времени.
        cv::setNumThreads(1);
        HailoModel model(argv[1]);
        process_file(model, argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
