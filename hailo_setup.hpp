#pragma once
// Минимальная обвязка HailoRT для сырых выходов (без NMS в HEF):
//   VDevice -> HEF -> сеть -> vstreams -> activate() -> write/read.
// Всё, что нужно двум бэкендам: вход uint8 NHWC 640x640x3, выходы float32 NHWC.
#include "common.hpp"
#include <hailo/hailort.hpp>
#include <iostream>
#include <thread>
#include <vector>

// Expected<T> — обёртка HailoRT над значением и статусом: без проверки значение не достать.
template <typename T> T take(hailort::Expected<T> &&result, const char *what) {
    if (!result)
        throw std::runtime_error(std::string(what) + " failed, status=" + std::to_string(result.status()));
    return result.release();
}

// Печать потоков HEF без платы: Hef::create() читает только файл.
// Это первый шаг для незнакомого HEF — из этих форм заполняются константы в .cpp.
inline void inspect(const char *path) {
    auto hef = take(hailort::Hef::create(path), "Hef::create");
    for (const auto &info : take(hef.get_input_vstream_infos(), "input infos"))
        std::cout << "IN  " << info.name << " " << info.shape.height << 'x' << info.shape.width << 'x'
                  << info.shape.features << '\n';
    for (const auto &info : take(hef.get_output_vstream_infos(), "output infos"))
        std::cout << "OUT " << info.name << " " << info.shape.height << 'x' << info.shape.width << 'x'
                  << info.shape.features << '\n';
}

class Session {
  public:
    explicit Session(const char *path)
        : device_(take(hailort::VDevice::create(), "VDevice")),
          hef_(take(hailort::Hef::create(path), "HEF")) {
        auto params = take(device_->create_configure_params(hef_), "create_configure_params");
        auto groups = take(device_->configure(hef_, params), "configure");
        if (groups.size() != 1)
            throw std::runtime_error("Expected one network group");
        group_ = groups.front();

        // Вход — uint8 RGB, выходы — float32 в формате NHWC (декодер читает (h, w, c)).
        auto in_params = take(group_->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8,
                                                               HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                               HAILO_DEFAULT_VSTREAM_QUEUE_SIZE),
                              "input params");
        auto out_params = take(group_->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32,
                                                                  HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                                  HAILO_DEFAULT_VSTREAM_QUEUE_SIZE),
                               "output params");
        for (auto &item : in_params)
            item.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        for (auto &item : out_params)
            item.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        inputs_ = take(hailort::VStreamsBuilder::create_input_vstreams(*group_, in_params),
                       "input streams");
        outputs_ = take(hailort::VStreamsBuilder::create_output_vstreams(*group_, out_params),
                        "output streams");
        if (inputs_.size() != 1)
            throw std::runtime_error("Expected one input stream");
        const auto shape = inputs_[0].get_info().shape;
        if (shape.height != S || shape.width != S || shape.features != 3)
            throw std::runtime_error("Unexpected input shape");

        // Без активации write/read не работают; ActivatedNetworkGroup живёт всю сессию.
        activated_ = take(group_->activate(), "activate");

        buffers_.resize(outputs_.size());
        for (size_t i = 0; i < outputs_.size(); ++i)
            buffers_[i].resize(outputs_[i].get_frame_size() / sizeof(float));
    }

    const std::vector<hailort::OutputVStream> &outputs() const { return outputs_; }

    void prepare_input(const cv::Mat &rgb) {
        if (rgb.type() != CV_8UC3 || rgb.cols != S || rgb.rows != S || !rgb.isContinuous())
            throw std::runtime_error("Input must be continuous 640x640x3 uint8");
        input_ = rgb;
    }

    // Выходы читаются параллельно с записью кадра: последовательное чтение
    // блокирует конвейер на первом же потоке.
    void infer() {
        std::vector<hailo_status> statuses(outputs_.size(), HAILO_UNINITIALIZED);
        std::vector<std::thread> readers;
        for (size_t i = 0; i < outputs_.size(); ++i)
            readers.emplace_back([this, i, &statuses] {
                statuses[i] = outputs_[i].read(hailort::MemoryView(
                    buffers_[i].data(), buffers_[i].size() * sizeof(float)));
            });
        auto status = inputs_[0].write(
            hailort::MemoryView(input_.data, size_t(input_.total()) * input_.elemSize()));
        for (auto &thread : readers)
            thread.join();
        if (status != HAILO_SUCCESS)
            throw std::runtime_error("Input write failed: " + std::to_string(status));
        for (size_t i = 0; i < statuses.size(); ++i)
            if (statuses[i] != HAILO_SUCCESS)
                throw std::runtime_error("Output " + outputs_[i].name() +
                                         " read failed: " + std::to_string(statuses[i]));
    }

    // Готовый выход как (h, w, c): так он и лежит в буфере при порядке NHWC.
    const float *data(size_t i) const { return buffers_.at(i).data(); }

  private:
    std::unique_ptr<hailort::VDevice> device_;
    hailort::Hef hef_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> group_;
    std::unique_ptr<hailort::ActivatedNetworkGroup> activated_;
    std::vector<hailort::InputVStream> inputs_;
    std::vector<hailort::OutputVStream> outputs_;
    std::vector<std::vector<float>> buffers_;
    cv::Mat input_;
};
