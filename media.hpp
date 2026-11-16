#pragma once
#include "common.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <string>

using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;

// Среднее и СКО по всем кадрам, без исключения первого кадра.
void print_times(const std::map<std::string, std::vector<double>> &times) {
    for (const char *name : {"read", "preprocess", "infer", "postprocess", "draw", "write"}) {
        const auto &v = times.at(name);
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double variance = 0;
        for (double x : v)
            variance += (x - mean) * (x - mean);
        std::cout << name << ": " << mean << " ± " << std::sqrt(variance / v.size())
                  << " мс/кадр\n";
    }
}

// steady_clock измеряет интервалы монотонно, без влияния изменения системных часов.
void record(std::map<std::string, std::vector<double>> &times, const char *name, Time start) {
    times[name].push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
}

bool is_image(const std::string &path) {
    auto ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (const char *e : {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        if (ext == e)
            return true;
    return false;
}

// Backend предоставляет prepare_input, infer и postprocess; цикл общий для HEF и ONNX.
template <class Backend> void process_file(Backend &model, const char *input, const char *output) {
    if (std::filesystem::absolute(input).lexically_normal() ==
        std::filesystem::absolute(output).lexically_normal())
        throw std::runtime_error("Input and output must differ");
    // По расширению выбираем одно изображение или последовательность кадров видео.
    bool image_mode = is_image(input);
    cv::VideoCapture capture;
    cv::VideoWriter writer;
    if (!image_mode) {
        capture.open(input);
        if (!capture.isOpened())
            throw std::runtime_error("Cannot open video");
    }
    std::map<std::string, std::vector<double>> times;
    int count = 0;
    while (true) {
        auto start = Clock::now();
        cv::Mat source;
        if (image_mode)
            source = cv::imread(input, cv::IMREAD_UNCHANGED);
        else
            capture.read(source);
        if (source.empty()) {
            if (image_mode)
                throw std::runtime_error("Cannot read image");
            break;
        }
        record(times, "read", start);

        // Приводим grayscale к BGR для отрисовки и готовим RGB для модели.
        start = Clock::now();
        if (source.depth() != CV_8U)
            throw std::runtime_error("Input must be uint8");
        cv::Mat frame;
        if (source.channels() == 1)
            cv::cvtColor(source, frame, cv::COLOR_GRAY2BGR);
        else if (source.channels() == 4)
            cv::cvtColor(source, frame, cv::COLOR_BGRA2BGR);
        else if (source.channels() == 3)
            frame = source;
        else
            throw std::runtime_error("Unsupported number of channels");
        Geo geo;
        cv::Mat rgb = prepare(frame, geo);
        model.prepare_input(rgb);
        record(times, "preprocess", start);

        // Время infer включает runtime и передачу данных, а не только работу чипа.
        start = Clock::now();
        model.infer();
        record(times, "infer", start);
        start = Clock::now();
        auto objects = model.postprocess();
        record(times, "postprocess", start);
        start = Clock::now();
        cv::Mat painted = frame.clone();
        annotate(painted, geo, objects);
        record(times, "draw", start);

        start = Clock::now();
        if (image_mode) {
            if (!cv::imwrite(output, painted))
                throw std::runtime_error("Cannot write image");
        } else {
            // Первый кадр даёт размеры выходного видео; используем FPS исходника.
            if (!writer.isOpened()) {
                double fps = capture.get(cv::CAP_PROP_FPS);
                if (!std::isfinite(fps) || fps <= 0)
                    throw std::runtime_error("Invalid FPS");
                // Кодек должен поддерживаться контейнером выходного файла: формат выхода
                // задаёт пользователь, поэтому кодек выбирается по расширению.
                auto ext = std::filesystem::path(output).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".avi")
                    writer.open(output, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                                painted.size());
                else
                    writer.open(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                                painted.size());
                if (!writer.isOpened())
                    throw std::runtime_error("Cannot create video");
            }
            writer.write(painted);
        }
        record(times, "write", start);
        ++count;
        if (image_mode)
            break;
    }
    if (!count)
        throw std::runtime_error("No frames");
    // Закрытие writer завершает запись контейнера до вывода статистики.
    capture.release();
    writer.release();
    std::cout << "Время на кадр: среднее ± СКО\n";
    print_times(times);
    std::cout << "Кадров: " << count << " | Результат: " << output << '\n';
}
