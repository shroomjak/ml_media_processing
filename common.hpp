#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int S = 640;
// Геометрия исходного изображения и его размещения внутри letterbox 640×640.
struct Geo {
    int left, top, nw, nh, w, h;
};
// Один объект: рамка в координатах 640×640, класс, оценка, коэффициенты и маска.
struct Det {
    float x1, y1, x2, y2, score;
    int cls;
    std::vector<float> coeff;
    cv::Mat mask;
};
using Key = std::pair<int, int>; // размер сетки и количество каналов

// Подготовка: 1 или 3 канала → RGB 640×640; нормализация /255 внутри HEF.
cv::Mat prepare(const cv::Mat &src, Geo &g) {
    g.w = src.cols;
    g.h = src.rows;
    // Сохраняем пропорции; для квадратного 1000×1000 отступы равны нулю.
    double scale = std::min(double(S) / g.w, double(S) / g.h);
    g.nw = std::max(1, int(std::nearbyint(g.w * scale)));
    g.nh = std::max(1, int(std::nearbyint(g.h * scale)));
    g.left = (S - g.nw) / 2;
    g.top = (S - g.nh) / 2;
    cv::Mat source;
    if (src.channels() == 1)
        cv::cvtColor(src, source, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 3)
        source = src;
    else if (src.channels() == 4)
        cv::cvtColor(src, source, cv::COLOR_BGRA2BGR);
    else
        throw std::runtime_error("Unsupported channel count");
    // Значение поля 114 должно совпадать с обучением и калибровкой.
    cv::Mat canvas(S, S, CV_8UC3, cv::Scalar(114, 114, 114)), scaled, rgb;
    cv::resize(source, scaled, cv::Size(g.nw, g.nh));
    scaled.copyTo(canvas(cv::Rect(g.left, g.top, g.nw, g.nh)));
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}
// Ограничение аргумента нужно для численной устойчивости exp.
float sig(float x) {
    x = std::max(-80.f, std::min(80.f, x));
    return 1.f / (1.f + std::exp(-x));
}
// IoU: площадь пересечения, делённая на площадь объединения двух рамок.
float overlap(const Det &a, const Det &b) {
    float w = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float h = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float inter = w * h, aa = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float bb = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (aa + bb - inter + 1e-7f);
}
// NMS отдельно для каждого класса; оставляем не больше 50 объектов.
std::vector<Det> select_nms(std::vector<Det> all) {
    std::sort(all.begin(), all.end(), [](const Det &a, const Det &b) { return a.score > b.score; });
    // Ограничение кандидатов сокращает стоимость попарного сравнения рамок.
    if (all.size() > 300)
        all.resize(300);
    std::vector<Det> keep;
    for (const auto &a : all) {
        bool suppressed = false;
        for (const auto &b : keep)
            if (a.cls == b.cls && overlap(a, b) > 0.45f) {
                suppressed = true;
                break;
            }
        if (!suppressed)
            keep.push_back(a);
        if (keep.size() >= 50)
            break;
    }
    return keep;
}
// Читаем выходы HEF и декодируем расстояния DFL из 16 бинов.
std::vector<Det> decode(const std::map<Key, std::vector<float>> &t) {
    std::vector<Det> all;
    for (int s : {20, 40, 80}) {
        int stride = S / s;
        // Для каждого масштаба отдельно: 64 DFL, 2 оценки классов, 32 коэффициента.
        const auto &box = t.at({s, 64});
        const auto &scores = t.at({s, 2});
        const auto &coeff = t.at({s, 32});
        for (int n = 0; n < s * s; n++) {
            int cls = 0;
            float score = -1e30f;
            for (int c = 0; c < 2; c++) {
                float v = scores[n * 2 + c];
                if (v > score) {
                    score = v;
                    cls = c;
                }
            }
            if (score < 0.25f)
                continue;
            // Каждая сторона: softmax по 16 бинам и математическое ожидание.
            // Вычитание максимума перед exp защищает от переполнения.
            float d[4];
            for (int k = 0; k < 4; k++) {
                const float *v = &box[n * 64 + k * 16];
                float clipped[16];
                for (int j = 0; j < 16; j++)
                    clipped[j] = std::clamp(v[j], -50.f, 50.f);
                v = clipped;
                float m = *std::max_element(v, v + 16), sum = 0, weighted = 0;
                for (int b = 0; b < 16; b++) {
                    float e = std::exp(v[b] - m);
                    sum += e;
                    weighted += e * b;
                }
                d[k] = weighted / sum * stride;
            }
            // Центр ячейки и расстояния переводим в xyxy в пространстве 640×640.
            float x = (n % s + 0.5f) * stride, y = (n / s + 0.5f) * stride;
            Det a{std::clamp(x - d[0], 0.f, float(S)),
                  std::clamp(y - d[1], 0.f, float(S)),
                  std::clamp(x + d[2], 0.f, float(S)),
                  std::clamp(y + d[3], 0.f, float(S)),
                  score,
                  cls,
                  {}};
            a.coeff.assign(coeff.begin() + n * 32, coeff.begin() + (n + 1) * 32);
            all.push_back(std::move(a));
        }
    }
    return select_nms(std::move(all));
}

// Маска объекта — сигмоида линейной комбинации 32 прототипов.
void make_masks(std::vector<Det> &ds, const std::vector<float> &proto) {
    for (auto &d : ds) {
        d.mask = cv::Mat(160, 160, CV_32FC1);
        for (int y = 0; y < 160; y++)
            for (int x = 0; x < 160; x++) {
                float v = 0;
                // HWC: 32 числа каждого пикселя расположены подряд в памяти.
                const float *p = &proto[(y * 160 + x) * 32];
                for (int c = 0; c < 32; c++)
                    v += p[c] * d.coeff[c];
                d.mask.at<float>(y, x) = sig(v);
            }
    }
}
// Возвращаем маску и рамку в систему координат исходного кадра.
void annotate(cv::Mat &image, const Geo &g, const std::vector<Det> &detections) {
    for (const auto &d : detections) {
        cv::Mat full, mask;
        // Убираем поля letterbox перед масштабированием к исходному изображению.
        cv::resize(d.mask, full, cv::Size(S, S));
        cv::resize(full(cv::Rect(g.left, g.top, g.nw, g.nh)), mask, image.size());
        int x1 = std::clamp(int(std::floor((d.x1 - g.left) * g.w / g.nw)), 0, g.w);
        int y1 = std::clamp(int(std::floor((d.y1 - g.top) * g.h / g.nh)), 0, g.h);
        int x2 = std::clamp(int(std::ceil((d.x2 - g.left) * g.w / g.nw)), 0, g.w);
        int y2 = std::clamp(int(std::ceil((d.y2 - g.top) * g.h / g.nh)), 0, g.h);
        if (x2 <= x1 || y2 <= y1)
            continue;
        cv::Vec3b color((37 * d.cls + 67) % 256, (17 * d.cls + 149) % 256, (97 * d.cls + 83) % 256);
        // Накладываем маску только внутри рамки, смешивая цвет с исходным пикселем.
        for (int y = y1; y < y2; y++)
            for (int x = x1; x < x2; x++)
                if (mask.at<float>(y, x) > 0.5f) {
                    auto &px = image.at<cv::Vec3b>(y, x);
                    for (int c = 0; c < 3; c++)
                        px[c] = static_cast<uchar>(0.6f * px[c] + 0.4f * color[c]);
                }
        cv::Scalar col(color[0], color[1], color[2]);
        cv::rectangle(image, cv::Point(x1, y1),
                      cv::Point(std::min(x2, g.w - 1), std::min(y2, g.h - 1)), col, 2);
        cv::putText(image, "id:" + std::to_string(d.cls) + " " + cv::format("%.2f", d.score),
                    cv::Point(x1, std::max(15, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 2,
                    cv::LINE_AA);
    }
}
