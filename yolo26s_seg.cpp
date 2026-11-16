// ============================================================================
//  YOLO26s-seg на Hailo: самостоятельная программа (один файл, без общих заголовков).
//
//  Что делает: читает фото или видео, прогоняет кадр через HEF на Hailo, рисует
//  маски/рамки/классы и пишет результат в файл того же формата.
//
//  Что лежит в HEF и почему так:
//    Hailo DFC собирает NMS только для детекционных голов (и отдельно для yolov5_seg),
//    поэтому для головы YOLO26-seg HEF отдаёт сырые тензоры головы — 10 выходов:
//      80x80x4  80x80xNC  80x80x32    (страйд 8)
//      40x40x4  40x40xNC  40x40x32    (страйд 16)
//      20x20x4  20x20xNC  20x20x32    (страйд 32)
//      160x160x32                     (прототипы масок)
//    Четыре канала бокса — не ошибка: у YOLO26 reg_max = 1, DFL-слоя нет, вместо
//    64 бинов приходят четыре расстояния (left, top, right, bottom) от центра ячейки.
//    NMS и сборка масок выполняются здесь, на CPU.
//
//  Сборка:  g++ -std=c++17 -O2 yolo26s_seg.cpp -o yolo26s_seg $(pkg-config --cflags --libs opencv4) -lhailort
//  Запуск:  ./yolo26s_seg --inspect model.hef          (формы выходов, плата не нужна)
//           ./yolo26s_seg model.hef in.jpg out.jpg
//           ./yolo26s_seg model.hef in.mp4 out.mp4
// ============================================================================
#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace hailort;

// ============================================================================
//  1. Параметры модели — единственное, что правится вручную.
//     Значения берутся из .alls и из вывода `--inspect model.hef`.
// ============================================================================
constexpr int IMG = 640;                 // сторона входа сети
constexpr int NC = 80;                   // классов в модели (COCO = 80)
constexpr int NM = 32;                   // прототипов масок
constexpr int PROTO = IMG / 4;           // сторона карты прототипов: 160
// true, если в .alls есть change_output_activation(<cls_conv>, sigmoid) для трёх
// классовых ветвей. Иначе ветви выдают логиты, и sigmoid применяется ниже.
constexpr bool SIGMOID_IN_HEF = false;

// Пороги постобработки. CONF задан для вероятностей (после sigmoid), а не для логитов.
constexpr float CONF = 0.25f;            // ниже этого объект отбрасывается до NMS
constexpr float NMS_IOU = 0.45f;         // дубли одного класса с IoU больше — подавляются
constexpr size_t NMS_CANDIDATES = 300;   // сколько лучших сравнивать попарно
constexpr size_t MAX_DET = 50;           // сколько объектов оставить в кадре

// ============================================================================
//  2. Простые типы данных (не классы с поведением).
// ============================================================================
// Геометрия letterbox: как кадр лёг в квадрат IMG x IMG.
struct Geo {
    int left, top;  // отступы полей
    int nw, nh;     // размеры картинки внутри квадрата
    int w, h;       // размеры исходного кадра
};

// Один найденный объект: рамка в координатах 640x640, класс, оценка, коэффициенты
// маски (NM чисел) и сама маска 160x160 (появляется после сборки масок).
struct Det {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0, score = 0;
    int cls = 0;
    std::vector<float> coeff;
    cv::Mat mask;
};

// Выходы одного масштаба: три указателя на уже прочитанные буферы HailoRT.
// Указатели живут до следующего infer(), поэтому копировать их не нужно.
struct Head {
    int size = 0;                 // сторона сетки: 80, 40 или 20
    const float *box = nullptr;   // 4 канала
    const float *cls = nullptr;   // NC каналов
    const float *coeff = nullptr; // NM каналов
};

// ============================================================================
//  3. HailoRT: проверка статусов и владение ресурсами.
// ============================================================================
// Все функции HailoRT возвращают Expected<T>: значение доступно только после проверки.
// Без проверки ошибка конфигурации выглядит как «пустые детекции», а не как ошибка.
template <typename T> T take(Expected<T> &&result, const char *what) {
    if (!result)
        throw std::runtime_error(std::string(what) + " failed, status=" +
                                 std::to_string(result.status()));
    return result.release();
}

// Выход NMS в HEF означает, что при компиляции боковые и классовые ветви были заменены
// постпроцессом, а ветви коэффициентов масок и прототипов потеряны: сопоставить маски
// с объектами нечем. Такой HEF для сегментации не годится — сообщаем это явно.
bool has_nms_output(const std::vector<OutputVStream> &outputs) {
    for (const auto &output : outputs) {
        const auto order = output.get_info().format.order;
        const bool nms = order == HAILO_FORMAT_ORDER_HAILO_NMS ||
                         order == HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP ||
                         order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS ||
                         order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE ||
                         order == HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK;
        if (nms || std::string(output.name()).find("nms") != std::string::npos)
            return true;
    }
    return false;
}

// Печать потоков HEF без платы: Hef::create() читает только файл.
// Это первый шаг для незнакомой модели — из этих форм заполняются константы выше.
void inspect(const char *path) {
    auto hef = take(Hef::create(path), "Hef::create");
    for (const auto &info : take(hef.get_input_vstream_infos(), "input infos"))
        std::cout << "IN  " << info.name << ' ' << info.shape.height << 'x' << info.shape.width
                  << 'x' << info.shape.features << '\n';
    for (const auto &info : take(hef.get_output_vstream_infos(), "output infos"))
        std::cout << "OUT " << info.name << ' ' << info.shape.height << 'x' << info.shape.width
                  << 'x' << info.shape.features << '\n';
}

// Владение устройством, сетью, потоками и приёмными буферами.
// Это структура данных с одним конструктором, а не абстракция: никаких методов
// «сделай инференс» — последовательность вызовов видна в цикле кадра.
struct Hailo {
    std::unique_ptr<VDevice> device;
    Hef hef;  // у Hef нет конструктора по умолчанию, поэтому инициализация в списке
    std::shared_ptr<ConfiguredNetworkGroup> group;
    std::unique_ptr<ActivatedNetworkGroup> active;
    std::vector<InputVStream> inputs;
    std::vector<OutputVStream> outputs;
    std::vector<std::vector<float>> buffers;  // по одному на выход, float32
    cv::Mat input;                            // RGB IMG x IMG, uint8

    explicit Hailo(const char *path)
        : device(take(VDevice::create(), "VDevice::create")),
          hef(take(Hef::create(path), "Hef::create")) {
        auto params = take(device->create_configure_params(hef), "create_configure_params");
        auto groups = take(device->configure(hef, params), "configure");
        if (groups.size() != 1)
            throw std::runtime_error("HEF must contain exactly one network group, got " +
                                     std::to_string(groups.size()));
        group = groups.front();

        // Вход — uint8 (сеть скомпилирована под 0..255), выходы — float32.
        // Порядок NHWC задаётся явно: декодер читает тензор как (h, w, c) подряд.
        auto in_params = take(group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8,
                                                               HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                               HAILO_DEFAULT_VSTREAM_QUEUE_SIZE),
                              "input params");
        auto out_params = take(group->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32,
                                                                 HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                                 HAILO_DEFAULT_VSTREAM_QUEUE_SIZE),
                               "output params");
        for (auto &item : in_params)
            item.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        for (auto &item : out_params)
            item.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;

        inputs = take(VStreamsBuilder::create_input_vstreams(*group, in_params), "input streams");
        outputs = take(VStreamsBuilder::create_output_vstreams(*group, out_params), "output streams");
        if (inputs.size() != 1)
            throw std::runtime_error("Expected one input stream, got " +
                                     std::to_string(inputs.size()));
        const auto shape = inputs[0].get_info().shape;
        if (shape.height != IMG || shape.width != IMG || shape.features != 3 ||
            inputs[0].get_frame_size() != size_t(IMG) * IMG * 3)
            throw std::runtime_error("Unexpected input shape: " + std::to_string(shape.height) +
                                     "x" + std::to_string(shape.width) + "x" +
                                     std::to_string(shape.features));

        // Без активации write/read не работают; активированная группа живёт всю сессию.
        active = take(group->activate(), "activate network group");

        // Приёмные буферы создаются один раз: их адреса передаются в read().
        buffers.resize(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i)
            buffers[i].resize(outputs[i].get_frame_size() / sizeof(float));
    }
};

// ============================================================================
//  4. Подготовка кадра: letterbox в IMG x IMG и BGR -> RGB.
// ============================================================================
// Пропорции сохраняются, поля заполняются 114 (значение из обучения и калибровки).
// Нормализация (деление на 255) сделана внутри HEF командой normalization в .alls.
cv::Mat letterbox(const cv::Mat &src, Geo &g) {
    g.w = src.cols;
    g.h = src.rows;
    if (g.w <= 0 || g.h <= 0)
        throw std::runtime_error("Empty frame");
    const double scale = std::min(double(IMG) / g.w, double(IMG) / g.h);
    g.nw = std::max(1, int(std::nearbyint(g.w * scale)));
    g.nh = std::max(1, int(std::nearbyint(g.h * scale)));
    g.left = (IMG - g.nw) / 2;
    g.top = (IMG - g.nh) / 2;

    cv::Mat bgr;
    if (src.channels() == 1)
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 3)
        bgr = src;
    else if (src.channels() == 4)
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    else
        throw std::runtime_error("Unsupported channel count: " + std::to_string(src.channels()));

    cv::Mat canvas(IMG, IMG, CV_8UC3, cv::Scalar(114, 114, 114)), scaled, rgb;
    cv::resize(bgr, scaled, cv::Size(g.nw, g.nh));
    scaled.copyTo(canvas(cv::Rect(g.left, g.top, g.nw, g.nh)));
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

// ============================================================================
//  5. Разбор выходов HEF: привязка по формам.
// ============================================================================
// Роли выходов определяются по числу каналов, а не по позиции: порядок выходов задаёт
// DFC, а имена конвов зависят от парсера конкретного HEF.
// Проверка «плоский NHWC» обязательна: если размер кадра не равен h*w*c*4, то данные
// лежат иначе и разбор по индексам даст мусор.
void resolve_outputs(const Hailo &hailo, std::vector<Head> &heads, const float *&proto) {
    const auto &outs = hailo.outputs;
    if (outs.size() != 10)
        throw std::runtime_error(
            has_nms_output(outs)
                ? "HEF contains an NMS output: боковые и классовые ветви заменены NMS, а ветви "
                  "коэффициентов масок и прототипов при такой компиляции теряются — сегментация "
                  "в этом HEF невозможна. Уберите nms_postprocess из .alls."
                : "yolo26s-seg HEF must have 10 outputs (9 head + protos), got " +
                      std::to_string(outs.size()) + ". Run --inspect.");

    heads.assign(3, Head{});
    proto = nullptr;
    for (size_t i = 0; i < outs.size(); ++i) {
        const auto shape = outs[i].get_info().shape;
        const size_t frame = outs[i].get_frame_size() / sizeof(float);
        if (size_t(shape.height) * shape.width * shape.features != frame)
            throw std::runtime_error("Output " + outs[i].name() + " is not a plain NHWC tensor");
        if (shape.height != shape.width)
            throw std::runtime_error("Unexpected non-square output " + outs[i].name());

        const float *data = hailo.buffers[i].data();
        // Прототипы — единственный выход с сеткой PROTO и NM каналами.
        if (int(shape.height) == PROTO && int(shape.features) == NM) {
            proto = data;
            continue;
        }
        // Остальные три сетки: 80, 40, 20 — это страйды 8, 16, 32.
        const int size = int(shape.height);
        const int index = size == IMG / 8 ? 0 : size == IMG / 16 ? 1 : size == IMG / 32 ? 2 : -1;
        if (index < 0)
            throw std::runtime_error("Unexpected output size " + std::to_string(size) + " in " +
                                     outs[i].name());
        Head &head = heads[index];
        head.size = size;
        // Бокс — 4 канала (нет DFL), коэффициенты — NM, остальное классы.
        if (int(shape.features) == 4) {
            if (head.box)
                throw std::runtime_error("Duplicate box output at scale " + std::to_string(size));
            head.box = data;
        } else if (int(shape.features) == NM) {
            if (head.coeff)
                throw std::runtime_error("Duplicate coeff output at scale " + std::to_string(size));
            head.coeff = data;
        } else {
            if (head.cls)
                throw std::runtime_error("Duplicate cls output at scale " + std::to_string(size));
            // При NC == 4 или NC == NM роли неразличимы — это надо править вручную.
            if (int(shape.features) != NC)
                throw std::runtime_error("Class channels " + std::to_string(shape.features) +
                                         " do not match NC=" + std::to_string(NC));
            head.cls = data;
        }
    }
    for (const Head &head : heads)
        if (!head.box || !head.cls || !head.coeff)
            throw std::runtime_error("Incomplete outputs for scale " + std::to_string(head.size));
    if (!proto)
        throw std::runtime_error("No mask protos output " + std::to_string(PROTO) + "x" +
                                 std::to_string(PROTO) + "x" + std::to_string(NM));
}

// ============================================================================
//  6. Декодирование головы: без DFL, бокс — четыре расстояния от центра ячейки.
// ============================================================================
std::vector<Det> decode(const std::vector<Head> &heads) {
    std::vector<Det> all;
    for (const Head &head : heads) {
        const int stride = IMG / head.size;  // 640/80 = 8, 640/40 = 16, 640/20 = 32
        const size_t cells = size_t(head.size) * head.size;
        for (size_t n = 0; n < cells; ++n) {
            // Лучший класс — полным перебором: NC невелико, это дешевле сортировки.
            int cls = 0;
            float score = -1e30f;
            for (int c = 0; c < NC; ++c) {
                // Каналы идут подряд для каждой ячейки: индекс n * NC + c.
                float v = head.cls[n * NC + c];
                if (!SIGMOID_IN_HEF) {
                    // Логиты: ограничиваем аргумент, чтобы не переполнить exp.
                    const float x = std::clamp(v, -80.f, 80.f);
                    v = 1.f / (1.f + std::exp(-x));
                }
                if (v > score) {
                    score = v;
                    cls = c;
                }
            }
            if (score < CONF)
                continue;

            const float *b = &head.box[n * 4];
            const float ax = float(n % head.size) + 0.5f;  // центр ячейки по X
            const float ay = float(n / head.size) + 0.5f;  // центр ячейки по Y
            Det d;
            // Клиппинг к IMG x IMG: за пределы letterbox объект выходить не должен,
            // а вырожденный бокс на краю (x1 == x2) не подавит соседей — у него IoU 0.
            d.x1 = std::clamp((ax - b[0]) * stride, 0.f, float(IMG));
            d.y1 = std::clamp((ay - b[1]) * stride, 0.f, float(IMG));
            d.x2 = std::clamp((ax + b[2]) * stride, 0.f, float(IMG));
            d.y2 = std::clamp((ay + b[3]) * stride, 0.f, float(IMG));
            d.score = score;
            d.cls = cls;
            // Коэффициенты масок лежат так же, как каналы классов: n * NM + c.
            d.coeff.assign(&head.coeff[n * NM], &head.coeff[n * NM] + NM);
            all.push_back(std::move(d));
        }
    }
    return all;
}

// ============================================================================
//  7. NMS на CPU: подавление дублей по боксам внутри каждого класса.
// ============================================================================
// IoU = площадь пересечения / площадь объединения. max(0, ...) не даёт отрицательных
// размеров после клиппинга, eps защищает от деления на ноль у двух вырожденных боксов.
float iou(const Det &a, const Det &b) {
    const float w = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float h = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = w * h;
    const float aa = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float bb = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (aa + bb - inter + 1e-7f);
}

// Классы не подавляют друг друга: два объекта разных классов с одинаковыми боксами
// сохраняются оба. Иначе на перекрывающихся объектах один класс «съедал» бы другой.
std::vector<Det> nms(std::vector<Det> all) {
    std::sort(all.begin(), all.end(), [](const Det &a, const Det &b) { return a.score > b.score; });
    // Ограничение кандидатов: 8400 якорей и попарное сравнение несовместимы.
    if (all.size() > NMS_CANDIDATES)
        all.resize(NMS_CANDIDATES);
    std::vector<Det> keep;
    for (const auto &a : all) {
        bool suppressed = false;
        for (const auto &b : keep)
            if (a.cls == b.cls && iou(a, b) > NMS_IOU) {
                suppressed = true;
                break;
            }
        if (!suppressed)
            keep.push_back(a);
        if (keep.size() >= MAX_DET)
            break;
    }
    return keep;
}

// ============================================================================
//  8. Маски: mask = sigmoid(coeff · proto), по 32 прототипам на каждый пиксель.
// ============================================================================
// Считается только для объектов, переживших NMS: 32 умножения на пиксель 160x160
// стоят недёшево, а маски подавленных дублей всё равно не нужны.
void build_masks(std::vector<Det> &objects, const float *proto) {
    for (auto &d : objects) {
        if (d.coeff.size() != size_t(NM))
            throw std::runtime_error("Unexpected number of mask coefficients");
        d.mask = cv::Mat(PROTO, PROTO, CV_32FC1);
        for (int y = 0; y < PROTO; ++y)
            for (int x = 0; x < PROTO; ++x) {
                // HWC: NM чисел каждого пикселя прототипов лежат подряд.
                const float *p = &proto[(size_t(y) * PROTO + x) * NM];
                float v = 0;
                for (int c = 0; c < NM; ++c)
                    v += p[c] * d.coeff[c];
                v = std::clamp(v, -80.f, 80.f);
                d.mask.at<float>(y, x) = 1.f / (1.f + std::exp(-v));
            }
    }
}

// ============================================================================
//  9. Отрисовка: маска внутри рамки, рамка, подпись.
// ============================================================================
// Маска живёт в 160x160, рамка — в 640x640 (letterbox), кадр — в своих координатах.
// Порядок преобразований: маска -> 640x640 -> вырезать поля -> размер кадра.
void annotate(cv::Mat &image, const Geo &g, const std::vector<Det> &objects) {
    for (const auto &d : objects) {
        cv::Mat mask;
        if (!d.mask.empty()) {
            cv::Mat full;
            cv::resize(d.mask, full, cv::Size(IMG, IMG));
            cv::resize(full(cv::Rect(g.left, g.top, g.nw, g.nh)), mask, image.size());
        }
        const int x1 = std::clamp(int(std::floor((d.x1 - g.left) * g.w / g.nw)), 0, g.w);
        const int y1 = std::clamp(int(std::floor((d.y1 - g.top) * g.h / g.nh)), 0, g.h);
        const int x2 = std::clamp(int(std::ceil((d.x2 - g.left) * g.w / g.nw)), 0, g.w);
        const int y2 = std::clamp(int(std::ceil((d.y2 - g.top) * g.h / g.nh)), 0, g.h);
        if (x2 <= x1 || y2 <= y1)
            continue;
        const cv::Vec3b color(uchar((37 * d.cls + 67) % 256), uchar((17 * d.cls + 149) % 256),
                              uchar((97 * d.cls + 83) % 256));
        // Заливка только внутри рамки: это и дешевле, и убирает ложные пиксели маски.
        if (!mask.empty())
            for (int y = y1; y < y2; ++y)
                for (int x = x1; x < x2; ++x)
                    if (mask.at<float>(y, x) > 0.5f) {
                        auto &px = image.at<cv::Vec3b>(y, x);
                        for (int c = 0; c < 3; ++c)
                            px[c] = uchar(0.6f * px[c] + 0.4f * color[c]);
                    }
        const cv::Scalar scalar(color[0], color[1], color[2]);
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(std::min(x2, g.w - 1), std::min(y2, g.h - 1)),
                      scalar, 2);
        cv::putText(image, "id:" + std::to_string(d.cls) + " " + cv::format("%.2f", d.score),
                    cv::Point(x1, std::max(15, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, scalar, 2,
                    cv::LINE_AA);
    }
}

// ============================================================================
//  10. Цикл по кадрам: чтение -> letterbox -> инференс -> декод -> отрисовка -> запись.
// ============================================================================
bool is_image(const std::string &path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    for (const char *e : {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        if (ext == e)
            return true;
    return false;
}

void process(Hailo &hailo, const char *input, const char *output) {
    // Выход в тот же файл затирал бы вход при последовательной записи.
    if (std::filesystem::absolute(input).lexically_normal() ==
        std::filesystem::absolute(output).lexically_normal())
        throw std::runtime_error("Input and output must differ");

    std::vector<Head> heads;
    const float *proto = nullptr;
    resolve_outputs(hailo, heads, proto);

    const bool image_mode = is_image(input);
    cv::VideoCapture capture;
    cv::VideoWriter writer;
    if (!image_mode) {
        capture.open(input);
        if (!capture.isOpened())
            throw std::runtime_error("Cannot open video: " + std::string(input));
    }

    double t_read = 0, t_prep = 0, t_infer = 0, t_post = 0, t_draw = 0, t_write = 0;
    int frames = 0;
    auto start = std::chrono::steady_clock::now();
    while (true) {
        start = std::chrono::steady_clock::now();
        cv::Mat source;
        if (image_mode)
            source = cv::imread(input, cv::IMREAD_UNCHANGED);
        else
            capture.read(source);
        if (source.empty()) {
            if (image_mode)
                throw std::runtime_error("Cannot read image: " + std::string(input));
            break;  // конец видео
        }
        if (source.depth() != CV_8U)
            throw std::runtime_error("Input must be uint8");
        t_read += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        start = std::chrono::steady_clock::now();
        Geo geo;
        cv::Mat rgb = letterbox(source, geo);
        // Матрица должна быть непрерывной: в Hailo уходит единый блок памяти.
        if (!rgb.isContinuous() || rgb.type() != CV_8UC3)
            throw std::runtime_error("Prepared frame must be continuous 640x640x3");
        hailo.input = rgb;
        t_prep += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        start = std::chrono::steady_clock::now();
        // Выходы читаются параллельно с записью кадра: последовательное чтение
        // блокирует конвейер на первом же потоке с неполными данными.
        std::vector<hailo_status> statuses(hailo.outputs.size(), HAILO_UNINITIALIZED);
        std::vector<std::thread> readers;
        for (size_t i = 0; i < hailo.outputs.size(); ++i)
            readers.emplace_back([&hailo, i, &statuses] {
                statuses[i] = hailo.outputs[i].read(MemoryView(hailo.buffers[i].data(),
                                                               hailo.buffers[i].size() * sizeof(float)));
            });
        const auto write_status =
            hailo.inputs[0].write(MemoryView(hailo.input.data, size_t(hailo.input.total()) * hailo.input.elemSize()));
        for (auto &thread : readers)
            thread.join();
        if (write_status != HAILO_SUCCESS)
            throw std::runtime_error("Input write failed, status=" + std::to_string(write_status));
        for (size_t i = 0; i < statuses.size(); ++i)
            if (statuses[i] != HAILO_SUCCESS)
                throw std::runtime_error("Read of output " + hailo.outputs[i].name() +
                                         " failed, status=" + std::to_string(statuses[i]));
        t_infer += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        start = std::chrono::steady_clock::now();
        auto objects = nms(decode(heads));
        if (!objects.empty())
            build_masks(objects, proto);
        t_post += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        start = std::chrono::steady_clock::now();
        cv::Mat painted = source.clone();
        if (painted.channels() == 1)
            cv::cvtColor(painted, painted, cv::COLOR_GRAY2BGR);
        annotate(painted, geo, objects);
        t_draw += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        start = std::chrono::steady_clock::now();
        if (image_mode) {
            if (!cv::imwrite(output, painted))
                throw std::runtime_error("Cannot write image: " + std::string(output));
        } else {
            if (!writer.isOpened()) {
                // FPS берём у исходника, кодек — по расширению выхода, иначе
                // контейнер и кодек не совпадут и файл не откроется.
                const double fps = capture.get(cv::CAP_PROP_FPS);
                if (!std::isfinite(fps) || fps <= 0)
                    throw std::runtime_error("Invalid FPS in source video");
                std::string ext = std::filesystem::path(output).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".avi")
                    writer.open(output, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, painted.size());
                else
                    writer.open(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, painted.size());
                if (!writer.isOpened())
                    throw std::runtime_error("Cannot create video: " + std::string(output));
            }
            writer.write(painted);
        }
        t_write += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        ++frames;
        if (image_mode)
            break;
    }
    if (frames == 0)
        throw std::runtime_error("No frames processed");
    capture.release();
    writer.release();

    const double n = frames;
    std::cout << "кадров: " << frames << ", мс/кадр: read " << t_read / n << ", preprocess " << t_prep / n
              << ", infer " << t_infer / n << ", postprocess " << t_post / n << ", draw " << t_draw / n
              << ", write " << t_write / n << '\n';
    std::cout << "результат: " << output << '\n';
}

// ============================================================================
//  11. Точка входа.
// ============================================================================
int main(int argc, char **argv) {
    try {
        cv::setNumThreads(1);  // одно число потоков OpenCV: замеры воспроизводимее
        if (argc == 3 && std::string(argv[1]) == "--inspect") {
            inspect(argv[2]);
            return 0;
        }
        if (argc != 4) {
            std::cerr << "Usage: yolo26s_seg model.hef input output\n"
                         "       yolo26s_seg --inspect model.hef\n";
            return 2;
        }
        Hailo hailo(argv[1]);
        process(hailo, argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
