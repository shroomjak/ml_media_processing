// YOLO26s-seg на Hailo: инференс с NMS на CPU.
//
// Отличие от yolo11s_seg.cpp — только в голове: у YOLO26 reg_max = 1, DFL-слоя нет,
// поэтому бокс приходит четырьмя числами вместо 64 бинов. Всё остальное (letterbox,
// NMS по боксам, маски по коэффициентам, запись файла) то же самое.
//
// Порядок обработки кадра:
//   prepare() -> 10 сырых выходов -> декод (anchor ± расстояние) -> NMS -> маски -> отрисовка
//
// NMS здесь обязателен: в HEF его нет и быть не может для сегментации (docs/REVIEW.md).
#include "hailo_setup.hpp"
#include "media.hpp"

namespace {

// ---- Параметры модели: переносятся вручную из .alls и из `--inspect` ----
constexpr int NC = 80;     // классов в модели (COCO = 80)
constexpr int NM = 32;     // прототипов масок
constexpr int PROTO = 160; // сторона карты прототипов: 640 / 4
// true, если в .alls есть change_output_activation(<cls_conv>, sigmoid) для трёх
// классовых ветвей. Иначе ветви выдают логиты, и sigmoid делается здесь.
constexpr bool SIGMOID_IN_HEF = false;

// Выходы одного масштаба. Роли определяются по числу каналов, порядок не важен.
struct Head {
    int size = 0;             // сторона сетки: 80 / 40 / 20
    const float *box = nullptr;   // 4 канала: расстояния left, top, right, bottom
    const float *cls = nullptr;   // NC каналов
    const float *coeff = nullptr; // NM каналов
};

class Yolo26Seg {
    Session session_;
    std::vector<Head> heads_;     // в порядке появления выходов в HEF
    const float *proto_ = nullptr;

  public:
    explicit Yolo26Seg(const char *path) : session_(path) {
        const auto &outs = session_.outputs();
        if (outs.size() != 10)
            throw std::runtime_error(
                has_nms_output(outs)
                    ? "HEF contains an NMS output: боковые и классовые ветви заменены NMS, а ветви "
                      "коэффициентов масок и прототипов при такой компиляции теряются — "
                      "сегментация в этом HEF невозможна. Уберите nms_postprocess из .alls."
                    : "yolo26s-seg HEF must have 10 outputs (9 head + protos), got " +
                          std::to_string(outs.size()) + ". Run --inspect.");

        for (size_t i = 0; i < outs.size(); ++i) {
            const auto shape = outs[i].get_info().shape;
            // Проверка «плоский NHWC»: ниже данные читаются как (h, w, c) подряд.
            if (size_t(shape.height) * shape.width * shape.features !=
                outs[i].get_frame_size() / sizeof(float))
                throw std::runtime_error("Output " + outs[i].name() +
                                         " is not a plain NHWC tensor");
            if (shape.height != shape.width)
                throw std::runtime_error("Unexpected non-square output " + outs[i].name());
            // Прототипы — единственный выход с сеткой PROTO и NM каналами.
            if (shape.height == PROTO && shape.features == NM) {
                proto_ = session_.data(i);
                continue;
            }
            // Остальные три сетки: 80, 40, 20 — это страйды 8, 16, 32.
            const int index = shape.height == 80 ? 0 : shape.height == 40 ? 1 : shape.height == 20 ? 2 : -1;
            if (index < 0)
                throw std::runtime_error("Unexpected output size: " + outs[i].name());
            if (heads_.size() <= size_t(index))
                heads_.resize(index + 1);
            Head &head = heads_[index];
            head.size = int(shape.height);
            // Бокс — 4 канала (нет DFL), коэффициенты — NM, остальное классы.
            if (shape.features == 4)
                head.box = session_.data(i);
            else if (shape.features == NM)
                head.coeff = session_.data(i);
            else
                head.cls = session_.data(i); // NC каналов, любое число классов
        }
        for (const Head &head : heads_)
            if (!head.box || !head.cls || !head.coeff)
                throw std::runtime_error("Incomplete outputs for scale " + std::to_string(head.size));
        if (!proto_)
            throw std::runtime_error("No mask protos output " + std::to_string(PROTO) + "x" +
                                     std::to_string(PROTO) + "x" + std::to_string(NM));
    }

    void prepare_input(const cv::Mat &image) { session_.prepare_input(image); }
    void infer() { session_.infer(); }

    std::vector<Det> postprocess() {
        std::vector<Det> all;
        for (const Head &head : heads_) {
            const int stride = S / head.size; // 640/80 = 8, 640/40 = 16, 640/20 = 32
            const size_t cells = size_t(head.size) * head.size;
            for (size_t n = 0; n < cells; ++n) {
                // Выбираем лучший класс полным перебором: NC невелико, это дешевле сортировки.
                int cls = 0;
                float score = -1e30f;
                for (int c = 0; c < NC; ++c) {
                    // Каналы идут подряд для каждого пикселя сетки: индекс n * NC + c.
                    float v = head.cls[n * NC + c];
                    if (!SIGMOID_IN_HEF)
                        v = sig(v); // логиты -> вероятности; порог CONF задан для вероятностей
                    if (v > score) {
                        score = v;
                        cls = c;
                    }
                }
                if (score < CONF)
                    continue;
                // Бокс YOLO26 — четыре расстояния от центра ячейки в единицах страйда
                // (аналог dist2bbox * stride, только без DFL). Центр ячейки: +0.5.
                const float *b = &head.box[n * 4];
                const float ax = float(n % head.size) + 0.5f;
                const float ay = float(n / head.size) + 0.5f;
                Det d;
                // Клиппинг к 640x640: вырожденные боксы на краю кадра не должны подавлять
                // соседей (IoU у них 0), и за пределы letterbox они выходить не должны.
                d.x1 = std::clamp((ax - b[0]) * stride, 0.f, float(S));
                d.y1 = std::clamp((ay - b[1]) * stride, 0.f, float(S));
                d.x2 = std::clamp((ax + b[2]) * stride, 0.f, float(S));
                d.y2 = std::clamp((ay + b[3]) * stride, 0.f, float(S));
                d.score = score;
                d.cls = cls;
                // Коэффициенты масок лежат так же, как каналы классов: n * NM + c.
                d.coeff.assign(&head.coeff[n * NM], &head.coeff[n * NM] + NM);
                all.push_back(std::move(d));
            }
        }
        // NMS по боксам внутри каждого класса, максимум NMS_MAX объектов.
        auto objects = select_nms(std::move(all));
        // Маски — только для выживших: 32 умножения на пиксель 160x160 стоят недёшево.
        if (!objects.empty())
            make_masks(objects, proto_, size_t(PROTO) * PROTO * NM, PROTO, NM);
        return objects;
    }
};

} // namespace

int main(int argc, char **argv) {
    try {
        cv::setNumThreads(1); // сравнимые замеры с yolo11s_seg
        if (argc == 3 && std::string(argv[1]) == "--inspect") {
            inspect(argv[2]);
            return 0;
        }
        if (argc != 4) {
            std::cerr << "Usage: yolo26s_seg model.hef input output\n"
                         "       yolo26s_seg --inspect model.hef\n";
            return 2;
        }
        Yolo26Seg model(argv[1]);
        process_file(model, argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
