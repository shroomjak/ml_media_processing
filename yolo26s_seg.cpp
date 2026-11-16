// YOLO26s-seg на Hailo: минимальный бэкенд.
//
// HEF для yolo26s-seg собран без NMS (nms_postprocess для сегментации не поддерживается),
// поэтому сеть отдаёт сырые тензоры головы, а декодирование, NMS и маски считаются здесь:
//   на каждый страйд 8/16/32 три выхода: бокс (4 канала), классы (NC), коэффициенты масок (NM);
//   плюс общий выход прототипов масок PROTO x PROTO x NM.
// В отличие от YOLOv8/YOLO11 здесь нет DFL: reg_max = 1, бокс — сразу 4 числа.
//
// Параметры ниже переносятся вручную из .alls и из вывода `yolo26s_seg --inspect model.hef`.
#include "hailo_setup.hpp"
#include "media.hpp"

namespace {

constexpr int NC = 80;     // классов в модели (COCO = 80)
constexpr int NM = 32;     // прототипов масок
constexpr int PROTO = 160; // сторона карты прототипов: 640 / 4
constexpr float CONF = 0.25f;
// true, если в .alls есть change_output_activation(..., sigmoid) для классовых ветвей.
constexpr bool SIGMOID_IN_HEF = false;

// Выходы одного масштаба. Порядок внутри головы: сначала P3 (80x80), затем P4, P5.
struct Head {
    int size;            // сторона сетки: 80 / 40 / 20
    const float *box;    // 4 канала
    const float *cls;    // NC каналов
    const float *coeff;  // NM каналов
};

class Yolo26Seg {
    Session session_;
    std::vector<Head> heads_;
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
        // Роли определяются по размерностям: имена конвов зависят от парсера DFC.
        for (size_t i = 0; i < outs.size(); ++i) {
            const auto shape = outs[i].get_info().shape;
            if (shape.height != shape.width)
                throw std::runtime_error("Unexpected non-square output " + outs[i].name());
            if (shape.height == PROTO && shape.features == NM) {
                proto_ = session_.data(i);
                continue;
            }
            const int index = shape.height == 80 ? 0 : shape.height == 40 ? 1 : shape.height == 20 ? 2 : -1;
            if (index < 0)
                throw std::runtime_error("Unexpected output size: " + outs[i].name());
            if (heads_.size() <= size_t(index))
                heads_.resize(index + 1, Head{0, nullptr, nullptr, nullptr});
            Head &head = heads_[index];
            head.size = int(shape.height);
            if (shape.features == 4)
                head.box = session_.data(i);
            else if (shape.features == NM)
                head.coeff = session_.data(i);
            else
                head.cls = session_.data(i);
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
            const int stride = S / head.size;
            const size_t cells = size_t(head.size) * head.size;
            for (size_t n = 0; n < cells; ++n) {
                // Класс с максимальной оценкой; NC небольшое, полный перебор дешевле сортировки.
                int cls = 0;
                float score = -1e30f;
                for (int c = 0; c < NC; ++c) {
                    float v = head.cls[n * NC + c];
                    if (!SIGMOID_IN_HEF)
                        v = sig(v); // логиты, если sigmoid не вынесен в HEF
                    if (v > score) {
                        score = v;
                        cls = c;
                    }
                }
                if (score < CONF)
                    continue;
                // Бокс: расстояния от центра ячейки в единицах страйда.
                const float *b = &head.box[n * 4];
                const float ax = float(n % head.size) + 0.5f;
                const float ay = float(n / head.size) + 0.5f;
                Det d;
                d.x1 = std::clamp((ax - b[0]) * stride, 0.f, float(S));
                d.y1 = std::clamp((ay - b[1]) * stride, 0.f, float(S));
                d.x2 = std::clamp((ax + b[2]) * stride, 0.f, float(S));
                d.y2 = std::clamp((ay + b[3]) * stride, 0.f, float(S));
                d.score = score;
                d.cls = cls;
                d.coeff.assign(&head.coeff[n * NM], &head.coeff[n * NM] + NM);
                all.push_back(std::move(d));
            }
        }
        auto objects = select_nms(std::move(all));
        if (!objects.empty())
            make_masks(objects, proto_, size_t(PROTO) * PROTO * NM, PROTO, NM);
        return objects;
    }
};

} // namespace

int main(int argc, char **argv) {
    try {
        cv::setNumThreads(1);
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
