// YOLO11s-seg на Hailo: бэкенд для HEF с сырыми выходами головы с DFL.
//
// Голова YOLOv8/YOLO11 (reg_max = 16): на каждый страйд приходят бокс (64 канала = 4 стороны
// по 16 бинов DFL), классы (NC) и коэффициенты масок (NM), плюс прототипы 160x160x32.
// DFL-декодирование выполняет common.hpp::decode(), который уже проверен на этой модели;
// число классов задаётся константой NC в common.hpp.
#include "hailo_setup.hpp"
#include "media.hpp"
#include <map>

class Yolo11Seg {
    Session session_;
    std::map<Key, size_t> ids_; // размер сетки и каналы -> индекс потока

  public:
    explicit Yolo11Seg(const char *path) : session_(path) {
        if (session_.outputs().size() != 10)
            throw std::runtime_error(
                has_nms_output(session_.outputs())
                    ? "HEF contains an NMS output: боковые и классовые ветви заменены NMS, а ветви "
                      "коэффициентов масок и прототипов при такой компиляции теряются — "
                      "сегментация в этом HEF невозможна. Уберите nms_postprocess из .alls."
                    : "yolo11s-seg expects 10 outputs (9 head + protos), got " +
                          std::to_string(session_.outputs().size()) + ". Run --inspect.");
        // Сопоставление по форме, а не по позиции: порядок выходов задаёт DFC.
        for (size_t i = 0; i < session_.outputs().size(); ++i) {
            const auto shape = session_.outputs()[i].get_info().shape;
            const Key key{int(shape.height), int(shape.features)};
            if (shape.height != shape.width || ids_.count(key))
                throw std::runtime_error("Unexpected or duplicate output " +
                                         session_.outputs()[i].name());
            if (size_t(shape.height) * shape.width * shape.features !=
                session_.outputs()[i].get_frame_size() / sizeof(float))
                throw std::runtime_error("Output " + session_.outputs()[i].name() +
                                         " is not a plain NHWC tensor");
            ids_[key] = i;
        }
        for (int size : {20, 40, 80})
            for (int channels : {NC, 32, 64})
                if (!ids_.count({size, channels}))
                    throw std::runtime_error("Missing output " + std::to_string(size) + "x" +
                                             std::to_string(size) + "x" +
                                             std::to_string(channels));
    }

    void prepare_input(const cv::Mat &image) { session_.prepare_input(image); }
    void infer() { session_.infer(); }

    std::vector<Det> postprocess() {
        // decode() принимает словарь тензоров: копия входит в замер postprocess.
        std::map<Key, std::vector<float>> tensors;
        for (const auto &item : ids_) {
            const size_t index = item.second;
            const size_t count = session_.outputs()[index].get_frame_size() / sizeof(float);
            tensors.emplace(item.first, std::vector<float>(session_.data(index),
                                                           session_.data(index) + count));
        }
        auto objects = decode(tensors); // sigmoid уже выполнен в HEF
        make_masks(objects, tensors.at({160, 32}));
        return objects;
    }
};

int main(int argc, char **argv) {
    try {
        cv::setNumThreads(1);
        if (argc == 3 && std::string(argv[1]) == "--inspect") {
            inspect(argv[2]);
            return 0;
        }
        if (argc != 4) {
            std::cerr << "Usage: yolo11s_seg model.hef input output\n"
                         "       yolo11s_seg --inspect model.hef\n";
            return 2;
        }
        Yolo11Seg model(argv[1]);
        process_file(model, argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
