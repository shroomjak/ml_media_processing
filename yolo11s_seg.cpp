// YOLO11s-seg на Hailo: инференс с NMS на CPU.
//
// Что делает программа целиком:
//   кадр -> letterbox 640x640 RGB (common.hpp::prepare)
//        -> HEF на Hailo выдаёт 10 сырых тензоров головы
//        -> декод DFL + sigmoid(уже в HEF) -> NMS по боксам -> маски по коэффициентам
//        -> отрисовка (common.hpp::annotate) -> запись в файл того же формата (media.hpp)
//
// Почему NMS тут, а не в HEF: DFC умеет собирать NMS только для детекционных голов
// (и отдельно для yolov5_seg), для голов YOLOv8/YOLO11-seg такой ветки нет. Выход NMS
// к тому же не содержит индекса якоря, так что коэффициенты масок было бы не с чем
// сопоставить. Подробный разбор — docs/REVIEW.md и docs/NMS_REVIEW.md.
#include "hailo_setup.hpp"
#include "media.hpp"
#include <map>

// Голова YOLO11-seg (reg_max = 16) отдаёт на каждый страйд три тензора:
//   box   64 = 4 стороны × 16 бинов DFL,
//   cls   NC  — вероятности классов (sigmoid уже выполнен в HEF),
//   coeff 32  — коэффициенты масок,
// плюс общая карта прототипов 160x160x32. Итого 10 выходов.
class Yolo11Seg {
    Session session_;
    std::map<Key, size_t> ids_; // (сторона сетки, число каналов) -> индекс потока

  public:
    explicit Yolo11Seg(const char *path) : session_(path) {
        // Число выходов — первая проверка: 9 головных + прототипы.
        if (session_.outputs().size() != 10)
            throw std::runtime_error(
                has_nms_output(session_.outputs())
                    ? "HEF contains an NMS output: боковые и классовые ветви заменены NMS, а ветви "
                      "коэффициентов масок и прототипов при такой компиляции теряются — "
                      "сегментация в этом HEF невозможна. Уберите nms_postprocess из .alls."
                    : "yolo11s-seg expects 10 outputs (9 head + protos), got " +
                          std::to_string(session_.outputs().size()) + ". Run --inspect.");

        // Роли выходов определяются по форме, а не по позиции: порядок задаёт DFC,
        // а имена конвов зависят от парсера. Формы должны совпасть с таблицей выше.
        for (size_t i = 0; i < session_.outputs().size(); ++i) {
            const auto shape = session_.outputs()[i].get_info().shape;
            const Key key{int(shape.height), int(shape.features)};
            if (shape.height != shape.width || ids_.count(key))
                throw std::runtime_error("Unexpected or duplicate output " +
                                         session_.outputs()[i].name());
            // Проверка, что выход — плоский NHWC-тензор: иначе разбор по индексам неверен.
            if (size_t(shape.height) * shape.width * shape.features !=
                session_.outputs()[i].get_frame_size() / sizeof(float))
                throw std::runtime_error("Output " + session_.outputs()[i].name() +
                                         " is not a plain NHWC tensor");
            ids_[key] = i;
        }
        // Полный набор: для каждой из сеток 20/40/80 — бокс 64, классы NC, коэффициенты 32.
        for (int size : {20, 40, 80})
            for (int channels : {64, NC, 32})
                if (!ids_.count({size, channels}))
                    throw std::runtime_error("Missing output " + std::to_string(size) + "x" +
                                             std::to_string(size) + "x" +
                                             std::to_string(channels));
    }

    // Вызывается из process_file(): подготовленный RGB-кадр и сам инференс.
    void prepare_input(const cv::Mat &image) { session_.prepare_input(image); }
    void infer() { session_.infer(); }

    std::vector<Det> postprocess() {
        // decode() принимает словарь тензоров, поэтому выходы копируются: копия входит
        // в замер postprocess и для 640x640 невелика (~2 МБ на кадр).
        std::map<Key, std::vector<float>> tensors;
        for (const auto &item : ids_) {
            const size_t index = item.second;
            const size_t count = session_.outputs()[index].get_frame_size() / sizeof(float);
            tensors.emplace(item.first, std::vector<float>(session_.data(index),
                                                           session_.data(index) + count));
        }
        // decode(): DFL (softmax по 16 бинам -> ожидание), выбор лучшего класса,
        // затем select_nms() — подавление дублей по боксам внутри каждого класса.
        auto objects = decode(tensors);
        // Маски строятся только для объектов, переживших NMS: mask = sigmoid(coeff · proto).
        // Для подавленных дублей это была бы двойная работа, а их маски всё равно не нужны.
        make_masks(objects, tensors.at({160, 32}));
        return objects;
    }
};

int main(int argc, char **argv) {
    try {
        // Одно число потоков OpenCV: иначе замеры между бэкендами несравнимы.
        cv::setNumThreads(1);
        if (argc == 3 && std::string(argv[1]) == "--inspect") {
            inspect(argv[2]); // формы выходов HEF без платы
            return 0;
        }
        if (argc != 4) {
            std::cerr << "Usage: yolo11s_seg model.hef input output\n"
                         "       yolo11s_seg --inspect model.hef\n";
            return 2;
        }
        Yolo11Seg model(argv[1]);
        // Общий цикл по фото/видео: замеры и запись результата тем же форматом.
        process_file(model, argv[2], argv[3]);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
