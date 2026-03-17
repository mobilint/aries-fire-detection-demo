#include "demo/model.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "demo/post_yolo_anchorless_fire_detection.h"
#include "opencv2/opencv.hpp"

Model::Model(const ModelSetting& model_setting, mobilint::Accelerator& acc)
    : mInputType(model_setting.input_type) {
    mobilint::StatusCode sc;
    mobilint::ModelConfig mc;

    if (model_setting.is_num_core) {
        mc.setSingleCoreMode(model_setting.num_core);
    } else {
        mc.setSingleCoreMode(model_setting.core_id);
    }

    mModel = mobilint::Model::create(model_setting.mxq_path, mc, sc);
    mModel->launch(acc);

    // clang-format off
    switch (model_setting.model_type) {
    case ModelType::OBJECT:
        initYolo11FireDetection();
        break;
    default:
        throw std::invalid_argument("Unsupported model type in fire-detection demo.");
    }
    // clang-format on
}

Model::~Model() { mModel->dispose(); }

void Model::initYolo11FireDetection() {
    float conf_thres = 0.05f;
    float iou_thres = 0.45f;
    bool decode_bbox = true;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int nc = 2;

    mPost = std::make_unique<YOLOAnchorlessPostFireDetection>(
        nc, h, w, conf_thres, iou_thres, decode_bbox);
    mInference = &Model::inferenceYolo11FireDetection;
}

void Model::work(Model* model, int worker_index, SizeState* size_state,
                 ItemQueue* item_queue, MatBuffer* feeder_buffer) {
    Benchmarker benchmarker;

    cv::Mat frame, result;
    cv::Size result_size;

    int64_t frame_index = 0;
    while (true) {
        // workerReceive 함수에서 Mat()를 받으면 worker가 죽은 것으로 간주하고 화면을
        // clear한다.
        auto ssc = size_state->checkUpdate(result_size);
        if (ssc != SizeState::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        auto msc = feeder_buffer->get(frame, frame_index);
        if (msc != MatBuffer::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        benchmarker.start();
#ifdef USE_SLEEP_DRIVER
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cv::resize(frame, result, result_size);
#else
        result = model->inference(frame, result_size, worker_index);
#endif
        benchmarker.end();

        item_queue->push({worker_index, result, benchmarker.getFPS(),
                          benchmarker.getTimeSinceCreated(), benchmarker.getCount()});
    }
}

cv::Mat Model::inference(cv::Mat frame, cv::Size size, int stream_id) {
    return (this->*mInference)(frame, size, stream_id);
}
