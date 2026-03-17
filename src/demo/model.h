#ifndef DEMO_INCLUDE_MODEL_H_
#define DEMO_INCLUDE_MODEL_H_

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "qbruntime/model.h"
#include "opencv2/opencv.hpp"
#include "post.h"

class Model {
public:
    Model() = delete;
    Model(const ModelSetting& model_setting, mobilint::Accelerator& acc);
    ~Model();

    static void work(Model* model, int worker_index, SizeState* size_state,
                     ItemQueue* item_queue, MatBuffer* feeder_buffer);

    cv::Mat inference(cv::Mat frame, cv::Size size, int stream_id);

private:
    cv::Mat (Model::*mInference)(cv::Mat, cv::Size, int);
    InputDataType mInputType = InputDataType::FLOAT32;

    std::unique_ptr<mobilint::Model> mModel;
    std::unique_ptr<PostProcessor> mPost;

    struct ScoreEmaTrack {
        std::array<float, 4> box;
        int label = -1;
        int missed = 0;
        float score_ema = 0.0f;
    };

    std::mutex mScoreTracksMutex;
    std::unordered_map<int, std::vector<ScoreEmaTrack>> mScoreTracksByWorker;

    float mScoreEmaAlpha = 0.5f;
    float mScoreEmaMatchIou = 0.3f;
    float mScoreEmaDisplayThres = 0.05f;
    int mScoreEmaMaxMissed = 30;

    void initYolo11FireDetection();

    cv::Mat inferenceYolo11FireDetection(cv::Mat frame, cv::Size size, int stream_id);
};
#endif
