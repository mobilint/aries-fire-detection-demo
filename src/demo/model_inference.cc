#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

#include "demo/model.h"
#include "qbruntime/qbruntime.h"
#include "opencv2/opencv.hpp"

#ifdef DEMO_DEBUG_LOG
#include <iostream>
#endif

namespace {
float iou_xyxy(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    float denom = area_a + area_b - inter;
    if (denom <= 0.0f) return 0.0f;
    return inter / denom;
}
}  // namespace

cv::Mat Model::inferenceYolo11FireDetection(cv::Mat frame, cv::Size size, int stream_id) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    static thread_local Benchmarker bm_prep;
    static thread_local Benchmarker bm_infer;
    static thread_local Benchmarker bm_post;
    static thread_local Benchmarker bm_draw;
    static thread_local int perf_count = 0;
    const bool perf_print = ((perf_count++ % 60) == 0);

#ifdef DEMO_DEBUG_LOG
    static int dbg_count = 0;
    const bool dbg_print = ((dbg_count++ % 60) == 0);
    if (dbg_print) {
        std::cout << "[YOLO11] frame=" << frame.cols << "x" << frame.rows
                  << " display=" << size.width << "x" << size.height << " input=" << w
                  << "x" << h << "x" << c << std::endl;
        auto outs = mModel->getOutputBufferInfo();
        std::cout << "[YOLO11] output_count=" << outs.size() << std::endl;
        for (size_t i = 0; i < outs.size(); i++) {
            std::cout << "  - out[" << i << "] " << outs[i].original_width << "x"
                      << outs[i].original_height << "x" << outs[i].original_channel
                      << std::endl;
        }
    }
#endif

    bm_prep.start();
    static thread_local int tl_w = 0;
    static thread_local int tl_h = 0;
    static thread_local int tl_c = 0;
    static thread_local size_t tl_input_size = 0;
    static thread_local mobilint::NDArray<float> input_img_f32;
    static thread_local mobilint::NDArray<uint8_t> input_img_u8;
    static thread_local cv::Mat resized_frame;
    static thread_local cv::Mat rgb;

    const size_t input_size = static_cast<size_t>(w) * h * c;
    if (tl_w != w || tl_h != h || tl_c != c || tl_input_size != input_size) {
        if (mInputType == InputDataType::FLOAT32) {
            input_img_f32 = mobilint::NDArray<float>({1, h, w, c}, sc);
        } else {
            input_img_u8 = mobilint::NDArray<uint8_t>({1, h, w, c}, sc);
        }
        tl_w = w;
        tl_h = h;
        tl_c = c;
        tl_input_size = input_size;
    }

    resized_frame.create(h, w, CV_8UC3);
    cv::resize(frame, resized_frame, cv::Size(w, h));

    rgb.create(h, w, CV_8UC3);
    cv::cvtColor(resized_frame, rgb, cv::COLOR_BGR2RGB);

    if (mInputType == InputDataType::FLOAT32) {
        cv::Mat input_mat(h, w, CV_32FC3, input_img_f32.data());
        rgb.convertTo(input_mat, CV_32FC3, 1.0f / 255.0f);
    } else {
        std::memcpy(input_img_u8.data(), rgb.data, input_size * sizeof(uint8_t));
    }
    bm_prep.end();

    bm_infer.start();
    std::vector<mobilint::NDArray<float>> result;
    if (mInputType == InputDataType::FLOAT32) {
        result = mModel->infer({input_img_f32}, sc);
    } else {
        result = mModel->infer({input_img_u8}, sc);
    }
    bm_infer.end();

    if (!sc) {
#ifdef DEMO_DEBUG_LOG
        std::cout << "[YOLO11] infer failed" << std::endl;
#endif
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef DEMO_DEBUG_LOG
    if (dbg_print) {
        std::cout << "[YOLO11] infer ok, npu_outs=" << result.size();
        if (!result.empty()) std::cout << " out0_size=" << result[0].size();
        std::cout << std::endl;
        if (!result.empty() && !result[0].empty()) {
            size_t nshow = std::min<size_t>(12, result[0].size());
            std::cout << "[YOLO11] out0_head:";
            for (size_t i = 0; i < nshow; i++) std::cout << " " << result[0][i];
            std::cout << std::endl;
        }
    }
#endif

    bm_post.start();
    static thread_local std::vector<std::array<float, 4>> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> labels;
    static thread_local std::vector<std::vector<float>> extras;

    boxes.clear();
    scores.clear();
    labels.clear();
    extras.clear();

    uint64_t ticket =
        mPost->enqueue(resized_frame, result, boxes, scores, labels, extras);
    mPost->receive(ticket);

    static thread_local std::vector<float> ema_scores;
    ema_scores.assign(scores.size(), 0.0f);
    {
        std::lock_guard<std::mutex> lk(mScoreTracksMutex);
        auto& score_tracks = mScoreTracksByWorker[stream_id];
        static thread_local std::vector<int> track_used;
        track_used.assign(score_tracks.size(), 0);

        for (size_t i = 0; i < scores.size(); i++) {
            int best_track = -1;
            float best_iou = 0.0f;

            for (size_t j = 0; j < score_tracks.size(); j++) {
                if (track_used[j]) continue;
                if (score_tracks[j].label != labels[i]) continue;

                float iou = iou_xyxy(score_tracks[j].box, boxes[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_track = (int)j;
                }
            }

            if (best_track >= 0 && best_iou >= mScoreEmaMatchIou) {
                auto& track = score_tracks[best_track];
                track.score_ema = mScoreEmaAlpha * scores[i] +
                                  (1.0f - mScoreEmaAlpha) * track.score_ema;

                track.box = boxes[i];
                track.label = labels[i];
                track.missed = 0;

                track_used[best_track] = 1;
                ema_scores[i] = track.score_ema;
            } else {
                ScoreEmaTrack new_track;

                new_track.box = boxes[i];
                new_track.label = labels[i];
                new_track.score_ema = scores[i];
                new_track.missed = 0;

                score_tracks.push_back(new_track);
                track_used.push_back(1);
                ema_scores[i] = new_track.score_ema;
            }
        }

        for (size_t j = 0; j < score_tracks.size(); j++) {
            if (!track_used[j]) score_tracks[j].missed++;
        }

        score_tracks.erase(std::remove_if(score_tracks.begin(), score_tracks.end(),
                                          [this](const ScoreEmaTrack& t) {
                                              return t.missed > mScoreEmaMaxMissed;
                                          }),
                           score_tracks.end());
    }
    bm_post.end();

#ifdef DEMO_DEBUG_LOG
    if (dbg_print) {
        std::cout << "[YOLO11] det_count=" << boxes.size() << std::endl;
        for (size_t i = 0; i < std::min<size_t>(boxes.size(), 5); i++) {
            std::cout << "  det[" << i << "] label=" << labels[i]
                      << " score=" << scores[i] << " ema=" << ema_scores[i] << " box=("
                      << boxes[i][0] << "," << boxes[i][1] << "," << boxes[i][2] << ","
                      << boxes[i][3] << ")" << std::endl;
        }
    }
#endif

    bm_draw.start();
    static thread_local cv::Mat result_frame;
    result_frame.create(size.height, size.width, frame.type());
    cv::resize(frame, result_frame, size);

    float sx = (float)size.width / (float)w;
    float sy = (float)size.height / (float)h;
    bool detected = false;
    for (int i = 0; i < (int)boxes.size(); i++) {
        int x1 = (int)(boxes[i][0] * sx);
        int y1 = (int)(boxes[i][1] * sy);
        int x2 = (int)(boxes[i][2] * sx);
        int y2 = (int)(boxes[i][3] * sy);

        x1 = std::max(0, std::min(x1, size.width - 1));
        y1 = std::max(0, std::min(y1, size.height - 1));
        x2 = std::max(0, std::min(x2, size.width - 1));
        y2 = std::max(0, std::min(y2, size.height - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        if (ema_scores[i] < mScoreEmaDisplayThres) continue;

        detected = true;
        cv::Scalar clr =
            (labels[i] == 0) ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 255);
        cv::rectangle(result_frame, cv::Point(x1, y1), cv::Point(x2, y2), clr, 2);
    }

    if (detected) {
        cv::rectangle(result_frame, cv::Point(0, 0),
                      cv::Point(size.width - 1, size.height - 1), cv::Scalar(0, 0, 255),
                      3);
    }
    bm_draw.end();

    if (perf_print) {
        // printf("[YOLO11-PERF] prep=%.3fms infer=%.3fms post=%.3fms draw=%.3fms\n",
        //        bm_prep.getSec() * 1000.0f, bm_infer.getSec() * 1000.0f,
        //        bm_post.getSec() * 1000.0f, bm_draw.getSec() * 1000.0f);
        // fflush(stdout);
    }

    return result_frame;
}

/*
cv::Mat Model::inferenceStyle(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;
    int wi = mModel->getInputBufferInfo()[0].original_width;
    int hi = mModel->getInputBufferInfo()[0].original_height;
    int ci = mModel->getInputBufferInfo()[0].original_channel;

    int wo = mModel->getOutputBufferInfo()[0].original_width;
    int ho = mModel->getOutputBufferInfo()[0].original_height;
    int co = mModel->getOutputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(wi, hi));

    auto input_img = std::make_unique<float[]>(wi * hi * ci);
    for (int i = 0; i < wi * hi; i++) {
        // BGR -> RGB
        input_img.get()[i * 3 + 0] = (float)resized_frame.data[i * 3 + 2] / 255.0f;
        input_img.get()[i * 3 + 1] = (float)resized_frame.data[i * 3 + 1] / 255.0f;
        input_img.get()[i * 3 + 2] = (float)resized_frame.data[i * 3 + 0] / 255.0f;
    }

    auto result = mModel->infer({input_img.get()}, sc);
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    for (int i = 0; i < wo * ho; i++) {
        // RGB -> BGR
        resized_frame.data[i * 3 + 0] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 2] * 255.0f, 255.0f));
        resized_frame.data[i * 3 + 1] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 1] * 255.0f, 255.0f));
        resized_frame.data[i * 3 + 2] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 0] * 255.0f, 255.0f));
    }

    // 추가된 Style Change 모델은 가장자리에 이물이 남아 이를 crop하여 사용하기로 한다.
    int crop_x = 35;
    int crop_y = 20;
    int crop_w = wo - crop_x * 2;
    int crop_h = ho - crop_y * 2;
    cv::Mat cropped_frame = resized_frame(cv::Rect{crop_x, crop_y, crop_w, crop_h});

    cv::Mat result_frame;
    cv::resize(cropped_frame, result_frame, size);

    return result_frame;
}
*/
/*
cv::Mat Model::inferenceFace(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;    // 640
    int h = mModel->getInputBufferInfo()[0].original_height;   // 512(480 + 2 * 16)
    int c = mModel->getInputBufferInfo()[0].original_channel;  // 3

    int y_pad = 16;
    int h_pad = h - y_pad * 2;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h_pad));

    // 480 이미지 위 아래에 16만큼 zero padding을 한다.
    // 512만한 pad 이미지에 resized_frame을 붙여넣기 한다.
    cv::Mat padded_resized_frame = cv::Mat::zeros(h, w, CV_8UC3);
    resized_frame.copyTo(padded_resized_frame({0, y_pad, w, h_pad}));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h * c; i++) {
        input_img.get()[i] = (float)padded_resized_frame.data[i] / 255;
    }

    auto result = mModel->infer({input_img.get()}, sc);
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> landmarks;
    uint64_t ticket =
        mPost->enqueue(resized_frame, result, boxes, scores, labels, landmarks);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(resized_frame, result_frame, size);

    return result_frame;
}
    */

/*
    cv::Mat Model::inferencePose(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;    // 640
    int h = mModel->getInputBufferInfo()[0].original_height;   // 512
    int c = mModel->getInputBufferInfo()[0].original_channel;  // 3

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h; i++) {
        int idx = i * 3;
        // BGR -> RGB 배열 변환
        input_img.get()[idx + 0] = ((float)resized_frame.data[idx + 2] / 255);
        input_img.get()[idx + 1] = ((float)resized_frame.data[idx + 1] / 255);
        input_img.get()[idx + 2] = ((float)resized_frame.data[idx + 0] / 255);
    }

    auto result =
        mModel->infer({input_img.get()}, sc);  // vector<vector<float>> {0, 1, 3, 2}
    if (!sc) {
        std::cout << "infer failed" << std::endl;
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef USE_ARIES2
    result = {
        std::move(result[1]), std::move(result[3]), std::move(result[5]),
        std::move(result[0]), std::move(result[2]), std::move(result[4]),
    };
#endif

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> keypoints;
    uint64_t ticket =
        mPost->enqueue(resized_frame, result, boxes, scores, labels, keypoints);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(resized_frame, result_frame, size);

    return result_frame;
}
*/

/*
cv::Mat Model::inferenceSeg(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;    // 640
    int h = mModel->getInputBufferInfo()[0].original_height;   // 512(480 + 2 * 16)
    int c = mModel->getInputBufferInfo()[0].original_channel;  // 3

    int y_pad = 16;
    int h_pad = h - y_pad * 2;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h_pad));

    // 480 이미지 위 아래에 16만큼 zero padding을 한다.
    // 512만한 pad 이미지에 resized_frame을 붙여넣기 한다.
    cv::Mat padded_resized_frame = cv::Mat::zeros(h, w, CV_8UC3);
    resized_frame.copyTo(padded_resized_frame({0, y_pad, w, h_pad}));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h * c; i++) {
        input_img.get()[i] = (float)padded_resized_frame.data[i] / 255;
    }

    auto result = mModel->infer({input_img.get()}, sc);
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef USE_ARIES2
    result = {
        std::move(result[2]), std::move(result[1]), std::move(result[4]),
        std::move(result[6]), std::move(result[0]), std::move(result[3]),
        std::move(result[5]),
    };
#endif

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;
    uint64_t ticket =
        mPost->enqueue(resized_frame, result, boxes, scores, labels, extras);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(resized_frame, result_frame, size);

    return result_frame;
}
*/
