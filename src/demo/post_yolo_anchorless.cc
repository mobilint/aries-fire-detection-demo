#include "demo/post_yolo_anchorless.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

YOLOAnchorlessPost::YOLOAnchorlessPost(int nc, int imh, int imw, float conf_thres,
                                       float iou_thres, bool decode_bbox)
    : mNc(nc),
      mImh(imh),
      mImw(imw),
      mConfThres(conf_thres),
      mIouThres(iou_thres),
      mDecodeBbox(decode_bbox) {}

float YOLOAnchorlessPost::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void YOLOAnchorlessPost::softmax16Inplace(std::array<float, 16>& a) {
    float maxv = a[0];
    for (int i = 1; i < 16; i++) maxv = std::max(maxv, a[i]);

    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        a[i] = std::exp(a[i] - maxv);
        sum += a[i];
    }
    if (sum <= 0.0f) return;
    for (int i = 0; i < 16; i++) a[i] /= sum;
}

float YOLOAnchorlessPost::iouXyxy(const std::array<float, 4>& a,
                                  const std::array<float, 4>& b) {
    const float inter_x1 = std::max(a[0], b[0]);
    const float inter_y1 = std::max(a[1], b[1]);
    const float inter_x2 = std::min(a[2], b[2]);
    const float inter_y2 = std::min(a[3], b[3]);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float denom = area_a + area_b - inter_area;
    if (denom <= 0.0f) return 0.0f;

    return inter_area / denom;
}

void YOLOAnchorlessPost::nmsClasswise(const std::vector<std::array<float, 4>>& boxes,
                                      const std::vector<float>& scores,
                                      const std::vector<int>& labels,
                                      float iou_thres,
                                      std::vector<int>& keep_indices) {
    keep_indices.clear();
    if (boxes.empty()) return;

    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return scores[i] > scores[j]; });

    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t oi = 0; oi < order.size(); oi++) {
        const int i = order[oi];
        if (suppressed[i]) continue;

        keep_indices.push_back(i);
        for (size_t oj = oi + 1; oj < order.size(); oj++) {
            const int j = order[oj];
            if (suppressed[j]) continue;
            if (labels[i] != labels[j]) continue;
            if (iouXyxy(boxes[i], boxes[j]) > iou_thres) suppressed[j] = true;
        }
    }
}

std::array<float, 4> YOLOAnchorlessPost::decodeBoxDfl(
    const mobilint::NDArray<float>& box_out, int cell_idx, int grid_x, int grid_y,
    int stride) const {
    constexpr int kBins = 16;
    constexpr int kPerCell = 4 * kBins;
    const int base = cell_idx * kPerCell;

    float dist[4] = {0, 0, 0, 0};
    for (int side = 0; side < 4; side++) {
        std::array<float, kBins> prob{};
        for (int k = 0; k < kBins; k++) prob[k] = box_out[base + side * kBins + k];

        softmax16Inplace(prob);

        float expected = 0.0f;
        for (int k = 0; k < kBins; k++) expected += prob[k] * static_cast<float>(k);
        dist[side] = expected;
    }

    const float xmin = static_cast<float>(grid_x) - dist[0] + 0.5f;
    const float ymin = static_cast<float>(grid_y) - dist[1] + 0.5f;
    const float xmax = static_cast<float>(grid_x) + dist[2] + 0.5f;
    const float ymax = static_cast<float>(grid_y) + dist[3] + 0.5f;

    const float cx = (xmin + xmax) * 0.5f * static_cast<float>(stride);
    const float cy = (ymin + ymax) * 0.5f * static_cast<float>(stride);
    const float w = (xmax - xmin) * static_cast<float>(stride);
    const float h = (ymax - ymin) * static_cast<float>(stride);

    return {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
}

std::array<float, 4> YOLOAnchorlessPost::decodeBoxDirect(
    const mobilint::NDArray<float>& box_out, int cell_idx, int stride) const {
    const int base = cell_idx * 4;
    float x1 = box_out[base + 0];
    float y1 = box_out[base + 1];
    float x2 = box_out[base + 2];
    float y2 = box_out[base + 3];

    // decode가 모델 내부에 포함된 경우를 우선 가정한다.
    if (!(x2 > x1 && y2 > y1)) {
        // (cx, cy, w, h) 형식이면 xyxy로 변환
        const float cx = x1 * static_cast<float>(stride);
        const float cy = y1 * static_cast<float>(stride);
        const float w = std::max(0.0f, x2 * static_cast<float>(stride));
        const float h = std::max(0.0f, y2 * static_cast<float>(stride));
        x1 = cx - w * 0.5f;
        y1 = cy - h * 0.5f;
        x2 = cx + w * 0.5f;
        y2 = cy + h * 0.5f;
    }

    return {x1, y1, x2, y2};
}

bool YOLOAnchorlessPost::tryEnqueueDecodedSingleOutput(
    const std::vector<mobilint::NDArray<float>>& npu_outs,
    std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels) const {
    if (npu_outs.size() != 1) return false;

    const auto& out = npu_outs[0];
    const int kElemPerDet = mNc + 4;
    if (kElemPerDet <= 4) return false;
    if (out.size() == 0 || (out.size() % static_cast<size_t>(kElemPerDet)) != 0) {
        return false;
    }

    const int n_det = static_cast<int>(out.size() / static_cast<size_t>(kElemPerDet));

    std::vector<std::array<float, 4>> cand_boxes;
    std::vector<float> cand_scores;
    std::vector<int> cand_labels;
    cand_boxes.reserve(n_det);
    cand_scores.reserve(n_det);
    cand_labels.reserve(n_det);

    for (int i = 0; i < n_det; i++) {
        const int base = i * kElemPerDet;

        int best_label = 0;
        float best_score = -1.0f;
        for (int c = 0; c < mNc; c++) {
            float conf = out[base + 4 + c];
            // 일부 모델은 logits를 반환할 수 있으므로 필요한 경우만 sigmoid를 적용.
            if (conf < 0.0f || conf > 1.0f) conf = sigmoid(conf);
            if (conf > best_score) {
                best_score = conf;
                best_label = c;
            }
        }
        if (best_score < mConfThres) continue;

        float x1 = out[base + 0];
        float y1 = out[base + 1];
        float x2 = out[base + 2];
        float y2 = out[base + 3];

        // decoded 출력이 cxcywh 형태로 섞여 들어오는 경우를 대비한 최소한의 보정.
        if (!(x2 > x1 && y2 > y1)) {
            const float cx = x1;
            const float cy = y1;
            const float w = std::max(0.0f, x2);
            const float h = std::max(0.0f, y2);
            x1 = cx - w * 0.5f;
            y1 = cy - h * 0.5f;
            x2 = cx + w * 0.5f;
            y2 = cy + h * 0.5f;
        }

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(mImw - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(mImh - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(mImw - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(mImh - 1)));
        if (x2 <= x1 || y2 <= y1) continue;

        cand_boxes.push_back({x1, y1, x2, y2});
        cand_scores.push_back(best_score);
        cand_labels.push_back(best_label);
    }

    std::vector<int> keep;
    nmsClasswise(cand_boxes, cand_scores, cand_labels, mIouThres, keep);

    boxes.reserve(keep.size());
    scores.reserve(keep.size());
    labels.reserve(keep.size());
    for (int idx : keep) {
        boxes.push_back(cand_boxes[idx]);
        scores.push_back(cand_scores[idx]);
        labels.push_back(cand_labels[idx]);
    }

    return true;
}

void YOLOAnchorlessPost::setupOutputMaps(
    const std::vector<mobilint::NDArray<float>>& npu_outs) {
    auto find_unset = [&](size_t target_size, const std::vector<bool>& used) -> int {
        for (int i = 0; i < static_cast<int>(npu_outs.size()); i++) {
            if (!used[i] && npu_outs[i].size() == target_size) return i;
        }
        return -1;
    };

    std::array<LevelMap, 3> levels = {
        LevelMap{32, mImh / 32, mImw / 32, -1, -1, 0},
        LevelMap{16, mImh / 16, mImw / 16, -1, -1, 0},
        LevelMap{8, mImh / 8, mImw / 8, -1, -1, 0},
    };

    std::vector<bool> used(npu_outs.size(), false);
    for (auto& lv : levels) {
        const size_t ncell = static_cast<size_t>(lv.grid_h) * static_cast<size_t>(lv.grid_w);
        const size_t cls_size = ncell * static_cast<size_t>(mNc);
        const size_t box_dfl_size = ncell * static_cast<size_t>(64);
        const size_t box_raw_size = ncell * static_cast<size_t>(4);
        const size_t combined_dfl_size = ncell * static_cast<size_t>(mNc + 64);
        const size_t combined_raw_size = ncell * static_cast<size_t>(mNc + 4);

        int combined_idx = find_unset(combined_dfl_size, used);
        if (combined_idx >= 0) {
            lv.cls_idx = combined_idx;
            lv.box_idx = combined_idx;
            lv.box_channels = 64;
            used[combined_idx] = true;
            continue;
        }

        combined_idx = find_unset(combined_raw_size, used);
        if (combined_idx >= 0) {
            lv.cls_idx = combined_idx;
            lv.box_idx = combined_idx;
            lv.box_channels = 4;
            used[combined_idx] = true;
            continue;
        }

        const int cls_idx = find_unset(cls_size, used);
        int box_idx = find_unset(box_dfl_size, used);
        int box_channels = 64;
        if (box_idx < 0) {
            box_idx = find_unset(box_raw_size, used);
            box_channels = 4;
        }

        if (cls_idx < 0 || box_idx < 0) {
            throw std::invalid_argument(
                "Unable to infer output layout in YOLOAnchorless post-processing.");
        }

        lv.cls_idx = cls_idx;
        lv.box_idx = box_idx;
        lv.box_channels = box_channels;
        used[cls_idx] = true;
        used[box_idx] = true;
    }

    mLevels = levels;
    mMapsReady = true;
}

uint64_t YOLOAnchorlessPost::enqueue(cv::Mat& im,
                                     std::vector<mobilint::NDArray<float>>& npu_outs,
                                     std::vector<std::array<float, 4>>& boxes,
                                     std::vector<float>& scores,
                                     std::vector<int>& labels,
                                     std::vector<std::vector<float>>& extras) {
    (void)im;
    boxes.clear();
    scores.clear();
    labels.clear();
    extras.clear();

    if (tryEnqueueDecodedSingleOutput(npu_outs, boxes, scores, labels)) {
        postDecode(boxes, scores, labels, extras, im);
        return ++mTicket;
    }

    if (!mMapsReady) {
        setupOutputMaps(npu_outs);
    }

    std::vector<std::array<float, 4>> cand_boxes;
    std::vector<float> cand_scores;
    std::vector<int> cand_labels;

    for (const auto& lv : mLevels) {
        const auto& cls_out = npu_outs[lv.cls_idx];
        const auto& box_out = npu_outs[lv.box_idx];

        const int ncell = lv.grid_h * lv.grid_w;
        const bool split_output = (lv.cls_idx != lv.box_idx);

        for (int cell = 0; cell < ncell; cell++) {
            int best_label = 0;
            float best_score = -1.0f;

            if (split_output) {
                const int cls_base = cell * mNc;
                for (int c = 0; c < mNc; c++) {
                    const float conf = sigmoid(cls_out[cls_base + c]);
                    if (conf > best_score) {
                        best_score = conf;
                        best_label = c;
                    }
                }
            } else {
                const int cls_offset = (lv.box_channels == 64) ? 64 : 4;
                const int cls_base = cell * (mNc + cls_offset) + cls_offset;
                for (int c = 0; c < mNc; c++) {
                    const float conf = sigmoid(cls_out[cls_base + c]);
                    if (conf > best_score) {
                        best_score = conf;
                        best_label = c;
                    }
                }
            }

            if (best_score < mConfThres) continue;

            std::array<float, 4> xyxy;
            if (mDecodeBbox && lv.box_channels == 64) {
                const int gx = cell % lv.grid_w;
                const int gy = cell / lv.grid_w;
                const auto& box_src = split_output ? box_out : cls_out;
                xyxy = decodeBoxDfl(box_src, cell, gx, gy, lv.stride);
            } else {
                const auto& box_src = split_output ? box_out : cls_out;
                xyxy = decodeBoxDirect(box_src, cell, lv.stride);
            }

            float x1 = std::max(0.0f, std::min(xyxy[0], static_cast<float>(mImw - 1)));
            float y1 = std::max(0.0f, std::min(xyxy[1], static_cast<float>(mImh - 1)));
            float x2 = std::max(0.0f, std::min(xyxy[2], static_cast<float>(mImw - 1)));
            float y2 = std::max(0.0f, std::min(xyxy[3], static_cast<float>(mImh - 1)));
            if (x2 <= x1 || y2 <= y1) continue;

            cand_boxes.push_back({x1, y1, x2, y2});
            cand_scores.push_back(best_score);
            cand_labels.push_back(best_label);
        }
    }

    std::vector<int> keep;
    nmsClasswise(cand_boxes, cand_scores, cand_labels, mIouThres, keep);

    boxes.reserve(keep.size());
    scores.reserve(keep.size());
    labels.reserve(keep.size());
    for (int idx : keep) {
        boxes.push_back(cand_boxes[idx]);
        scores.push_back(cand_scores[idx]);
        labels.push_back(cand_labels[idx]);
    }

    postDecode(boxes, scores, labels, extras, im);
    return ++mTicket;
}

void YOLOAnchorlessPost::receive(uint64_t receipt_no) { (void)receipt_no; }

void YOLOAnchorlessPost::postDecode(std::vector<std::array<float, 4>>& boxes,
                                    std::vector<float>& scores,
                                    std::vector<int>& labels,
                                    std::vector<std::vector<float>>& extras,
                                    const cv::Mat& model_input_image) {
    (void)boxes;
    (void)scores;
    (void)labels;
    (void)extras;
    (void)model_input_image;
}
