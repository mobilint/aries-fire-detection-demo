#ifndef DEMO_INCLUDE_POST_YOLO_ANCHORLESS_H_
#define DEMO_INCLUDE_POST_YOLO_ANCHORLESS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "demo/post.h"

class YOLOAnchorlessPost : public PostProcessor {
public:
    YOLOAnchorlessPost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                       bool decode_bbox);
    ~YOLOAnchorlessPost() override = default;

    uint64_t enqueue(cv::Mat& im, std::vector<mobilint::NDArray<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels,
                     std::vector<std::vector<float>>& extras) override;

    void receive(uint64_t receipt_no) override;

protected:
    virtual void postDecode(std::vector<std::array<float, 4>>& boxes,
                            std::vector<float>& scores, std::vector<int>& labels,
                            std::vector<std::vector<float>>& extras,
                            const cv::Mat& model_input_image);

private:
    struct LevelMap {
        int stride;
        int grid_h;
        int grid_w;
        int cls_idx;
        int box_idx;
        int box_channels;
    };

    int mNc;
    int mImh;
    int mImw;
    float mConfThres;
    float mIouThres;
    bool mDecodeBbox;
    uint64_t mTicket = 0;

    bool mMapsReady = false;
    std::array<LevelMap, 3> mLevels;

    static float sigmoid(float x);
    static void softmax16Inplace(std::array<float, 16>& a);
    static float iouXyxy(const std::array<float, 4>& a, const std::array<float, 4>& b);
    static void nmsClasswise(const std::vector<std::array<float, 4>>& boxes,
                             const std::vector<float>& scores,
                             const std::vector<int>& labels, float iou_thres,
                             std::vector<int>& keep_indices);

    std::array<float, 4> decodeBoxDfl(const mobilint::NDArray<float>& box_out,
                                      int cell_idx, int grid_x, int grid_y,
                                      int stride) const;
    std::array<float, 4> decodeBoxDirect(const mobilint::NDArray<float>& box_out,
                                         int cell_idx, int stride) const;
    bool tryEnqueueDecodedSingleOutput(
        const std::vector<mobilint::NDArray<float>>& npu_outs,
        std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
        std::vector<int>& labels) const;

    void setupOutputMaps(const std::vector<mobilint::NDArray<float>>& npu_outs);
};

#endif
