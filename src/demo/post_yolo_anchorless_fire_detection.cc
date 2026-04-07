#include "demo/post_yolo_anchorless_fire_detection.h"

YOLOAnchorlessPostFireDetection::YOLOAnchorlessPostFireDetection(
    int nc, int imh, int imw, float conf_thres, float iou_thres, bool decode_bbox)
    : YOLOAnchorlessPost(nc, imh, imw, conf_thres, iou_thres, decode_bbox) {}

void YOLOAnchorlessPostFireDetection::postDecode(
    std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels, std::vector<std::vector<float>>& extras,
    const cv::Mat& model_input_image) {
    (void)boxes;
    (void)scores;
    (void)labels;
    (void)extras;
    (void)model_input_image;
}
