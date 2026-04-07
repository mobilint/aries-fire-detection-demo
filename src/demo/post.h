#ifndef DEMO_INCLUDE_POST_H_
#define DEMO_INCLUDE_POST_H_

#include <cstdint>
#include <vector>

#include "qbruntime/qbruntime.h"
#include "opencv2/opencv.hpp"

class PostProcessor {
public:
    virtual ~PostProcessor() {}
    virtual uint64_t enqueue(cv::Mat& im, std::vector<mobilint::NDArray<float>>& npu_outs,
                             std::vector<std::array<float, 4>>& boxes,
                             std::vector<float>& scores, std::vector<int>& labels,
                             std::vector<std::vector<float>>& extras) {
        return 0;
    }

    virtual void receive(uint64_t receipt_no) {}
};
#endif
