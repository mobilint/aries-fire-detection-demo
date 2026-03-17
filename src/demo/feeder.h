#ifndef DEMO_INCLUDE_FEEDER_H_
#define DEMO_INCLUDE_FEEDER_H_

#include <mutex>
#include <string>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

class Feeder {
public:
    Feeder() = delete;
    Feeder(const FeederSetting& feeder_setting);
    ~Feeder() = default;

    bool readFrame(cv::Mat& frame);
    void feed(int index, ItemQueue& item_queue, cv::Size roi_size);
    MatBuffer& getMatBuffer() { return mFeederBuffer; }
    void start() { mIsFeederRunning = true; }
    void stop() { mIsFeederRunning = false; }

private:
    void feedInternal(int index, ItemQueue& item_queue, cv::VideoCapture& cap,
                      cv::Size roi_size, bool delay_on);
    void feedInternalDummy(int index, ItemQueue& item_queue, cv::Size roi_size);

    FeederSetting mFeederSetting;
    MatBuffer mFeederBuffer;
    bool mIsFeederRunning = true;
    std::vector<cv::VideoCapture> mCap;
    bool mDelayOn;
    std::mutex mCapMutex;
    size_t mNextCapIndex = 0;
};
#endif
