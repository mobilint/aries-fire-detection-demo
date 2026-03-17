#include "demo/feeder.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

Feeder::Feeder(const FeederSetting& feeder_setting) : mFeederSetting(feeder_setting) {
    for (int i = 0; i < mFeederSetting.src_path.size(); i++) {
        cv::VideoCapture cap;
        switch (mFeederSetting.feeder_type) {
        case FeederType::CAMERA: {
            cap.open(stoi(mFeederSetting.src_path[i]), cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
            cap.set(cv::CAP_PROP_FPS, 30);
            mDelayOn = false;
            break;
        }
        case FeederType::VIDEO: {
            cap.open(mFeederSetting.src_path[i]);
            mDelayOn = true;
            break;
        }
        case FeederType::IPCAMERA: {
            cap.open(mFeederSetting.src_path[i]);
            mDelayOn = false;
            break;
        }
        }
        mCap.push_back(cap);
    }
}

bool Feeder::readFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lk(mCapMutex);
    if (mCap.empty()) {
        frame = cv::Mat::zeros(360, 640, CV_8UC3);
        cv::putText(frame, "Dummy Feeder", cv::Point(140, 190),
                    cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
        return true;
    }

    size_t start = mNextCapIndex;
    for (size_t attempt = 0; attempt < mCap.size(); attempt++) {
        size_t idx = (start + attempt) % mCap.size();
        cv::VideoCapture& cap = mCap[idx];
        if (!cap.isOpened()) {
            continue;
        }

        cap >> frame;
        if (frame.empty() && mFeederSetting.feeder_type == FeederType::VIDEO) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
        }

        if (!frame.empty()) {
            mNextCapIndex = (idx + 1) % mCap.size();
            return true;
        }
    }

    frame = cv::Mat::zeros(360, 640, CV_8UC3);
    cv::putText(frame, "Dummy Feeder", cv::Point(140, 190),
                cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
    return true;
}

void Feeder::feed(int index, ItemQueue& item_queue, cv::Size roi_size) {
    mFeederBuffer.open();
    while (mIsFeederRunning) {
        for (int i = 0; i < mCap.size(); i++) {
            if (mCap[i].isOpened()) {
                if (mFeederSetting.feeder_type == FeederType::VIDEO) {
                    double total_frames = mCap[i].get(cv::CAP_PROP_FRAME_COUNT);
                    if (total_frames > 0) {
                        int random_start = rand() % static_cast<int>(total_frames);
                        mCap[i].set(cv::CAP_PROP_POS_FRAMES, random_start);
                    }
                }
                feedInternal(index, item_queue, mCap[i], roi_size, mDelayOn);
                mCap[i].set(cv::CAP_PROP_POS_FRAMES, 0);
            } else {
                feedInternalDummy(index, item_queue, roi_size);
            }
        }
    }
    mFeederBuffer.close();
}

void Feeder::feedInternal(int index, ItemQueue& item_queue, cv::VideoCapture& cap,
                          cv::Size roi_size, bool delay_on) {
    Benchmarker benchmarker;
    int perf_count = 0;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty() || !mIsFeederRunning) {
            break;
        }

        mFeederBuffer.put(frame);

        if (!roi_size.empty()) {
            cv::Mat resized_frame;
            resize(frame, resized_frame, roi_size);

            ItemQueue::StatusCode sc;
            sc = item_queue.push({index, resized_frame, benchmarker.getFPS(), 0.0, 0});
            if (sc != ItemQueue::OK) {
                break;
            }
        }

        if (delay_on) {
            std::this_thread::sleep_for(std::chrono::milliseconds(45));
        }

        benchmarker.end();
        if ((perf_count++ % 60) == 0) {
            // printf("[FEED] idx=%d interval=%.3fms fps=%.2f\n", index,
            //        benchmarker.getSec() * 1000, benchmarker.getFPS());
            // fflush(stdout);
        }
    }
}

void Feeder::feedInternalDummy(int index, ItemQueue& item_queue, cv::Size roi_size) {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        frame = cv::Mat::zeros(360, 640, CV_8UC3);
        cv::putText(frame, "Dummy Feeder", cv::Point(140, 190), cv::FONT_HERSHEY_DUPLEX,
                    1.5, cv::Scalar(0, 255, 0), 2);
        if (frame.empty() || !mIsFeederRunning) {
            break;
        }

        mFeederBuffer.put(frame);

        if (!roi_size.empty()) {
            cv::Mat resized_frame;
            resize(frame, resized_frame, roi_size);

            ItemQueue::StatusCode sc;
            sc = item_queue.push({index, resized_frame, benchmarker.getFPS(), 0.0, 0});
            if (sc != ItemQueue::OK) {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        benchmarker.end();
    }
}
