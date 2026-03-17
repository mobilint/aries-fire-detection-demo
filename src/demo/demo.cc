#include "demo/demo.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/feeder.h"
#include "demo/model.h"
#include "qbruntime/qbruntime.h"
#include "opencv2/opencv.hpp"

using mobilint::Accelerator;
using mobilint::Cluster;
using mobilint::Core;
using mobilint::ModelConfig;
using mobilint::StatusCode;
using namespace std;

namespace {
void sleepForMS(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

std::string fpsToString(float fps) {
    char buf[20];
    snprintf(buf, sizeof(buf), "%8.2f", fps);
    return std::string(buf);
}

std::string secToString(int sec) {
    int h = sec / 3600;
    int m = (sec % 3600) / 60;
    int s = sec % 60;

    char buf[20];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, m, s);
    return std::string(buf);
}

std::string countToString(int count) {
    char buf[20];
    snprintf(buf, sizeof(buf), "%8d", count);
    return std::string(buf);
}

void displayAvgFPS(cv::Mat& display, float avg_fps) {
    const int x_start = 1400;
    const int y_start = 50;
    const double font_scale = 0.8;
    const int thickness = 1;
    const int font = cv::FONT_HERSHEY_DUPLEX;

    std::string label = "AVG FPS";
    std::string value = fpsToString(avg_fps);

    int baseline = 0;
    cv::Size label_size = cv::getTextSize(label, font, font_scale, thickness, &baseline);
    cv::Size value_size = cv::getTextSize(value, font, font_scale, thickness, &baseline);

    int w = label_size.width + 12 + value_size.width + 16;
    int h = std::max(label_size.height, value_size.height) + 16;

    cv::Rect box(x_start, y_start, w, h);
    if (box.x + box.width > display.cols || box.y + box.height > display.rows) {
        return;
    }

    cv::Mat roi = display(box);
    cv::Mat overlay = cv::Mat::zeros(roi.size(), roi.type());
    cv::addWeighted(overlay, 0.5, roi, 0.5, 0, roi);

    int x = x_start + 8;
    int y = y_start + h - 6;
    cv::putText(display, label, cv::Point(x, y), font, font_scale,
                cv::Scalar(255, 255, 255), thickness);
    cv::putText(display, value, cv::Point(x + label_size.width + 12, y), font, font_scale,
                cv::Scalar(0, 255, 0), thickness);
}

// ROI 사이즈 변화에 대해서 일정한 폰트를 유지할 수 있도록 한다.
// - 고정된 사이즈(w, h)에 대해 해당 폰트 사이즈와 굵기로 Benchmark 창을 만든다.
// - 필요한 만큼만 Benchmark 창을 Clip한다.
// - Clip한 창을 scale만큼 resize 후 frame에 띄운다.
void displayBenchmark(Item& item, bool is_fps_only = false) {
    float scale = 0.55;
    int w = 300;
    int h = 200;
    double font_scale = 1.0;
    int font_thickness = 1;

    cv::Mat board = cv::Mat::zeros(h, w, CV_8UC3);

    putText(board, "FPS", cv::Point(15, 40), cv::FONT_HERSHEY_DUPLEX, font_scale,
            cv::Scalar(255, 255, 255), font_thickness);
    putText(board, fpsToString(item.fps), cv::Point(112, 40), cv::FONT_HERSHEY_DUPLEX,
            font_scale, cv::Scalar(0, 255, 0), font_thickness);

    if (!is_fps_only) {
        putText(board, "Time", cv::Point(15, 80), cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(255, 255, 255), font_thickness);
        putText(board, secToString(item.time), cv::Point(110, 80),
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 255, 0),
                font_thickness);

        putText(board, "Infer", cv::Point(15, 115), cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(255, 255, 255), font_thickness);
        putText(board, countToString(item.count), cv::Point(112, 115),
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 255, 0),
                font_thickness);
    }

    int clip_w = 265;
    int clip_h;
    if (is_fps_only) {
        clip_h = 60;
    } else {
        clip_h = 94;
    }

    cv::Mat clip = board({0, 0, clip_w, clip_h});

    float resize_scale = (float)item.img.size().width / w * scale;
    // 이미지가 커지만 Debug 창도 같이 커진다.
    // 일정이상 비율이라면 더 이상 커지지 않게끔 한다.
    if (resize_scale > 0.9) {
        resize_scale = 0.9;
    }
    cv::resize(clip, clip, {0, 0}, resize_scale, resize_scale);

    int offset = (int)(item.img.size().width * 0.03);
    cv::Mat roi = item.img({{offset, offset}, clip.size()});
    cv::addWeighted(clip, 1, roi, 0.5, 0, roi);
}

void displayTime(cv::Mat& display, bool validate, float time = 0.0f) {
    float scale = 0.20;
    int w = 300;
    int h = 200;

    cv::Mat board = cv::Mat::zeros(h, w, CV_8UC3);
    if (validate) {
        double font_scale = 1.0;
        int font_thickness = 1;
        putText(board, "Time", cv::Point(15, 40), cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(255, 255, 255), font_thickness);
        putText(board, secToString(time), cv::Point(110, 40), cv::FONT_HERSHEY_DUPLEX,
                font_scale, cv::Scalar(0, 255, 0), font_thickness);
    } else {
        board = cv::Scalar(255, 255, 255);
    }

    cv::Size size = display.size();
    int clip_w = 265;
    int clip_h = 60;
    cv::Mat clip = board({0, 0, clip_w, clip_h});
    float resize_scale = (float)size.width * scale / w;
    cv::resize(clip, clip, {0, 0}, resize_scale, resize_scale);

    int x = (int)(size.width * 0.25);
    int y = (int)(size.height * 0.3);
    cv::Mat roi = display({{x, y}, clip.size()});
    clip.copyTo(roi);

    if (!validate) {
        sleepForMS(50);
    }
}
}  // namespace

Demo::Demo() : mDisplayFPSMode(false), mDisplayTimeMode(false), mModeIndex(-1) {}

void Demo::startWorker(int index) {
    if (index < 0 || index >= (int)mLayoutSetting.worker_layout.size()) {
        return;
    }
    std::lock_guard<std::mutex> lk(mWorkerEnabledMutex);
    if ((size_t)index >= mWorkerEnabled.size()) return;
    mWorkerEnabled[index] = 1;
}

void Demo::stopWorker(int index) {
    if (index < 0 || index >= (int)mLayoutSetting.worker_layout.size()) {
        return;
    }
    std::lock_guard<std::mutex> lk(mWorkerEnabledMutex);
    if ((size_t)index >= mWorkerEnabled.size()) return;
    mWorkerEnabled[index] = 0;
}

void Demo::startWorkerAll() {
    std::lock_guard<std::mutex> lk(mWorkerEnabledMutex);
    if (mWorkerEnabled.size() != mLayoutSetting.worker_layout.size()) {
        mWorkerEnabled.resize(mLayoutSetting.worker_layout.size(), 1);
    }
    std::fill(mWorkerEnabled.begin(), mWorkerEnabled.end(), 1);
}

void Demo::stopWorkerAll() {
    std::lock_guard<std::mutex> lk(mWorkerEnabledMutex);
    if (mWorkerEnabled.size() != mLayoutSetting.worker_layout.size()) {
        mWorkerEnabled.resize(mLayoutSetting.worker_layout.size(), 0);
    }
    std::fill(mWorkerEnabled.begin(), mWorkerEnabled.end(), 0);
}

void Demo::startProcessing() {
    if (mProcessingOn.exchange(true)) {
        return;
    }
    mInferThreads.clear();
    for (size_t mi = 0; mi < mModelSetting.size(); mi++) {
        int core_count = mModelSetting[mi].num_core;
        if (core_count <= 0) continue;
        for (int ci = 0; ci < core_count; ci++) {
            mInferThreads.emplace_back(&Demo::modelInferLoop, this, mi, ci, core_count);
        }
    }
}

void Demo::stopProcessing() {
    if (!mProcessingOn.exchange(false)) {
        return;
    }
    for (auto& t : mInferThreads) {
        if (t.joinable()) t.join();
    }
    mInferThreads.clear();
}

int Demo::getWorkerIndex(int x, int y) {
    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        if (mLayoutSetting.worker_layout[i].roi.contains(cv::Point(x, y))) {
            return i;
        }
    }
    return -1;
}

void Demo::onMouseEvent(int event, int x, int y, int flags, void* ctx) {
    if (event != cv::EVENT_RBUTTONDOWN && event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    Demo* demo = (Demo*)ctx;
    int worker_index = demo->getWorkerIndex(x, y);
    if (worker_index == -1) {
        return;
    }

    switch (event) {
    case cv::EVENT_RBUTTONDOWN:
        demo->stopWorker(worker_index);
        break;
    case cv::EVENT_LBUTTONDOWN:
        demo->startWorker(worker_index);
        break;
    }
}

void Demo::modelInferLoop(size_t model_index, int core_index, int core_count) {
    if (model_index >= mWorkersByModel.size() || core_count <= 0) {
        return;
    }

    while (mProcessingOn.load(std::memory_order_relaxed)) {
        const auto& workers = mWorkersByModel[model_index];
        for (size_t wi = core_index; wi < workers.size(); wi += core_count) {
            int worker_index = workers[wi];
            if (worker_index < 0 ||
                worker_index >= (int)mLayoutSetting.worker_layout.size()) {
                continue;
            }

            bool enabled = true;
            {
                std::lock_guard<std::mutex> lk(mWorkerEnabledMutex);
                if ((size_t)worker_index < mWorkerEnabled.size()) {
                    enabled = mWorkerEnabled[worker_index];
                }
            }
            if (!enabled) continue;

            if ((size_t)worker_index >= mWorkerLayoutValid.size() ||
                !mWorkerLayoutValid[worker_index]) {
                continue;
            }

            const auto& wl = mLayoutSetting.worker_layout[worker_index];
            if ((size_t)wl.model_index != model_index) continue;
            if (wl.feeder_index < 0 || wl.feeder_index >= (int)mFeeders.size()) {
                continue;
            }

            cv::Mat frame;
            if (!mFeeders[wl.feeder_index]->readFrame(frame)) {
                if (!mProcessingOn.load(std::memory_order_relaxed)) return;
                continue;
            }
            if (frame.empty()) continue;

            cv::Size out_size = wl.roi.size();
            Benchmarker& bench = mWorkerBench[worker_index];

            bench.start();
            cv::Mat result =
                mModels[wl.model_index]->inference(frame, out_size, worker_index);
            bench.end();

            if (result.empty() || result.size() != out_size) {
                continue;
            }

            float fps = bench.getFPS();
            if ((size_t)worker_index < mWorkerFps.size()) {
                mWorkerFps[worker_index].store(fps, std::memory_order_relaxed);
            }

            // if (mDisplayFPSMode) {
            //     Item item{worker_index, result, fps, bench.getTimeSinceCreated(),
            //               bench.getCount()};
            //     displayBenchmark(item);
            //     result = item.img;
            // }
            {
                // std::lock_guard<std::mutex> lk(mDisplayMutex);
                result.copyTo(mDisplay(wl.roi));
            }
        }
    }
}

void Demo::initWindow() {
    cv::Size window_size(1920, 1080);

    // Init Window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_GUI_NORMAL);
    cv::resizeWindow(WINDOW_NAME, window_size / 2);
    cv::moveWindow(WINDOW_NAME, 0, 0);
    cv::setMouseCallback(WINDOW_NAME, onMouseEvent, this);

    mDisplay = cv::Mat(window_size, CV_8UC3, {255, 255, 255});
    mDisplayBase = mDisplay.clone();

    mSplashes.clear();
    for (string path : {"../rc/layout/splash_01.png", "../rc/layout/splash_02.png"}) {
        cv::Mat splash = cv::imread(path);
        cv::resize(splash, splash, cv::Size(1920, 1080));
        mSplashes.push_back(splash);
    }
}

void Demo::initLayout(std::string path) {
    mLayoutSetting = loadLayoutSettingYAML(path);

    mDisplayBase.setTo(cv::Scalar(255, 255, 255));

    // Draw Banner
    for (const auto& il : mLayoutSetting.image_layout) {
        il.img.copyTo(mDisplayBase(il.roi));
    }

    {
        unique_lock<mutex> lock(mDisplayMutex);
        mDisplayBase.copyTo(mDisplay);
    }

    {
        std::lock_guard<std::mutex> lk(mWorkerEnabledMutex);
        mWorkerEnabled.assign(mLayoutSetting.worker_layout.size(), 1);
    }
    mWorkerBench.assign(mLayoutSetting.worker_layout.size(), Benchmarker());

    mWorkerFps = std::vector<std::atomic<float>>(mLayoutSetting.worker_layout.size());
    for (auto& v : mWorkerFps) {
        v.store(0.0f, std::memory_order_relaxed);
    }

    checkLayoutValidation();
}

void Demo::initFeeders(std::string path) {
    mFeederSetting = loadFeederSettingYAML(path);

    mFeeders.resize(mFeederSetting.size());
    for (int i = 0; i < mFeederSetting.size(); i++) {
        mFeeders[i] = make_unique<Feeder>(mFeederSetting[i]);
    }

    checkLayoutValidation();
}

void Demo::initModels(std::string path) {
    mModelSetting = loadModelSettingYAML(path);

    mModels.clear();
    mAccs.clear();
    mModels.resize(mModelSetting.size());
    for (int i = 0; i < mModelSetting.size(); i++) {
        int dev_no = mModelSetting[i].dev_no;
        auto it = mAccs.find(dev_no);
        if (it == mAccs.end()) {
            StatusCode sc;
            mAccs.emplace(dev_no, Accelerator::create(dev_no, sc));
        }
        mModels[i] = std::make_unique<Model>(mModelSetting[i], *mAccs[dev_no]);
    }
    checkLayoutValidation();
}

void Demo::checkLayoutValidation() {
    mWorkerLayoutValid.assign(mLayoutSetting.worker_layout.size(), 0);
    mWorkersByModel.assign(mModels.size(), {});
    if (mModels.empty() || mFeeders.empty()) {
        return;
    }

    for (size_t wi = 0; wi < mLayoutSetting.worker_layout.size(); wi++) {
        const auto& wl = mLayoutSetting.worker_layout[wi];

        bool valid = true;
        if (wl.feeder_index < 0 || wl.feeder_index >= (int)mFeeders.size()) valid = false;
        if (wl.model_index < 0 || wl.model_index >= (int)mModels.size()) valid = false;

        if (valid) {
            mWorkerLayoutValid[wi] = 1;
            if (wl.model_index >= 0 && (size_t)wl.model_index < mWorkersByModel.size()) {
                mWorkersByModel[wl.model_index].push_back((int)wi);
            }
        } else {
            printf(
                "[WARNING] Worker[%zu]: Invalid index detected (Feeder:%d, Model:%d)\n",
                wi, wl.feeder_index, wl.model_index);
        }
    }
}

void Demo::display() {
    unique_lock<mutex> lock(mDisplayMutex);
    if (mDisplayTimeMode) {
        displayTime(mDisplay, true, mBenchmarker.getTimeSinceCreated());
    }
    if (mDisplayFPSMode) {
        std::vector<uint8_t> enabled_snapshot;
        {
            std::lock_guard<std::mutex> lk(mWorkerEnabledMutex);
            enabled_snapshot = mWorkerEnabled;
        }

        float sum = 0.0f;
        int count = 0;
        for (size_t i = 0; i < enabled_snapshot.size() && i < mWorkerFps.size(); i++) {
            if (!enabled_snapshot[i]) continue;
            sum += mWorkerFps[i].load(std::memory_order_relaxed) / 3;
            count++;
        }
        if (count > 0) {
            displayAvgFPS(mDisplay, sum / count);
        }
    }
    cv::imshow(WINDOW_NAME, mDisplay);
}

void Demo::toggleDisplayFPSMode() { mDisplayFPSMode = !mDisplayFPSMode; }

void Demo::toggleDisplayTimeMode() {
    mDisplayTimeMode = !mDisplayTimeMode;
    if (!mDisplayTimeMode) {
        displayTime(mDisplay, false);
    }
}

void Demo::toggleScreenSize() {
    int cur = cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, !cur);

    cv::Size window_size(1920, 1080);
    cv::resizeWindow(WINDOW_NAME, window_size / 2);
}

bool Demo::keyHandler(int key) {
    if (cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_AUTOSIZE) != 0) {
        return false;
    }

    if (key == -1) {
        return true;
    }

    if (key >= 128) {  // Numpad 반환값은 128을 빼서 사용
        key -= 128;
    }

    key = tolower(key);

    if (key == 'd') {  // 'D'ebug
        toggleDisplayFPSMode();
    } else if (key == 't') {  // 'T'ime
        toggleDisplayTimeMode();
    } else if (key == 'm') {  // 'M'aximize Screen
        toggleScreenSize();
    } else if (key == 'c') {  // 'C'lear
        stopWorkerAll();
    } else if (key == 'f') {  // 'F'ill Grid
        startWorkerAll();
    } else if (key == 'q' || key == 27) {  // 'Q'uit, esc
        return false;
    } else if (key == '1' || key == '2') {
        setMode(key - '0');
    }

    return true;
}

void Demo::setMode(int mode_index) {
    // clang-format off
    switch (mode_index) {
        case 1: setMode1(); break;
        case 2: setMode2(); break;
    }
    // clang-format on
}

void Demo::showSplash(int splash_index) {
    if (splash_index < 0 || splash_index >= (int)mSplashes.size()) {
        return;
    }
    unique_lock<mutex> lock(mDisplayMutex);
    mSplashes[splash_index].copyTo(mDisplay);
    cv::imshow(WINDOW_NAME, mDisplay);
    cv::waitKey(100);
}

void Demo::applyMode(int mode_index, const std::string& layout_path,
                     const std::string& model_path, const std::string& feeder_path,
                     int splash_index) {
    if (mModeIndex == mode_index) {
        return;
    }

    stopProcessing();
    stopWorkerAll();
    showSplash(splash_index);
    initLayout(layout_path);
    initModels(model_path);
    initFeeders(feeder_path);
    startWorkerAll();
    startProcessing();
    mModeIndex = mode_index;
    sleepForMS(500);
}

void Demo::setMode1() {
    applyMode(1, "../rc/LayoutSetting_MLA100.yaml", "../rc/ModelSetting_MLA100.yaml",
              "../rc/FeederSetting_MLA100.yaml", 0);
}

void Demo::setMode2() {
    applyMode(2, "../rc/LayoutSetting_MLA400.yaml", "../rc/ModelSetting_MLA400.yaml",
              "../rc/FeederSetting_MLA400.yaml", 1);
}

void Demo::run() {
    initWindow();
    initLayout("../rc/LayoutSetting_MLA100.yaml");
    initModels("../rc/ModelSetting_MLA100.yaml");
    initFeeders("../rc/FeederSetting_MLA100.yaml");
    mModeIndex = 1;

    startWorkerAll();
    startProcessing();

    toggleScreenSize();
    toggleDisplayFPSMode();

    while (true) {
        display();
        if (!keyHandler(cv::waitKey(10))) {
            break;
        }
    }

    stopProcessing();
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    Demo demo;
    demo.run();

    return 0;
}
