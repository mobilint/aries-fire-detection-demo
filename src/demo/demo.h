#ifndef DEMO_INCLUDE_DEMO_H_
#define DEMO_INCLUDE_DEMO_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <thread>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "qbruntime/qbruntime.h"
#include "qbruntime/model.h"
#include "opencv2/opencv.hpp"

class Model;
class Feeder;

class Demo {
public:
    Demo();
    void run();

private:
    void startWorker(int index);
    void stopWorker(int index);

    void startWorkerAll();
    void stopWorkerAll();

    void startProcessing();
    void stopProcessing();

    int getWorkerIndex(int x, int y);
    static void onMouseEvent(int event, int x, int y, int flags, void* userdata);

    void modelInferLoop(size_t model_index, int core_index, int core_count);

    void initWindow();
    void initLayout(std::string path);
    void initModels(std::string path);
    void initFeeders(std::string path);
    void checkLayoutValidation();
    void display();

    void toggleDisplayFPSMode();
    void toggleDisplayTimeMode();
    void toggleScreenSize();
    bool keyHandler(int key);

    void setMode(int mode_index);
    void setMode1();
    void setMode2();
    void applyMode(int mode_index, const std::string& layout_path,
                   const std::string& model_path, const std::string& feeder_path,
                   int splash_index);
    void showSplash(int splash_index);

    std::vector<FeederSetting> loadFeederSettingYAML(const std::string& path);
    std::vector<ModelSetting> loadModelSettingYAML(const std::string& path);
    LayoutSetting loadLayoutSettingYAML(const std::string& path);

    const std::string WINDOW_NAME = "Mobilint Inference Demo";

    std::mutex mDisplayMutex;
    cv::Mat mDisplay;      // front buffer
    cv::Mat mDisplayBase;  // static layout/background
    Benchmarker mBenchmarker;

    bool mDisplayFPSMode;
    bool mDisplayTimeMode;

    std::vector<cv::Mat> mSplashes;
    int mModeIndex;

    std::vector<FeederSetting> mFeederSetting;  // FeederSetting.yaml에서 읽은 정보 저장
    std::vector<ModelSetting> mModelSetting;    // ModelSetting.yaml에서 읽은 정보 저장
    LayoutSetting mLayoutSetting;               // LayoutSetting.yaml에서 읽은 정보 저장
    std::vector<uint8_t> mWorkerLayoutValid;
    std::vector<std::vector<int>> mWorkersByModel;

    std::map<int, std::unique_ptr<mobilint::Accelerator>> mAccs;

    std::vector<std::unique_ptr<Model>> mModels;
    std::vector<std::unique_ptr<Feeder>> mFeeders;

    std::atomic<bool> mProcessingOn{false};

    std::vector<std::thread> mInferThreads;

    std::mutex mWorkerEnabledMutex;
    std::vector<uint8_t> mWorkerEnabled;
    std::vector<Benchmarker> mWorkerBench;
    std::vector<std::atomic<float>> mWorkerFps;
};

#endif
