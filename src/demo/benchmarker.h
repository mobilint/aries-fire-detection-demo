#ifndef DEMO_BENCHMARKER_H_
#define DEMO_BENCHMARKER_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <optional>

class Benchmarker {
    using Clock = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;
    static constexpr std::size_t kWindowSize = 1000;

   public:
    Benchmarker()
        : mSamples{},
          mCreated(Clock::now()) {}

    void start() {
        mStart = Clock::now();
        mIsStarted = true;
    }

    std::optional<double> end() {
        if (!mIsStarted) {
            return std::nullopt;
        }

        const auto now = Clock::now();
        const double sec = Seconds(now - *mStart).count();

        if (mCount >= kWindowSize) {
            mSum -= mSamples[mCount % kWindowSize];
        }

        mSamples[mCount % kWindowSize] = sec;
        mSum += sec;
        mRunningTime += sec;
        mLastSec = sec;
        ++mCount;
        mIsStarted = false;
        mStart.reset();

        return sec;
    }

    double getSec() const {
        return mLastSec.value_or(0.0);
    }

    double getAvgSec() const {
        const std::size_t n = sampleCount();
        return (n == 0) ? 0.0 : (mSum / static_cast<double>(n));
    }

    double getFPS() const {
        const double s = getSec();
        return (s > 0.0) ? (1.0 / s) : 0.0;
    }

    double getAvgFPS() const {
        const double s = getAvgSec();
        return (s > 0.0) ? (1.0 / s) : 0.0;
    }

    double getRunningTime() const {
        return mRunningTime;
    }

    std::size_t getCount() const {
        return mCount;
    }

    double getTimeSinceCreated() const {
        return Seconds(Clock::now() - mCreated).count();
    }

    bool isStarted() const {
        return mIsStarted;
    }

    void reset() {
        mSamples.fill(0.0);
        mSum = 0.0;
        mCount = 0;
        mRunningTime = 0.0;
        mLastSec.reset();
        mStart.reset();
        mIsStarted = false;
        mCreated = Clock::now();
    }

   private:
    std::size_t sampleCount() const {
        return (mCount < kWindowSize) ? mCount : kWindowSize;
    }

   private:
    std::array<double, kWindowSize> mSamples{};
    double mSum = 0.0;
    std::size_t mCount = 0;
    Clock::time_point mCreated{};

    std::optional<Clock::time_point> mStart;
    double mRunningTime = 0.0;
    std::optional<double> mLastSec;
    bool mIsStarted = false;
};

#endif  // DEMO_BENCHMARKER_H_