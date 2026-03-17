#ifndef DEMO_BENCHMARKER_H_
#define DEMO_BENCHMARKER_H_

#include <array>
#include <chrono>
#include <cstddef>

/**
 * @brief Benchmarker that measures elapsed time between start() and end()
 *        and provides the last and averaged results.
 *
 * - Calling start() records the beginning of a measurement interval.
 * - Calling end() computes the elapsed time since start() and stores it
 *   as a single sample.
 * - Provides the average of the most recent SIZE samples (getAvgSec)
 *   and the most recent single sample (getLastSec).
 *
 * Notes:
 * - The intended usage order is start() -> end().
 * - If end() is called without a preceding start(), the result is treated
 *   as 0 seconds for safety.
 */
class Benchmarker {
    using Clock = std::chrono::steady_clock;
    static constexpr size_t SIZE = 1000;

public:
    /**
     * @brief Constructs a Benchmarker and initializes internal state.
     *
     * - Initializes the sample buffer to zero.
     * - Records the creation time and initializes the previous timestamp.
     */
    Benchmarker()
        : mTimes{},
          mSum(0.0f),
          mCount(0),
          mCreated(Clock::now()),
          mPrev(mCreated),
          mRunningTime(0.0f),
          mHasLast(false),
          mLastSec(0.0f),
          mIsStarted(false) {}

    /**
     * @brief Starts a measurement interval.
     */
    void start() {
        mPrev = Clock::now();
        mIsStarted = true;
    }

    /**
     * @brief Ends the measurement interval and records the elapsed time.
     *
     * @return The elapsed time of this interval in seconds.
     *         Returns 0 if called without a preceding start().
     */
    float end() {
        if (!mIsStarted) {
            mHasLast = true;
            mLastSec = 0.0f;
            return 0.0f;
        }

        std::chrono::duration<float> dt = Clock::now() - mPrev;
        float t = dt.count();

        // Circular buffer handling: remove the overwritten value from the sum
        if (mCount >= SIZE) {
            mSum -= mTimes[mCount % SIZE];
        }

        mTimes[mCount % SIZE] = t;
        mSum += t;

        mRunningTime += t;
        mCount++;
        mHasLast = true;
        mLastSec = t;

        mIsStarted = false;
        return t;
    }

    /**
     * @brief Returns the most recent measured interval in seconds.
     *
     * @return The last measured interval, or 0 if no measurement exists.
     */
    float getSec() const { return mHasLast ? mLastSec : 0.0f; }

    /**
     * @brief Returns the average duration of recent samples in seconds.
     *
     * @return The average duration in seconds, or 0 if no samples exist.
     */
    float getAvgSec() const {
        size_t n = getSampleCount_();
        if (n == 0) return 0.0f;
        return mSum / static_cast<float>(n);
    }

    /**
     * @brief Returns FPS based on the most recent measurement.
     *
     * @return Frames per second based on the last interval,
     *         or 0 if the last interval is zero.
     */
    float getFPS() const {
        float s = getSec();
        return (s == 0.0f) ? 0.0f : (1.0f / s);
    }

    /**
     * @brief Returns FPS based on the averaged measurement.
     *
     * @return Frames per second based on the average interval,
     *         or 0 if the average interval is zero.
     */
    float getAvgFPS() const {
        float s = getAvgSec();
        return (s == 0.0f) ? 0.0f : (1.0f / s);
    }

    /**
     * @brief Returns the cumulative time of all recorded samples in seconds.
     */
    float getRunningTime() const { return mRunningTime; }

    /**
     * @brief Returns the total number of recorded samples.
     */
    size_t getCount() const { return mCount; }

    /**
     * @brief Returns the elapsed time since this object was created.
     *
     * @return Time since construction in seconds.
     */
    float getTimeSinceCreated() const {
        std::chrono::duration<float> dt = Clock::now() - mCreated;
        return dt.count();
    }

    /**
     * @brief Indicates whether a measurement has been started but not ended.
     *
     * @return true if start() has been called without a corresponding end().
     */
    bool isStarted() const { return mIsStarted; }

private:
    /**
     * @brief Returns the number of samples used for averaging.
     */
    size_t getSampleCount_() const { return (mCount < SIZE) ? mCount : SIZE; }

private:
    // Circular buffer storing the most recent SIZE sample durations (seconds)
    std::array<float, SIZE> mTimes{};

    // Sum of durations of the most recent samples (for averaging)
    float mSum = 0.0f;

    // Total number of recorded samples
    size_t mCount = 0;

    // Timestamp when the object was created
    Clock::time_point mCreated{};

    // Timestamp recorded at start()
    Clock::time_point mPrev{};

    // Cumulative duration of all recorded samples
    float mRunningTime = 0.0f;

    // Indicates whether a valid last sample exists and stores its value
    bool mHasLast = false;
    float mLastSec = 0.0f;

    // Indicates whether start() has been called without end()
    bool mIsStarted = false;
};

#endif  // DEMO_BENCHMARKER_H_
