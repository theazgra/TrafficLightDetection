/** \file Stopwatch.h
 * Stopwatch for timing.
 */

#ifndef BACHELOR_STOPWATCH_H
#define BACHELOR_STOPWATCH_H

#include <map>
#include <chrono>
#include <exception>
#include <string>
#include <tgmath.h>
#include <vector>

/// Stopwatch class, support multiple stopwatches inside one instance. Contains functionality to manage those stopwatches.
class Stopwatch {

private:
    ///Basic stopwatch, containing all stopwatch functionality.
    class BasicStopwatch{
    private:
        double seconds, milliseconds, nanoseconds;

        /// Start point of timing.
        std::chrono::high_resolution_clock::time_point startPoint;

        /// End point of timing.
        std::chrono::high_resolution_clock::time_point endPoint;

        /// Last lap start point.
        std::chrono::high_resolution_clock::time_point lastLapStart;

        /// Saved lap times.
        std::vector<double> lapTimes;

        /// Save last lap time into lap times.
        /// \param endPoint Last lap time end point.
        void save_lap_time(std::chrono::high_resolution_clock::time_point endPoint);
    public:
        /// Creates basic stopwatch.
        BasicStopwatch();
        /// Creates start point of stopwatches.
        void start();
        /// Reset start point of stopwatches.
        void reset();
        /// Creates end point of stopwatches and calculate elapsed time.
        void stop();
        /// Start new lap/
        void start_new_lap();
        /// End current lap and save its time.
        void end_lap();

        /// Get elapsed time in seconds.
        /// \return Seconds.
        double elapsed_seconds();
        /// Get elapsed time in milliseconds.
        /// \return Milliseconds.
        double elapsed_milliseconds();
        /// Get elapsed time in nanoseconds.
        /// \return Nanoseconds.
        double elapsed_nanoseconds();
        /// Get average time for lap in milliseconds.
        /// \return Milliseconds.
        double average_elapsed_milliseconds();

    };
    /// Id of default stopwatches
    int basicStopwatchId = 0;
    /// Next stopwatches id.
    int nextStopwatchId = 1;
    /// Map of stopwatches.
    std::map<int, Stopwatch::BasicStopwatch> stopwatches;

    /// Check if stopwatch exist in stopwatch map.
    /// \param stopwatchId Id of stopwatch to test.
    /// \return True if stopwatch exists.
    bool stopwatch_exists(int stopwatchId);

    /// Return formatted string of time.
    /// \param milliseconds Number of milliseconds.
    /// \return String.
    std::string format_time(double milliseconds);
    std::string name;

public:
    /// Create stopwatch with optional name.
    /// \param stopwatchName Name of stopwatches.
    Stopwatch(std::string stopwatchName = "");

    /// Start new stopwatch and return its id.
    /// \return Id of new stopwatch.
    int start_new_stopwatch();

    /// Start stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    void start(int stopwatchId = 0);

    /// Stop stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    void stop(int stopwatchId = 0);

    /// Reset stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    void reset(int stopwatchId = 0);

    /// Start new lap for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    void start_new_lap(int stopwatchId = 0);

    /// End last lap for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    void end_lap(int stopwatchId = 0);

    /// Elapsed seconds for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    /// \return Seconds.
    double elapsed_seconds(int stopwatchId = 0);

    /// Elapsed milliseconds for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    /// \return Millieconds.
    double elapsed_milliseconds(int stopwatchId = 0);

    /// Elapsed nanoseconds for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    /// \return Nanoeconds.
    double elapsed_nanoseconds(int stopwatchId = 0);

    /// Average lap time in milliseconds for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    /// \return Milliseconds
    double average_lap_time_in_milliseconds(int stopwatchId = 0);

    /// Get new stopwatch id.
    /// \return Unique integer in.
    int get_next_stopwatch_id();

    /// Get formatted time for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    /// \return Formatted string.
    std::string formatted(int stopwatchId = 0);

    /// Get formatted average time for stopwatch with id.
    /// \param stopwatchId Stopwatch id.
    /// \return Formatted string.
    std::string formatted_average(int stopwatchId = 0);

    /// Get stopwatch name.
    /// \return Name in string.
    const std::string get_name();
};


#endif //BACHELOR_STOPWATCH_H
