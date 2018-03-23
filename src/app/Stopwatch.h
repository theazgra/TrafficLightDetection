//
// Created by azgra on 19.2.18.
//

#ifndef BACHELOR_STOPWATCH_H
#define BACHELOR_STOPWATCH_H

#include <map>
#include <chrono>
#include <exception>
#include <string>
#include <tgmath.h>
#include <vector>

class Stopwatch {

private:
    class BasicStopwatch{
    private:
        double seconds, milliseconds, nanoseconds;

        std::chrono::high_resolution_clock::time_point startPoint;
        std::chrono::high_resolution_clock::time_point endPoint;
        std::chrono::high_resolution_clock::time_point lastLapStart;

        std::vector<double> lapTimes;
        void save_lap_time(std::chrono::high_resolution_clock::time_point endPoint);
    public:
        BasicStopwatch();
        void start();
        void reset();
        void stop();
        void start_new_lap();
        void end_lap();

        double elapsed_seconds();
        double elapsed_milliseconds();
        double elapsed_nanoseconds();

        double average_elapsed_milliseconds();

    };
    int basicStopwatchId = 0;
    int nextStopwatchId = 1;
    std::map<int, Stopwatch::BasicStopwatch> stopwatches;

    bool stopwatch_exists(int stopwatchId);
    std::string format_time(double milliseconds);

public:
    Stopwatch();
    int start_new_stopwatch();

    void start(int stopwatchId = 0);
    void stop(int stopwatchId = 0);
    void reset(int stopwatchId = 0);

    void start_new_lap(int stopwatchId = 0);
    void end_lap(int stopwatchId = 0);

    double elapsed_seconds(int stopwatchId = 0);
    double elapsed_milliseconds(int stopwatchId = 0);
    double elapsed_nanoseconds(int stopwatchId = 0);

    double average_lap_time_in_milliseconds(int stopwatchId = 0);

    int get_next_stopwatch_id();

    std::string formatted(int stopwatchId = 0);
    std::string formatted_average(int stopwatchId = 0);
};


#endif //BACHELOR_STOPWATCH_H
