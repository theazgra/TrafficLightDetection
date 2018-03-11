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

class Stopwatch {

private:
    class BasicStopwatch{
    private:
        double seconds, milliseconds, nanoseconds;

        std::chrono::high_resolution_clock::time_point startPoint;
        std::chrono::high_resolution_clock::time_point endPoint;

    public:
        BasicStopwatch();
        void start();
        void reset();
        void stop();

        double elapsedSeconds();
        double elapsedMilliseconds();
        double elapsedNanoseconds();

    };
    int basicStopwatchId = 0;
    int nextStopwatchId = 1;
    std::map<int, Stopwatch::BasicStopwatch> stopwatches;

    bool stopwatch_exists(int stopwatchId);

public:
    Stopwatch();

    void start();
    void start(int stopwatchId);
    int start_new_stopwatch();
    
    void stop();
    void stop(int stopwatchId);
    void reset();
    void reset(int stopwatchId);

    double elapsed_seconds();
    double elapsed_seconds(int stopwatchId);
    double elapsed_milliseconds();
    double elapsed_milliseconds(int stopwatchId);
    double elapsed_nanoseconds();
    double elapsed_nanoseconds(int stopwatchId);

    int get_next_stopwatch_id();

    std::string formatted(int stopwatchId = 0);
};


#endif //BACHELOR_STOPWATCH_H
