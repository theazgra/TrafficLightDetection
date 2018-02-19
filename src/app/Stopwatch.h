//
// Created by azgra on 19.2.18.
//

#ifndef BACHELOR_STOPWATCH_H
#define BACHELOR_STOPWATCH_H

#include <chrono>

class Stopwatch {

private:
    std::chrono::high_resolution_clock::time_point startPoint;
    std::chrono::high_resolution_clock::time_point endPoint;
    std::chrono::duration<double> elapsedTime;
public:
    void start();
    void reset();
    void stop();

    double elapsed();
    double elapledAndReset();

};


#endif //BACHELOR_STOPWATCH_H
