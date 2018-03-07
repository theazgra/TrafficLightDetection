#include "Stopwatch.h"


void Stopwatch::BasicStopwatch::start()
{
    this->startPoint = std::chrono::high_resolution_clock::now();
}
void Stopwatch::BasicStopwatch::reset()
{
    this->startPoint = std::chrono::high_resolution_clock::now();
}

void Stopwatch::BasicStopwatch::stop()
{
    this->endPoint = std::chrono::high_resolution_clock::now();

    auto duration = this->endPoint - this->startPoint;
    this->nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    this->milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    this->seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

double Stopwatch::BasicStopwatch::elapsedSeconds()
{
    return this->seconds;
}

double Stopwatch::BasicStopwatch::elapsedMilliseconds()
{
    return this->milliseconds;
}

double Stopwatch::BasicStopwatch::elapsedNanoseconds()
{
    return this->nanoseconds;
}

Stopwatch::Stopwatch()
{
    this->stopwatches[this->basicStopwatchId] = BasicStopwatch();
}

void Stopwatch::start()
{
    this->stopwatches[this->basicStopwatchId].start();
}

void Stopwatch::start(int stopwatchId)
{
    if (!this->stopwatches.count(stopwatchId))
        this->stopwatches[stopwatchId] = BasicStopwatch() ;

    this->stopwatches[stopwatchId].start();
}

void Stopwatch::stop()
{
    this->stopwatches[this->basicStopwatchId].stop();
}

void Stopwatch::stop(int stopwatchId)
{
    this->stopwatches[stopwatchId].stop();
}

void Stopwatch::reset()
{
    this->stopwatches[this->basicStopwatchId].reset();
}

void Stopwatch::reset(int stopwatchId)
{
    this->stopwatches[stopwatchId].reset();
}

double Stopwatch::elapsed_seconds()
{
    return this->stopwatches[this->basicStopwatchId].elapsedSeconds();
}

double Stopwatch::elapsed_seconds(int stopwatchId)
{
    return this->stopwatches[stopwatchId].elapsedSeconds();
}

double Stopwatch::elapsed_milliseconds()
{
    return this->stopwatches[this->basicStopwatchId].elapsedMilliseconds();
}

double Stopwatch::elapsed_milliseconds(int stopwatchId)
{
    return this->stopwatches[stopwatchId].elapsedMilliseconds();
}

double Stopwatch::elapsed_nanoseconds()
{
    return this->stopwatches[this->basicStopwatchId].elapsedNanoseconds();
}

double Stopwatch::elapsed_nanoseconds(int stopwatchId)
{
    return this->stopwatches[stopwatchId].elapsedNanoseconds();
}

int Stopwatch::get_next_stopwatch_id()
{
    return this->nextStopwatchId++;
}