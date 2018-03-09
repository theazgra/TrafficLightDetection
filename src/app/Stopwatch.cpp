#include "Stopwatch.h"

#pragma region BasicStopwatch
Stopwatch::BasicStopwatch::BasicStopwatch()
{
    this->seconds = 0;
    this->milliseconds = 0;
    this->nanoseconds = 0;
}
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
#pragma endregion
#pragma region Stopwatch
bool Stopwatch::stopwatch_exists(int stopwatchId)
{
    if (!this->stopwatches.count(stopwatchId))
    {
        throw std::runtime_error("Stopwatch with given id does not exist");
        return false;
    }
    return true;
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
        this->stopwatches[stopwatchId] = BasicStopwatch();

    this->stopwatches[stopwatchId].start();
}

int Stopwatch::start_new_stopwatch()
{
    int stopwatchId = this->nextStopwatchId++;
    this->start(stopwatchId);

    return stopwatchId;
}

void Stopwatch::stop()
{
    this->stopwatches[this->basicStopwatchId].stop();
}

void Stopwatch::stop(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        this->stopwatches[stopwatchId].stop();
}

void Stopwatch::reset()
{
    this->stopwatches[this->basicStopwatchId].reset();
}

void Stopwatch::reset(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        this->stopwatches[stopwatchId].reset();
}

double Stopwatch::elapsed_seconds()
{
    return this->stopwatches[this->basicStopwatchId].elapsedSeconds();
}

double Stopwatch::elapsed_seconds(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return this->stopwatches[stopwatchId].elapsedSeconds();

	return 0;
}

double Stopwatch::elapsed_milliseconds()
{
    return this->stopwatches[this->basicStopwatchId].elapsedMilliseconds();
}

double Stopwatch::elapsed_milliseconds(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return this->stopwatches[stopwatchId].elapsedMilliseconds();

	return 0;
}

double Stopwatch::elapsed_nanoseconds()
{
    return this->stopwatches[this->basicStopwatchId].elapsedNanoseconds();
}

double Stopwatch::elapsed_nanoseconds(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return this->stopwatches[stopwatchId].elapsedNanoseconds();
	
	return 0;
}

int Stopwatch::get_next_stopwatch_id()
{
    return this->nextStopwatchId++;
}
#pragma endregion
