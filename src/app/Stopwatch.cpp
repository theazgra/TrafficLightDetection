#include "Stopwatch.h"


void Stopwatch::start()
{
    this->startPoint = std::chrono::high_resolution_clock::now();
}
void Stopwatch::reset()
{
    this->startPoint = std::chrono::high_resolution_clock::now();
}

void Stopwatch::stop()
{
    this->endPoint = std::chrono::high_resolution_clock::now();
}

double Stopwatch::elapsed()
{
    this->elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(this->endPoint - this->startPoint);
    return this->elapsedTime.count();
}

double Stopwatch::elapledAndReset()
{
    this->startPoint = std::chrono::high_resolution_clock::now();
    return this->elapsedTime.count();
}
