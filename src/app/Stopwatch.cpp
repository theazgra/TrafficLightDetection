#include "Stopwatch.h"

#pragma region BasicStopwatch
/**
 * Initialize basic stopwatch.
 */
Stopwatch::BasicStopwatch::BasicStopwatch()
{
    this->seconds = 0;
    this->milliseconds = 0;
    this->nanoseconds = 0;
}
/*********************************************************************************************************************************************************/
void Stopwatch::BasicStopwatch::start()
{
    this->startPoint = std::chrono::high_resolution_clock::now();
    this->lastLapStart = startPoint;
}
/*********************************************************************************************************************************************************/
void Stopwatch::BasicStopwatch::reset()
{
    start();
}
/*********************************************************************************************************************************************************/
void Stopwatch::BasicStopwatch::stop()
{
    this->endPoint = std::chrono::high_resolution_clock::now();

    save_lap_time(endPoint);

    auto duration = this->endPoint - this->startPoint;
    this->nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    this->milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    this->seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}
/*********************************************************************************************************************************************************/
double Stopwatch::BasicStopwatch::elapsed_seconds()
{
    return this->seconds;
}
/*********************************************************************************************************************************************************/
double Stopwatch::BasicStopwatch::elapsed_milliseconds()
{
    return this->milliseconds;
}
/*********************************************************************************************************************************************************/
double Stopwatch::BasicStopwatch::elapsed_nanoseconds()
{
    return this->nanoseconds;
}
/*********************************************************************************************************************************************************/
void Stopwatch::BasicStopwatch::save_lap_time(std::chrono::high_resolution_clock::time_point endPoint)
{
    auto lastLapDuration = endPoint - this->lastLapStart;
    double lapMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(lastLapDuration).count();

    this->lapTimes.push_back(lapMilliseconds);
}
/*********************************************************************************************************************************************************/
void Stopwatch::BasicStopwatch::start_new_lap()
{
    this->lastLapStart = std::chrono::high_resolution_clock::now();
}
/*********************************************************************************************************************************************************/
void Stopwatch::BasicStopwatch::end_lap()
{
    auto lapEndPoint = std::chrono::high_resolution_clock::now();
    save_lap_time(lapEndPoint);
}
/*********************************************************************************************************************************************************/
double Stopwatch::BasicStopwatch::average_elapsed_milliseconds()
{
    double totalLapTimes = 0.0;

    for (double lapTime : this->lapTimes)
        totalLapTimes += lapTime;

    return (totalLapTimes / ((double)this->lapTimes.size()));
}
/*********************************************************************************************************************************************************/

#pragma endregion

#pragma region Stopwatch

/*********************************************************************************************************************************************************/
bool Stopwatch::stopwatch_exists(int stopwatchId)
{
    if (!this->stopwatches.count(stopwatchId))
    {
        //throw std::runtime_error("Stopwatch with given id does not exist");
        return false;
    }
    return true;
}
/*********************************************************************************************************************************************************/
Stopwatch::Stopwatch()
{
    this->stopwatches[this->basicStopwatchId] = BasicStopwatch();
}
/*********************************************************************************************************************************************************/
void Stopwatch::start(int stopwatchId)
{
    if (!this->stopwatches.count(stopwatchId))
        this->stopwatches[stopwatchId] = BasicStopwatch();

    this->stopwatches[stopwatchId].start();
}
/*********************************************************************************************************************************************************/
int Stopwatch::start_new_stopwatch()
{
    int stopwatchId = this->nextStopwatchId++;
    this->start(stopwatchId);

    return stopwatchId;
}
/*********************************************************************************************************************************************************/
void Stopwatch::stop(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        this->stopwatches[stopwatchId].stop();
}
/*********************************************************************************************************************************************************/
void Stopwatch::reset(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        this->stopwatches[stopwatchId].reset();
}
/*********************************************************************************************************************************************************/
double Stopwatch::elapsed_seconds(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return this->stopwatches[stopwatchId].elapsed_seconds();

	return 0;
}
/*********************************************************************************************************************************************************/
double Stopwatch::elapsed_milliseconds(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return this->stopwatches[stopwatchId].elapsed_milliseconds();

	return 0;
}
/*********************************************************************************************************************************************************/
double Stopwatch::elapsed_nanoseconds(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return this->stopwatches[stopwatchId].elapsed_nanoseconds();
	
	return 0;
}
/*********************************************************************************************************************************************************/
int Stopwatch::get_next_stopwatch_id()
{
    return this->nextStopwatchId++;
}
/*********************************************************************************************************************************************************/
std::string Stopwatch::formatted(int stopwatchId)
{
    double MS = this->stopwatches[stopwatchId].elapsed_milliseconds();

    return format_time(MS);
}
/*********************************************************************************************************************************************************/
void Stopwatch::start_new_lap(int stopwatchId)
{
    if (!stopwatch_exists(stopwatchId))
        this->stopwatches[stopwatchId] = BasicStopwatch();

    this->stopwatches[stopwatchId].start_new_lap();
}
/*********************************************************************************************************************************************************/
double Stopwatch::average_lap_time_in_milliseconds(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return this->stopwatches[stopwatchId].average_elapsed_milliseconds();

    return 0;
}
/*********************************************************************************************************************************************************/
std::string Stopwatch::formatted_average(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        return format_time(this->stopwatches[stopwatchId].average_elapsed_milliseconds());

   return ("");
}
/*********************************************************************************************************************************************************/
std::string Stopwatch::format_time(double milliseconds)
{
    double M = 0;
    double S = 0;
    double H = 0;

    S = milliseconds / 1000;
    double fS = std::floor(S);
    milliseconds = (S - fS) * 1000;   //We have ms.

    M = fS / 60;
    double fM = std::floor(M);
    S = (M - fM) * 60; //We have s.

    H = fM / 60;
    double fH = std::floor(H);
    M = (H - fH) * 60; //We have m.

    H = fH;

    std::string result = std::to_string(H) + " hours, " + std::to_string(M) + " minutes, " +
                         std::to_string(S) + " seconds, " + std::to_string(milliseconds) + " milliseconds";

    return result;
}
/*********************************************************************************************************************************************************/
void Stopwatch::end_lap(int stopwatchId)
{
    if (stopwatch_exists(stopwatchId))
        this->stopwatches[stopwatchId].end_lap();
}

#pragma endregion
