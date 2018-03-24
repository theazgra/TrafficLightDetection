//
// Created by azgra on 19.2.18.
//

#include "Logger.h"

using namespace std;

Logger::Logger(std::string logFileName, bool writeToFile, bool printToConsole)
{
    this->fileName = logFileName;
    this->consolePrint = printToConsole;
    this->writeToFile  = writeToFile;

    if (this->writeToFile)
    {
        this->fileStream.open(logFileName, ios::out | ios::app);
    }
}
/*********************************************************************************************************************************************************/
Logger::~Logger()
{
    if (this->fileStream.is_open())
    {
        this->fileStream.flush();
        this->fileStream.close();
    }
}
/*********************************************************************************************************************************************************/
void Logger::write_line(std::string message)
{
    if (this->writeToFile)
        this->fileStream << message << std::endl;

    if (this->consolePrint)
        cout << message << std::endl;
}
/*********************************************************************************************************************************************************/
void Logger::write(std::string message)
{
    if (this->writeToFile)
        this->fileStream << message;

    if (this->consolePrint)
        cout << message;
}
/*********************************************************************************************************************************************************/
void Logger::write_line(std::ostream& str)
{
    if (this->writeToFile)
        this->fileStream << str.rdbuf() << std::endl;

    if (this->consolePrint)
        cout << str.rdbuf() << std::endl;
}
/*********************************************************************************************************************************************************/
void Logger::write(std::ostream& str)
{
    if (this->writeToFile)
        this->fileStream << str.rdbuf();

    if (this->consolePrint)
        cout << str.rdbuf();
}
/*********************************************************************************************************************************************************/
void Logger::write_lines(std::vector<std::string> messages)
{
    for (string s : messages)
        write_line(s);
}
/*********************************************************************************************************************************************************/
void Logger::disable_writing_to_file()
{
    this->writeToFile = false;

    if (this->fileStream.is_open())
    {
        this->fileStream.flush();
        this->fileStream.close();
    }
}
/*********************************************************************************************************************************************************/
void Logger::enable_writing_to_file()
{
    this->writeToFile = true;
    if (!this->fileStream.is_open())
    {
        this->fileStream.open(this->fileName, ios::out | ios::app);
    }
}