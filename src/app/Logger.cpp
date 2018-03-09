//
// Created by azgra on 19.2.18.
//

#include "Logger.h"

using namespace std;

Logger::Logger(std::string logFileName, bool writeToConsole)
{
    this->fileName = logFileName;
    this->consolePrint = writeToConsole;
    this->fileStream.open(logFileName, ios::out | ios::app);
}

Logger::~Logger()
{
    this->fileStream.flush();
    this->fileStream.close();
    cout << "Closed file " << this->fileName << endl;
}

void Logger::write_line(std::string message)
{
    this->fileStream << message << std::endl;

    if (this->consolePrint)
        cout << message << std::endl;
}

void Logger::write(std::string message)
{
    this->fileStream << message;

    if (this->consolePrint)
        cout << message;
}

void Logger::write_line(std::ostream& str)
{
    this->fileStream << str.rdbuf() << std::endl;

    if (this->consolePrint)
        cout << str.rdbuf() << std::endl;
}

void Logger::write(std::ostream& str)
{
    this->fileStream << str.rdbuf();

    if (this->consolePrint)
        cout << str.rdbuf();
}

void Logger::write_lines(std::vector<std::string> messages)
{
    for (string s : messages)
        write_line(s);
}
