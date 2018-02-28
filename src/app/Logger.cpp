//
// Created by azgra on 19.2.18.
//

#include "Logger.h"

using namespace std;

Logger::Logger(std::string logFileName) {

    this->fileStream.open(logFileName, ios::out | ios::app);
}

Logger::~Logger() {
    this->fileStream.flush();
    this->fileStream.close();
}

void Logger::write_line(std::string message) {
    this->fileStream << message << std::endl;
    cout << message << std::endl;
}

void Logger::write(std::string message) {
    this->fileStream << message;
    cout << message;
}

void Logger::write_lines(std::vector<std::string> messages) {
    for (string s : messages)
        write_line(s);
}
