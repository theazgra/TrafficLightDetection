#ifndef BACHELOR_LOGGER_H
#define BACHELOR_LOGGER_H

#include <string>
#include <vector>
#include <fstream>

class Logger {

private:
    std::fstream fileStream;
public:
    Logger(std::string logFileName);

    void write_line(std::string message);
    void write(std::string message);
    void write_lines(std::vector<std::string> messages);

    ~Logger();
};


#endif //BACHELOR_LOGGER_H
