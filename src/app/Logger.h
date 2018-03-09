#ifndef BACHELOR_LOGGER_H
#define BACHELOR_LOGGER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class Logger {

private:
    std::fstream fileStream;
    bool consolePrint = false;
    std::string fileName;
public:
    Logger(std::string logFileName, bool writeToConsole = true);

    void write_line(std::string message);
    void write_line(std::ostream& str);
    void write(std::string message);
    void write(std::ostream& str);
    void write_lines(std::vector<std::string> messages);



    ~Logger();
};


#endif //BACHELOR_LOGGER_H
