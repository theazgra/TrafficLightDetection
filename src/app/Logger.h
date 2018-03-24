/** \file Logger.h
 * Logger class.
 */

#ifndef BACHELOR_LOGGER_H
#define BACHELOR_LOGGER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

/// Logger class used to log into file or standart output.
class Logger {

private:
    /// File stream of output file.
    std::fstream fileStream;

    /// If should print to standart output.
    bool consolePrint = false;

    /// If should log to file.
    bool writeToFile;

    /// Log file name.
    std::string fileName;

public:
    /// Creates logger instance.
    /// \param logFileName Log file name.
    /// \param writeToFile If should log to file.
    /// \param printToConsole If should log to standart output.
    Logger(std::string logFileName, bool writeToFile = false, bool printToConsole = true);

    /// Log line.
    /// \param message String to log.
    void write_line(std::string message);

    /// Log line.
    /// \param str Log stream.
    void write_line(std::ostream& str);

    /// Log message without line terminator.
    /// \param message String to log.
    void write(std::string message);

    /// Log message without line terminator.
    /// \param str Log stream.
    void write(std::ostream& str);

    /// Log multiple lines with line terminator.
    /// \param messages Vector of strings.
    void write_lines(std::vector<std::string> messages);

    /// Enable logging to file.
    void enable_writing_to_file();

    /// Disable logging to file.
    void disable_writing_to_file();

    /// Destruct instance, if file stream is opened, flushes and closes it.
    ~Logger();
};


#endif //BACHELOR_LOGGER_H
