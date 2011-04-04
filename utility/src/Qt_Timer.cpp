#include "utility/Timer.h"

#include <QtCore/QDateTime>

#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

namespace oskar {


/**
 * @details
 * Timer class constructor. Creates a new timer object settings initialising
 * and starting the timer.
 */
Timer::Timer()
{
    _timer.start();
}


/**
 * @details
 * Returns the elapsed time in seconds
 */
double Timer::elapsed()
{
    return double(_timer.elapsed()) / 1.0e3;
}


/**
 * @details 
 * Restarts the timer.
 */
void Timer::restart()
{
    _timer.restart();
}


/**
 * @details
 * Prints a formatted message with the elapsed time.
 */
double Timer::message(const QString& message, unsigned indent)
{
    const double timeTaken = elapsed();
    cout << std::fixed << std::setprecision(3);
    cout << "= " << message.toStdString();
    cout << std::setw(indent - message.length());
    cout << "[time = " << timeTaken << "s]";
    cout << endl;
    cout << std::resetiosflags(std::ios::fixed);
    return timeTaken;
}


} // namespace oskar
