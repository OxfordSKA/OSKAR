#ifndef TIMER_H_
#define TIMER_H_

/**
 * @file Timer.h
 */

#include <QtCore/QString>
#include <QtCore/QTime>

namespace oskar {

/**
 * @class Timer
 *
 * @brief
 * Timer class.
 *
 * @details
 */

class Timer
{
    public:
        /// Constructs the timer.
        Timer();

        /// Destroys the timer.
        virtual ~Timer() {}

    public:
        /// Returns the elapsed time in seconds
        double elapsed();

        /// Restarts the timer.
        void restart();

        /// Restarts the timer.
        void reset() { restart(); }

        /// Print a formatted message of the elapsed time.
        double message(const QString & message, unsigned indent = 70);

    private:
        QTime _timer;
};

} // namespace oskar
#endif // TIMER_H_
