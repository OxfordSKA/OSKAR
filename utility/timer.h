#ifndef TIMER_H_
#define TIMER_H_

#include <ctime>
#include <cstdio>

#ifdef _WIN32
    #undef TIMER_ENABLE
#else
    #include "sys/time.h"
#endif

/**
 * @brief
 * Preprocessor macros used for timing.
 *
 * @details
 * This header defines two macros to measure the time interval
 * taken between their calls.
 *
 * Define TIMER_ENABLE to use them.
 *
 * Call TIMER_START to start the timer, and TIMER_STOP(...) to print the time
 * taken (with a message prefix) to standard output (STDOUT). The timer uses
 * QTime and therefore has millisecond accuracy.
 *
 * To use a CPU clock timer instead, call CLOCK_START and CLOCK_STOP(...)
 * instead.
 */
#ifdef TIMER_ENABLE
#define CLOCK_START {clock_t _start_time = clock();
#define CLOCK_STOP(...) clock_t _timer_interval = clock() - _start_time; \
    float _time_sec = (float)(_timer_interval) / CLOCKS_PER_SEC; \
    fprintf(stdout, "\n"); \
    fprintf(stdout, __VA_ARGS__); \
    fprintf(stdout, ": %.2f sec.\n", _time_sec);}
#else
#define CLOCK_START ;
#define CLOCK_STOP(...) ;
#endif

#ifdef TIMER_ENABLE
#define TIMER_START {struct timeval _t1; gettimeofday(&_t1, NULL);
#define TIMER_STOP(...) struct timeval _t2; gettimeofday(&_t2, NULL); \
    double _start = _t1.tv_sec + _t1.tv_usec * 1.0e-6; \
    double _end = _t2.tv_sec + _t2.tv_usec * 1.0e-6; \
    fprintf(stdout, "\n"); \
    fprintf(stdout, __VA_ARGS__); \
    fprintf(stdout, ": %.3f sec.\n", _end - _start);};
#else
#define TIMER_START ;
#define TIMER_STOP(...) ;
#endif

#endif /* TIMER_H_ */
