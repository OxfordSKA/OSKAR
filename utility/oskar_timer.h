/*
 * Copyright (c) 2013, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_TIMER_H_
#define OSKAR_TIMER_H_

/**
 * @file oskar_timer.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Timer;
#ifndef OSKAR_TIMER_TYPEDEF_
#define OSKAR_TIMER_TYPEDEF_
typedef struct oskar_Timer oskar_Timer;
#endif /* OSKAR_TIMER_TYPEDEF_ */

enum OSKAR_TIMER_TYPE
{
    OSKAR_TIMER_NATIVE,
    OSKAR_TIMER_CUDA,
    OSKAR_TIMER_OMP
};

/**
 * @brief Creates a timer.
 *
 * @details
 * Creates a timer. The timer is created in a paused state.
 *
 * The \p type parameter may take the values:
 *
 * - OSKAR_TIMER_CUDA
 * - OSKAR_TIMER_OMP
 * - OSKAR_TIMER_NATIVE
 *
 * These timers are the ones provided by, respectively, CUDA, OpenMP or the
 * native system.
 *
 * @param[in,out] timer Pointer to timer.
 * @param[in] type Type of timer to create.
 */
OSKAR_EXPORT
oskar_Timer* oskar_timer_create(int type);

/**
 * @brief Destroys the timer.
 *
 * @details
 * Destroys the timer.
 *
 * @param[in,out] timer Pointer to timer.
 */
OSKAR_EXPORT
void oskar_timer_free(oskar_Timer* timer);

/**
 * @brief Returns the total elapsed time, in seconds.
 *
 * @details
 * Returns the number of seconds since the timer was started.
 *
 * @param[in,out] timer Pointer to timer.
 *
 * @return The number of seconds since the timer was started.
 */
OSKAR_EXPORT
double oskar_timer_elapsed(oskar_Timer* timer);

/**
 * @brief Pauses the timer.
 *
 * @details
 * Pauses the timer.
 *
 * @param[in,out] timer Pointer to timer.
 */
OSKAR_EXPORT
void oskar_timer_pause(oskar_Timer* timer);

/**
 * @brief Resumes the timer.
 *
 * @details
 * Resumes the timer from a paused state.
 *
 * @param[in,out] timer Pointer to timer.
 */
OSKAR_EXPORT
void oskar_timer_resume(oskar_Timer* timer);

/**
 * @brief Restarts the timer.
 *
 * @details
 * Restarts the timer.
 *
 * @param[in,out] timer Pointer to timer.
 */
OSKAR_EXPORT
void oskar_timer_restart(oskar_Timer* timer);

/**
 * @brief Starts and resets the timer.
 *
 * @details
 * Starts and resets the timer, clearing the current elapsed time.
 *
 * @param[in,out] timer Pointer to timer.
 */
OSKAR_EXPORT
void oskar_timer_start(oskar_Timer* timer);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TIMER_H_ */
