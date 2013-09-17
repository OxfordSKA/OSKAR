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

#ifndef OSKAR_TIMERS_H_
#define OSKAR_TIMERS_H_

/**
 * @file oskar_timers.h
 */

#include <oskar_global.h>
#include <oskar_timer.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct oskar_Timers
 *
 * @brief Structure to hold all simulation timers.
 *
 * @details
 * The structure holds all simulation timers.
 */
struct OSKAR_EXPORT oskar_Timers
{
    oskar_Timer* tmr;
    oskar_Timer* tmr_init_copy;
    oskar_Timer* tmr_clip;
    oskar_Timer* tmr_correlate;
    oskar_Timer* tmr_join;
    oskar_Timer* tmr_R;
    oskar_Timer* tmr_E;
    oskar_Timer* tmr_K;
};
typedef struct oskar_Timers oskar_Timers;

/**
 * @brief Creates simulation timers.
 *
 * @details
 * Creates all simulation timers. The timers are created in a paused state.
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
 * @param[in,out] timer Pointer to timers.
 * @param[in] type Type of timers to create.
 */
OSKAR_EXPORT
void oskar_timers_create(oskar_Timers* timer, int type);

/**
 * @brief Destroys simulation timers.
 *
 * @details
 * Destroys all simulation timers.
 *
 * @param[in,out] timers Pointer to timers.
 */
OSKAR_EXPORT
void oskar_timers_free(oskar_Timers* timers);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TIMERS_H_ */
