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

#include "utility/oskar_timers_functions.h"
#include "utility/oskar_timer_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_timers_create(oskar_Timers* timers, int type)
{
    oskar_timer_create(&timers->tmr, type);
    oskar_timer_create(&timers->tmr_init_copy, type);
    oskar_timer_create(&timers->tmr_clip, type);
    oskar_timer_create(&timers->tmr_R, type);
    oskar_timer_create(&timers->tmr_E, type);
    oskar_timer_create(&timers->tmr_K, type);
    oskar_timer_create(&timers->tmr_join, type);
    oskar_timer_create(&timers->tmr_correlate, type);
}

void oskar_timers_destroy(oskar_Timers* timers)
{
    oskar_timer_destroy(&timers->tmr);
    oskar_timer_destroy(&timers->tmr_init_copy);
    oskar_timer_destroy(&timers->tmr_clip);
    oskar_timer_destroy(&timers->tmr_R);
    oskar_timer_destroy(&timers->tmr_E);
    oskar_timer_destroy(&timers->tmr_K);
    oskar_timer_destroy(&timers->tmr_join);
    oskar_timer_destroy(&timers->tmr_correlate);
}

#ifdef __cplusplus
}
#endif
