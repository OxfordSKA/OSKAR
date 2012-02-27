/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_SYSTEM_CLOCK_TIME_H_
#define OSKAR_SYSTEM_CLOCK_TIME_H_

/**
 * @file oskar_system_clock_time.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns information about the system clock time.
 *
 * @details
 * This function returns a string containing the system date and time in
 * the format "yyyy-mm-dd, hh:mm:ss (timezone)".
 *
 * The data are also optionally returned in a 7-element array containing:
 * - data[0] = year
 * - data[1] = month
 * - data[2] = day of month
 * - data[3] = hour
 * - data[4] = minute
 * - data[5] = second
 * - data[6] = DST flag
 *
 * If not NULL on input, the \p data array must be able to hold at least 7
 * elements.
 *
 * @param[in] utc    If set, return UTC time; else return local time.
 * @param[out] data  If not NULL, this array must contain 7 elements in which
 *                   the system time and date information is returned.
 */
OSKAR_EXPORT
const char* oskar_system_clock_time(int utc, int* data);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SYSTEM_CLOCK_TIME_H_ */
