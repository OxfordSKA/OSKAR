/*
 * Copyright (c) 2014, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
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

#ifndef OSKAR_LOG_SYSTEM_CLOCK_DATA_H_
#define OSKAR_LOG_SYSTEM_CLOCK_DATA_H_

/**
 * @file oskar_log_system_clock_data.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns information about the system clock time and date.
 *
 * @details
 * This function fills a 9-element array containing information about the
 * system time and date, in the order:
 *
 * - data[0] = seconds after the minute (range 0-61).
 * - data[1] = minutes after the hour (range 0-59).
 * - data[2] = hours since midnight (range 0-23).
 * - data[3] = day of the month (range 1-31).
 * - data[4] = month number (range 1-12).
 * - data[5] = year.
 * - data[6] = days since Sunday (range 0-6).
 * - data[7] = days since 1 January (range 0-365).
 * - data[8] = DST flag
 *
 * @param[in]  utc   If set, return UTC time; else return local time.
 * @param[out] data  A 9-element array in which the system time and date
 *                   information is returned.
 */
OSKAR_EXPORT
void oskar_log_system_clock_data(int utc, int* data);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_LOG_SYSTEM_CLOCK_DATA_H_ */
