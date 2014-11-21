/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_LOG_H_
#define OSKAR_LOG_H_

/**
 * @file oskar_log.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Log;
#ifndef OSKAR_LOG_TYPEDEF_
#define OSKAR_LOG_TYPEDEF_
typedef struct oskar_Log oskar_Log;
#endif /* OSKAR_LOG_TYPEDEF_ */

#ifdef __cplusplus
}
#endif

#define OSKAR_LOG_DEFAULT_PRIORITY    3  /* 3 = OSKAR_LOG_STATUS (or code 'S') */
#define OSKAR_LOG_DEFAULT_VALUE_WIDTH 40 /* Default width for value log entries */

enum OSKAR_LOG_SPECIAL_DEPTH {
    OSKAR_LOG_NO_LIST_MARKER = -1,
    OSKAR_LOG_INFO_PREFIX    = -101,
    OSKAR_LOG_SECTION        = -102,
    OSKAR_LOG_LINE           = -1000
};

/* Enum describing the logging level.
 * The logging level determines the maximum verbosity of the log
 * ie. settings the level to DEBUG will print all logs, and setting it WARN
 * will only print warning logs
 */
enum OSKAR_LOG_PRIORITY {
    OSKAR_LOG_NONE    = -1,
    OSKAR_LOG_ERROR   =  0,  /* == 'E' */
    OSKAR_LOG_WARNING =  1,  /* == 'W' */
    OSKAR_LOG_MESSAGE =  2,  /* == 'M' */
    OSKAR_LOG_STATUS  =  3,  /* == 'S' */
    OSKAR_LOG_DEBUG   =  4   /* == 'D' */
};

#include <oskar_log_accessors.h>
#include <oskar_log_create.h>
#include <oskar_log_error.h>
#include <oskar_log_file_data.h>
#include <oskar_log_file_exists.h>
#include <oskar_log_free.h>
#include <oskar_log_line.h>
#include <oskar_log_message.h>
#include <oskar_log_section.h>
#include <oskar_log_system_clock_data.h>
#include <oskar_log_system_clock_string.h>
#include <oskar_log_value.h>
#include <oskar_log_warning.h>

#endif /* OSKAR_LOG_H_ */
