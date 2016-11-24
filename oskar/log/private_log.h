/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#ifndef OSKAR_PRIVATE_LOG_H_
#define OSKAR_PRIVATE_LOG_H_

/**
 * @file private_log.h
 */

#include <oskar_global.h>
#include <stdio.h>
#include <time.h>

struct oskar_Log
{
    /* Variables to control which log entries will be printed */
    int file_priority; /**< Controls which log messages are printed in the log file */
    int term_priority; /**< Controls which log messages are printed in the terminal */

    /* The column start index for value log entries. */
    int value_width;   /**< Width of value message */

    /* File pointer to the log file */
    FILE* file;        /**< Pointer to an open log file stream. */
    int keep_file;     /**< Flag, true if log file will be retained on completion. */

    /* Log record meta-data */
    int size;          /**< The number of entries in the log. */
    int capacity;      /**< The capacity of the arrays in this structure. */
    char* name;        /**< The name of the log file. */
    char* code;        /**< Array containing the code of each entry. */
    int* offset;       /**< Array containing the memory offsets in bytes of each entry. */
    int* length;       /**< Array containing the length in bytes of each entry. */
    time_t* timestamp; /**< Array containing log time stamps. */
};

#ifndef OSKAR_LOG_TYPEDEF_
#define OSKAR_LOG_TYPEDEF_
typedef struct oskar_Log oskar_Log;
#endif /* OSKAR_LOG_TYPEDEF_ */

/* Private function. */
#include <log/oskar_log_write.h>

#endif /* OSKAR_PRIVATE_LOG_H_ */
