/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#include "log/private_log.h"
#include "log/oskar_log.h"
#include "oskar_version.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef OSKAR_OS_WIN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static int oskar_log_file_exists(const char* filename)
{
    FILE* stream;
    if (!filename || !*filename) return 0;
    stream = fopen(filename, "r");
    if (stream)
    {
        fclose(stream);
        return 1;
    }
    return 0;
}

void oskar_log_create(int file_priority, int term_priority)
{
    oskar_Log* log = 0;
    size_t buf_len = 0;
    struct tm* timeinfo;
    char fname1[64], fname2[64], time_str[80], *current_dir = 0;
    int i = 0, n = 0;

    /* Get handle to log. */
    log = oskar_log_handle();
    if (log->file)
    {
        fclose(log->file);
        log->file = 0;
    }
    if (log->name)
    {
        free(log->name);
        log->name = 0;
    }
    /*log = (oskar_Log*) calloc(1, sizeof(oskar_Log));*/
    /*if (!log) return 0;*/

    /* Initialise memory for the log data. */
    log->keep_file = 1;
    log->file_priority = file_priority;
    log->term_priority = term_priority;
    log->value_width = OSKAR_LOG_DEFAULT_VALUE_WIDTH;

    /* Construct log file name root. */
    const time_t unix_time = time(NULL);
    timeinfo = localtime(&unix_time);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)", timeinfo);
    timeinfo->tm_mon += 1;
    timeinfo->tm_year += 1900;
#if __STDC_VERSION__ >= 199901L
    n = snprintf(fname1, sizeof(fname1),
            "oskar_%.4d-%.2d-%.2d_%.2d%.2d%.2d",
            timeinfo->tm_year, timeinfo->tm_mon, timeinfo->tm_mday,
            timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
#else
    n = sprintf(fname1,
            "oskar_%.4d-%.2d-%.2d_%.2d%.2d%.2d",
            timeinfo->tm_year, timeinfo->tm_mon, timeinfo->tm_mday,
            timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
#endif
    if (n < 0 || n >= (int)sizeof(fname1))
        return;

    /* Construct a unique log file name. */
    do
    {
        ++i;
        if (i == 1)
        {
#if __STDC_VERSION__ >= 199901L
            n = snprintf(fname2, sizeof(fname2), "%s.log", fname1);
#else
            n = sprintf(fname2, "%s.log", fname1);
#endif
        }
        else
        {
#if __STDC_VERSION__ >= 199901L
            n = snprintf(fname2, sizeof(fname2), "%s_%d.log", fname1, i);
#else
            n = sprintf(fname2, "%s_%d.log", fname1, i);
#endif
        }
        if (n < 0 || n >= (int)sizeof(fname1))
            return;
    }
    while (oskar_log_file_exists(fname2));

    /* Open the log file for appending, and save the file handle. */
    if (log->file_priority > OSKAR_LOG_NONE)
        log->file = fopen(fname2, "a+");

    /* Write standard header. */
    oskar_log_section('M', "OSKAR-%s starting at %s.",
            OSKAR_VERSION_STR, time_str);

    /* Get the current working directory. */
#ifdef OSKAR_OS_WIN
    buf_len = GetCurrentDirectory(0, NULL);
    current_dir = (char*) calloc(buf_len, sizeof(char));
    GetCurrentDirectory((DWORD) buf_len, current_dir);
#else
    do
    {
        buf_len += 256;
        current_dir = (char*) realloc(current_dir, buf_len);
    }
    while (getcwd(current_dir, buf_len) == NULL);
#endif
    oskar_log_message('M', 0, "Current dir is %s", current_dir);
    free(current_dir);

    /* Write a message to say if the log file was opened successfully. */
    if (log->file)
    {
        /* Save the file name in the structure. */
        log->name = (char*) calloc(n + 1, sizeof(char));
        strcpy(log->name, fname2);
        oskar_log_message('M', 0, "Logging to file %s", fname2);
    }
    else if (log->file_priority > OSKAR_LOG_NONE)
    {
        oskar_log_warning("File error: log file could not be created.");
    }
}

#ifdef __cplusplus
}
#endif
