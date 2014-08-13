/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <private_log.h>
#include <oskar_log.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Static function prototype. */
static void print_entry(FILE* stream, char code, int depth, int width,
        const char* prefix, const char* format, va_list args);

void oskar_log_write(oskar_Log* log, char code, int depth, int width,
        const char* prefix, const char* format, ...)
{
    va_list args;

    /* Write to standard output. */
    va_start(args, format);
    oskar_log_writev_stdout(code, depth, width, prefix, format, args);
    va_end(args);

    /* Write to log file. */
    va_start(args, format);
    oskar_log_writev(log, code, depth, width, prefix, format, args);
    va_end(args);
}

void oskar_log_writev(oskar_Log* log, char code, int depth, int width,
        const char* prefix, const char* format, va_list args)
{
    /* Catch if both strings are NULL. */
    if (!format && !prefix && depth > -10)
        return;

    /* Return if no log is set. */
    if (!log)
        return;

#ifdef _OPENMP
    /* Lock mutex. */
    omp_set_lock(&log->mutex);
#endif

    /* Resize arrays in log structure if required. */
    if (log->size % 100 == 0)
    {
        int i;
        i = log->size + 100;
        log->code = realloc(log->code, i);
        log->offset = realloc(log->offset, i * sizeof(int));
        log->length = realloc(log->length, i * sizeof(int));
        log->timestamp = realloc(log->timestamp, i * sizeof(time_t));
        log->capacity = i;
    }

    /* Store the code and time stamp of the entry. */
    log->code[log->size] = code;
    log->timestamp[log->size] = time(NULL);
    log->offset[log->size] = 0;
    log->length[log->size] = 0;

    /* Check if log file exists. */
    if (log->file)
    {
        /* Store the offset of the entry. */
        log->offset[log->size] = ftell(log->file);

        /* Write the entry. */
        print_entry(log->file, code, depth, width, prefix, format, args);

        /* Store the length of the log entry. */
        log->length[log->size] = ftell(log->file) -
                log->offset[log->size];
    }

    /* Increment the log entry counter. */
    log->size++;

#ifdef _OPENMP
    /* Unlock mutex. */
    omp_unset_lock(&log->mutex);
#endif
}

void oskar_log_writev_stderr(char code, int depth, int width,
        const char* prefix, const char* format, va_list args)
{
    /* Catch if both strings are NULL. */
    if (!format && !prefix && depth > -10) return;

    /* Print log entry to standard error. */
    print_entry(stderr, code, depth, width, prefix, format, args);
    fflush(stderr);
}

void oskar_log_writev_stdout(char code, int depth, int width,
        const char* prefix, const char* format, va_list args)
{
    /* Catch if both strings are NULL. */
    if (!format && !prefix && depth > -10) return;

    /* Print log entry to standard output. */
    print_entry(stdout, code, depth, width, prefix, format, args);
    fflush(stdout);
}

static void print_entry(FILE* stream, char code, int depth, int width,
        const char* prefix, const char* format, va_list args)
{
    int i;
    const char* sym;

    /* Ensure code is a printable character. */
    if (code < 32) code += 48;

    /* Check if depth signifies a line. */
    if (depth == -1000)
    {
        fprintf(stream, " |");
        for (i = 0; i < 68; ++i)
            fprintf(stream, "%c", code);
        fprintf(stream, "\n");
        return;
    }

    /* Print a blank line around section headings and errors. */
    if (depth == -100)
        fprintf(stream, " |\n");

    /* Print the message code. */
    fprintf(stream, "%c|", code);

    /* Print leading whitespace for depth. */
    if (abs(depth) < 50)
        for (i = 0; i < abs(depth); ++i) fprintf(stream, "  ");

    /* Print symbol for this depth.
     * All symbols should contain the same number of characters. */
    switch (depth)
    {
    case -100: /* Special case. */
        sym = "==";
        break;
    case -101:
        sym = "";
        break;
    case 0:
        sym = " +";
        break;
    case 1:
        sym = " -";
        break;
    case 2:
        sym = " *";
        break;
    case 3:
        sym = " +";
        break;
    case 4:
        sym = " -";
        break;
    case 5:
        sym = " *";
        break;
    case 6:
        sym = " +";
        break;
    case 7:
        sym = " -";
        break;
    case 8:
        sym = " *";
        break;
    default: /* Negative depth means no symbol. */
        sym = "  ";
        break;
    }
    fprintf(stream, "%s ", sym);

    /* Check if a prefix string is present. */
    if (prefix && *prefix > 0)
    {
        /* Print prefix. */
        fprintf(stream, "%s", prefix);

        /* Print trailing whitespace if format string is present. */
        if (format && *format > 0)
        {
            int n;
            n = abs(2 * depth + 4 + strlen(prefix));
            for (i = 0; i < width - n; ++i) fprintf(stream, " ");
            fprintf(stream, ": ");
        }
    }

    /* Print main message from format string and arguments. */
    if (format && *format > 0)
    {
        vfprintf(stream, format, args);
    }
    fprintf(stream, "\n");

    /* Print a blank line around section headings and errors. */
    if (depth == -100)
        fprintf(stream, " |\n");
}

#ifdef __cplusplus
}
#endif
