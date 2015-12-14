/*
 * Copyright (c) 2012-2014, The University of Oxford
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

/* Static function prototypes. */
static void print_entry(FILE* stream, char priority, char code, int depth,
        const char* prefix, int width, const char* format, va_list args);
static void oskar_log_update_record(oskar_Log* log, char code);
static int oskar_log_priority_level(char code);
static int should_print_term_entry(oskar_Log* log, char priority);
static int should_print_file_entry(oskar_Log* log, char priority);

void oskar_log_write(oskar_Log* log, FILE* stream, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args)
{
    int width = 0, is_file = 0;

    /* If both strings are NULL and not printing a line the entry is invalid */
    if (!format && !prefix && depth != OSKAR_LOG_LINE) return;

    width = log ? log->value_width : OSKAR_LOG_DEFAULT_VALUE_WIDTH;
    is_file = (stream == stdout || stream == stderr) ? 0 : 1;

#ifdef _OPENMP
    omp_set_lock(&log->mutex); /* Lock the mutex. */
#endif

    /* Write the entry to the terminal */
    if (!is_file && should_print_term_entry(log, priority))
    {
        print_entry(stream, priority, code, depth, prefix, width, format, args);
        fflush(stream);
    }

    /* Write the entry to the log file */
    else if (is_file && log && log->file && should_print_file_entry(log, priority))
    {
        print_entry(log->file, priority, code, depth, prefix, width, format, args);
        oskar_log_update_record(log, code);
    }

#ifdef _OPENMP
    omp_unset_lock(&log->mutex); /* Unlock the mutex. */
#endif
}

/*
 * Update the record meta data. As the length of a log entry cannot be known
 * a-priori in C89 the length of each entry is determined from the log file
 *
 * **IMPORTANT**: This function should be called **AFTER** the log entry has
 * been written to file and inside a omp mutex lock (if using OpenMP threads)
 */
static void oskar_log_update_record(oskar_Log* log, char code)
{
    if (!log) return;

    /* Resize arrays in log structure if required. */
    if (log->size % 100 == 0)
    {
        int i;
        i = log->size + 100;
        log->code      = realloc(log->code, i);
        log->offset    = realloc(log->offset, i * sizeof(int));
        log->length    = realloc(log->length, i * sizeof(int));
        log->timestamp = realloc(log->timestamp, i * sizeof(time_t));
        log->capacity  = i;
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
        if (log->size > 0)
            log->offset[log->size] = log->offset[log->size-1] + log->length[log->size-1];
        else
            log->offset[log->size] = 0;

        /* Store the length of the log entry. */
        log->length[log->size] = ftell(log->file) - log->offset[log->size];
    }

    /* Increment the log entry counter. */
    log->size++;
}

char oskar_log_get_entry_code(char priority)
{
    switch (priority) {
    case 'E':
    case 'e':
        return 'E';
    case 'W':
    case 'w':
        return 'W';
    case 'M':
    case 'm':
    case ' ':
    case 'S': /* Status */
    case 's': /* Status */
        return ' ';
    case 'D':
    case 'd':
        return 'D';
    default:
        return 'U';
    };
    return 'U';
}

static void print_entry(FILE* stream, char priority, char code, int depth,
        const char* prefix, int width, const char* format, va_list args)
{
    int i;
    const char* sym;

    /* Ensure code is a printable character. */
    if (code < 32) code += 48;

    /* Check if depth signifies a line. */
    if (depth == OSKAR_LOG_LINE)
    {
        char priority_code = oskar_log_get_entry_code(priority);
        fprintf(stream, "%c|", priority_code);
        for (i = 0; i < 67; ++i) fprintf(stream, "%c", code);
        fprintf(stream, "\n");
        return;
    }

    /* Print the message code. */
    fprintf(stream, "%c|", code);

    /* Print leading whitespace and symbol for this depth. */
    if (depth >= 0) {
        char list_symbols[3] = {'+', '-', '*'};
        for (i = 0; i < depth; ++i) fprintf(stream, "  ");
        fprintf(stream, " %c ", list_symbols[depth%3]);
    }
    else {
        /* Negative depth codes with special meaning */
        switch (depth)
        {
        case OSKAR_LOG_NO_LIST_MARKER:
            sym = " ";
            break;
        case OSKAR_LOG_INFO_PREFIX:
        case OSKAR_LOG_SECTION:
            sym = "";
            break;
        default: /* Negative depth means no symbol. */
            sym = "   ";
            break;
        }
        fprintf(stream, "%s", sym);
    }

    /* Check if a prefix string is present. */
    if (prefix && *prefix > 0)
    {
        /* Print prefix. */
        fprintf(stream, "%s", prefix);

        /* Print trailing whitespace if format string is present. */
        if (format && *format > 0)
        {
            int n;
            n = abs(2 * depth + 4 + (int)strlen(prefix));
            for (i = 0; i < width - n; ++i) fprintf(stream, " ");
            if (depth != OSKAR_LOG_SECTION) fprintf(stream, ": ");
        }
    }

    /* Print main message from format string and arguments. */
    if (format && *format > 0)
    {
        vfprintf(stream, format, args);
    }
    fprintf(stream, "\n");
}

/*
 * Returns the enumerated priority level for the given message code.
 *
 * Code to integer conversion is performed to make priority comparison
 * easier
 */
static int oskar_log_priority_level(char code)
{
    switch (code) {
    case 'n':
    case 'N':
        return OSKAR_LOG_NONE;
    case 'e':
    case 'E':
        return OSKAR_LOG_ERROR;
    case 'w':
    case 'W':
        return OSKAR_LOG_WARNING;
    case 'm':
    case 'M':
        return OSKAR_LOG_MESSAGE;
    case 's':
    case 'S':
        return OSKAR_LOG_STATUS;
    case 'd':
    case 'D':
        return OSKAR_LOG_DEBUG;
    default:
        return OSKAR_LOG_NONE;
    };
    return OSKAR_LOG_NONE;
}

static int should_print_term_entry(oskar_Log* log, char priority)
{
    int log_priority = log ? log->term_priority : OSKAR_LOG_DEFAULT_PRIORITY;
    return (oskar_log_priority_level(priority) <= log_priority)? 1 : 0;
}

static int should_print_file_entry(oskar_Log* log, char priority)
{
    int log_priority = log ? log->file_priority : OSKAR_LOG_DEFAULT_PRIORITY;
    return (oskar_log_priority_level(priority) <= log_priority)? 1 : 0;
}

#ifdef __cplusplus
}
#endif
