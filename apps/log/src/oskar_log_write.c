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
static void print_entry(FILE* stream, oskar_Log* log, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args);

/*
 *
 */
static void print_prefix(FILE* stream, char code, const char* prefix);
static void print_list_symbol(FILE* stream, int depth);
static void print_message(FILE* stream, const char* format, va_list args);

static int oskar_log_priority_level(char code);
static int should_print_term_entry(oskar_Log* log, char priority);
static int should_print_file_entry(oskar_Log* log, char priority);

/*---------------------------------------------------------------------------*/

void oskar_log_writev(oskar_Log* log, char priority, char code, int depth,
        const char* prefix, const char* format, va_list args)
{
    /* Catch if no log is set or both strings are NULL. */
    if (!log || (!format && !prefix && depth > -10)) return;

    if (!should_print_file_entry(log, priority)) return;

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
        print_entry(log->file, log, priority, code, depth, prefix, format, args);

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

void oskar_log_writev_stderr(oskar_Log* log, char priority, char code, int depth,
        const char* prefix, const char* format, va_list args)
{
    /* Catch if both strings are NULL. */
    if (!format && !prefix && depth > -10) return;

    if (!should_print_term_entry(log, priority)) return;

    /* Print log entry to standard error. */
    print_entry(stderr, log, priority, code, depth, prefix, format, args);
    fflush(stderr);
}

void oskar_log_writev_stdout(oskar_Log* log, char priority, char code, int depth,
        const char* prefix, const char* format, va_list args)
{
    /* Catch if both strings are NULL. */
    if (!format && !prefix && depth > -10) return;

    if (!should_print_term_entry(log, priority)) return;

    /* Print log entry to standard output. */
    print_entry(stdout, log, priority, code, depth, prefix, format, args);
    fflush(stdout);
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
        return ' ';
    case 'D':
    case 'd':
        return 'D';
    default:
        return 'U';
    };
    return 'U';
}

static void print_prefix(FILE* stream, char code, const char* prefix)
{
    /* Print message code */
    fprintf(stream, "%c|", code);

    /* Print prefix string is present. */
    if (prefix && *prefix > 0)
        fprintf(stream, " %s", prefix);
}

static void print_list_symbol(FILE* stream, int depth)
{
    int i;
    char symbols[3] = {'+', '-', '*'};
    for (i = 0; i < depth; ++i) fprintf(stream, "  ");
    fprintf(stream, " %c", symbols[depth%3]);
}

static void print_message(FILE* stream, const char* format, va_list args)
{
    if (format && *format > 0) {
        fprintf(stream, " ");
        vfprintf(stream, format, args);
    }
}

/* FIXME this function is a mess! */
/*
 * Depth codes:
 *  - Positive code indicates a list indent.
 *  - Negative code have special meaning:
 *      * -1000 : a line
 *      * -100  : surround entry with blank lines (sections, errors, and warnings)
 *      * -100  : '==' prefix (sections, errors, and warnings)
 *      * -101  : no prefix, no space
 *      * -1    : no prefix, single space (this is the default 'message' type)
 *
 *  Break this function higher level functions don't have call a single
 *  catch all function.
 *
 *      - function to print the line prefix (code + prefix)
 *      - function to print the message
 *
 */
static void print_entry(FILE* stream, oskar_Log* log, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args)
{
    int i, width;
    const char* sym;

    /* Width for value entries */
    width = log ? log->value_width : OSKAR_LOG_DEFAULT_VALUE_WIDTH;

    /* Ensure code is a printable character. */
    if (code < 32) code += 48;

    /* Check if depth signifies a line. */
    if (depth == -1000)
    {
        char priority_code = oskar_log_get_entry_code(priority);
        fprintf(stream, "%c|", priority_code);
        for (i = 0; i < 68; ++i) fprintf(stream, "%c", code);
        fprintf(stream, "\n");
        return;
    }

    /* Print a blank line around section headings and errors. */
    if (depth == -100)
        fprintf(stream, " |\n");

    /* Print the message code. */
    fprintf(stream, "%c|", code);

    /* Print leading whitespace for depth. */
    if (depth > 0 && depth < 50)
        for (i = 0; i < depth; ++i) fprintf(stream, "  ");

    /* Print symbol for this depth.
     * All symbols should contain the same number of characters. */
    if (depth >= 0) {
        char symbols[3] = {'+', '-', '*'};
        fprintf(stream, " %c ", symbols[depth%3]);
    }
    else {
        switch (depth)
        {
        case -100:
            sym = "== ";
            break;
        case -101:
            sym = "";
            break;
        case -1:
            sym = " ";
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
        return OSKAR_LOG_WARN;
    case 'm':
    case 'M':
        return OSKAR_LOG_MESSAGE;
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
