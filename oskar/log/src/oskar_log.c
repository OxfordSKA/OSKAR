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

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static void print_entry(FILE* stream, char priority, char code, int depth,
        const char* prefix, int width, const char* format, va_list args);
static int log_priority_level(char code);
static int should_print_term_entry(char priority);
static int should_print_file_entry(char priority);
static char get_entry_code(char priority);
static void write_log(oskar_Log* log, FILE* stream, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args);

static oskar_Log log_ = {0, OSKAR_LOG_NONE, OSKAR_LOG_WARNING, 40, 0, 0};

void oskar_log_error(const char* format, ...)
{
    va_list args;
    const char code = 'E', priority = 'E', *prefix = "== ERROR";
    const int depth = OSKAR_LOG_INFO_PREFIX;
    oskar_log_line(priority, ' ');
    va_start(args, format);
    write_log(&log_, stderr, priority, code, depth, prefix, format, args);
    va_end(args);
    if (log_.file)
    {
        va_start(args, format);
        write_log(&log_, log_.file, priority, code, depth, prefix, format, args);
        va_end(args);
    }
    oskar_log_line(priority, ' ');
}


#ifdef OSKAR_OS_WIN
/* http://stackoverflow.com/questions/15321493/how-should-i-pass-null-to-the-va-list-function-parameter */
static va_list do_create_empty_va_list(int i, ...)
{
    va_list vl;
    va_start(vl, i);
    return vl;
}
static va_list create_empty_va_list()
{
    return do_create_empty_va_list(0);
}
#endif

void oskar_log_line(char priority, char symbol)
{
    const char code = symbol, *format = 0, *prefix = 0;
    const int depth = OSKAR_LOG_LINE;
    FILE* stream = (priority == 'E') ? stderr : stdout;
#ifdef OSKAR_OS_WIN
    {
        va_list vl = create_empty_va_list();
        write_log(&log_, stream, priority, code, depth, prefix, format, vl);
        va_end(vl);
    }
    if (log_.file)
    {
        va_list vl = create_empty_va_list();
        write_log(&log_, log_.file, priority, code, depth, prefix, format, vl);
        va_end(vl);
    }
#else
    va_list vl;
    write_log(&log_, stream, priority, code, depth, prefix, format, vl);
    if (log_.file)
    {
        write_log(&log_, log_.file, priority, code, depth, prefix, format, vl);
    }
#endif
}


void oskar_log_message(char priority, int depth, const char* format, ...)
{
    va_list args;
    const char code = get_entry_code(priority), *prefix = 0;
    FILE* stream = (priority == 'E') ? stderr : stdout;
    va_start(args, format);
    write_log(&log_, stream, priority, code, depth, prefix, format, args);
    va_end(args);
    if (log_.file)
    {
        va_start(args, format);
        write_log(&log_, log_.file, priority, code, depth, prefix, format, args);
        va_end(args);
    }
}


void oskar_log_section(char priority, const char* format, ...)
{
    va_list args;
    const char code = '=', *prefix = "== ";
    const int depth = OSKAR_LOG_SECTION;
    FILE* stream = (priority == 'E') ? stderr : stdout;
    oskar_log_line(priority, ' ');
    va_start(args, format);
    write_log(&log_, stream, priority, code, depth, prefix, format, args);
    va_end(args);
    if (log_.file)
    {
        va_start(args, format);
        write_log(&log_, log_.file, priority, code, depth, prefix, format, args);
        va_end(args);
    }
    oskar_log_line(priority, ' ');
}


void oskar_log_value(char priority, int depth,
        const char* prefix, const char* format, ...)
{
    va_list args;
    const char code = get_entry_code(priority);
    FILE* stream = (priority == 'E') ? stderr : stdout;
    /* Only depth codes > -1 are valid for value log entries */
    if (depth < -1) return;
    va_start(args, format);
    write_log(&log_, stream, priority, code, depth, prefix, format, args);
    va_end(args);
    if (log_.file)
    {
        va_start(args, format);
        write_log(&log_, log_.file, priority, code, depth, prefix, format, args);
        va_end(args);
    }
}


void oskar_log_warning(const char* format, ...)
{
    va_list args;
    const char code = 'W', priority = 'W', *prefix = "== WARNING";
    const int depth = OSKAR_LOG_INFO_PREFIX;
    oskar_log_line(priority, ' ');
    va_start(args, format);
    write_log(&log_, stdout, priority, code, depth, prefix, format, args);
    va_end(args);
    if (log_.file)
    {
        va_start(args, format);
        write_log(&log_, log_.file, priority, code, depth, prefix, format, args);
        va_end(args);
    }
    oskar_log_line(priority, ' ');
}


oskar_Log* oskar_log_handle(void)
{
    return &log_;
}

void oskar_log_set_keep_file(int value)
{
    log_.keep_file = value;
}

void oskar_log_set_file_priority(int value)
{
    log_.file_priority = value;
}

void oskar_log_set_term_priority(int value)
{
    log_.term_priority = value;
}

void oskar_log_set_value_width(int value)
{
    log_.value_width = value;
}


static void write_log(oskar_Log* log, FILE* stream, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args)
{
    /* If both strings are NULL and not printing a line the entry is invalid */
    if (!format && !prefix && depth != OSKAR_LOG_LINE) return;

    const int width = log->value_width;
    const int is_file = (stream == stdout || stream == stderr) ? 0 : 1;

    /* Write the entry to the terminal */
    if (!is_file && should_print_term_entry(priority))
    {
        print_entry(stream, priority, code, depth, prefix, width, format, args);
        fflush(stream);
    }

    /* Write the entry to the log file */
    else if (is_file && log->file && should_print_file_entry(priority))
    {
        print_entry(log->file, priority, code, depth, prefix, width, format, args);
    }
}

static char get_entry_code(char priority)
{
    switch (priority) {
    case 'E':
    case 'e':
        return 'E';
    case 'W':
    case 'w':
        return 'W';
    case 'D':
    case 'd':
        return 'D';
    default:
        return ' ';
    };
    return ' ';
}

static void print_entry(FILE* stream, char priority, char code, int depth,
        const char* prefix, int width, const char* format, va_list args)
{
    int i;

    /* Ensure code is a printable character. */
    if (code < 32) code += 48;

    /* Check if depth signifies a line. */
    if (depth == OSKAR_LOG_LINE)
    {
        fprintf(stream, "%c|", get_entry_code(priority));
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
        fprintf(stream, " %c ", list_symbols[depth % 3]);
    }
    else {
        /* Negative depth codes with special meaning */
        switch (depth)
        {
        case OSKAR_LOG_INFO_PREFIX:
        case OSKAR_LOG_SECTION:
            break;
        default: /* Negative depth means no symbol. */
            fprintf(stream, " ");
            for (i = 0; i < abs(depth); ++i) fprintf(stream, "  ");
            break;
        }
    }

    /* Check if a prefix string is present. */
    if (prefix && *prefix > 0)
    {
        /* Print prefix. */
        fprintf(stream, "%s", prefix);

        /* Print trailing whitespace if format string is present. */
        if (format && *format > 0)
        {
            const int n = abs(2 * depth + 4 + (int)strlen(prefix));
            for (i = 0; i < width - n; ++i) fprintf(stream, " ");
            if (depth != OSKAR_LOG_SECTION) fprintf(stream, ": ");
        }
    }

    /* Print main message from format string and arguments. */
    if (format && *format > 0) vfprintf(stream, format, args);
    fprintf(stream, "\n");
}

/* Returns the enumerated priority level for the given message code.
 * This makes it possible to compare priority levels. */
static int log_priority_level(char code)
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

static int should_print_term_entry(char priority)
{
    return (log_priority_level(priority) <= log_.term_priority) ? 1 : 0;
}

static int should_print_file_entry(char priority)
{
    return (log_priority_level(priority) <= log_.file_priority) ? 1 : 0;
}

#ifdef __cplusplus
}
#endif
