/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "log/oskar_log.h"
#include "utility/oskar_lock_file.h"
#include "oskar_version.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef OSKAR_OS_WIN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define WRITE_TIMESTAMP 0

struct oskar_Log
{
    int init;                         /* Initialisation flag. */
    int keep_file;                    /* If true, log file will be kept. */
    int file_priority, term_priority; /* Message priority filters. */
    int value_width;                  /* Width of value message */
    int write_header;                 /* If true, write standard headers. */
    FILE* file;                       /* Log file handle. */
    double timestamp_start;           /* Timestamp of log creation. */
    char name[120];                   /* Log file pathname. */
};

#ifndef OSKAR_LOG_TYPEDEF_
#define OSKAR_LOG_TYPEDEF_
typedef struct oskar_Log oskar_Log;
#endif /* OSKAR_LOG_TYPEDEF_ */


static void init_log(oskar_Log* log);
static void print_entry(oskar_Log* log, FILE* stream, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args);
static int log_priority_level(char code);
static char get_entry_code(char priority);
static void write_log(oskar_Log* log, int to_file, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args);

/* Root logger. */
static oskar_Log log_ = {
        0, /* Init flag. */
        0, /* Keep file. */
        OSKAR_LOG_NONE, /* File message priority. */
        OSKAR_LOG_WARNING, /* Terminal message priority. */
        OSKAR_LOG_DEFAULT_VALUE_WIDTH, /* Value width. */
        0, /* Write standard headers. */
        0, /* File pointer. */
        0.0, /* Timestamp start. */
        {0} /* File name. */
};

#if __STDC_VERSION__ >= 199901L || (defined(__cplusplus) && __cplusplus >= 201103L)
#define SNPRINTF(BUF, SIZE, FMT, ...) snprintf(BUF, SIZE, FMT, __VA_ARGS__);
#else
#define SNPRINTF(BUF, SIZE, FMT, ...) sprintf(BUF, FMT, __VA_ARGS__);
#endif


void oskar_log_advice(oskar_Log* log, const char* format, ...)
{
    va_list args;
    const char code = 'W', priority = 'A', *prefix = "== ADVICE";
    const int depth = OSKAR_LOG_INFO_PREFIX;
    oskar_log_line(log, priority, ' ');
    va_start(args, format);
    write_log(log, 0, priority, code, depth, prefix, format, args);
    va_end(args);
    va_start(args, format);
    write_log(log, 1, priority, code, depth, prefix, format, args);
    va_end(args);
    oskar_log_line(log, priority, ' ');
}


void oskar_log_close(oskar_Log* log)
{
    if (log->write_header && log->init)
    {
        char time_str[80];
        const time_t unix_time = time(NULL);
        struct tm* timeinfo = localtime(&unix_time);
        strftime(time_str, sizeof(time_str),
                "%Y-%m-%d, %H:%M:%S (%Z)", timeinfo);
        oskar_log_section(log, 'M', "OSKAR-%s ending at %s.",
                OSKAR_VERSION_STR, time_str);
    }
    if (log->file) fclose(log->file);
    log->file = 0;
    if (!log->keep_file && strlen(log->name) > 0)
    {
        FILE* f = fopen(log->name, "r");
        if (f)
        {
            fclose(f);
            remove(log->name);
        }
    }
    log->name[0] = 0;
    log->init = 0;
}


oskar_Log* oskar_log_create(int file_priority, int term_priority)
{
    oskar_Log* log = 0;
    log = (oskar_Log*) calloc(1, sizeof(oskar_Log));
    log->file_priority = file_priority;
    log->term_priority = term_priority;
    log->value_width = OSKAR_LOG_DEFAULT_VALUE_WIDTH;
    log->write_header = 1;
    return log;
}


double oskar_log_timestamp()
{
#if defined(OSKAR_OS_WIN)
    /* Windows-specific version. */
    LARGE_INTEGER cntr, ifreq;
    QueryPerformanceCounter(&cntr);
    QueryPerformanceFrequency(&ifreq);
    const double freq = (double)(ifreq.QuadPart);
    return (double)(cntr.QuadPart) / freq;
#elif _POSIX_MONOTONIC_CLOCK > 0
    /* Use monotonic clock if available. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#else
    /* Use gettimeofday() as fallback. */
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec / 1e6;
#endif
}


void oskar_log_error(oskar_Log* log, const char* format, ...)
{
    va_list args;
    const char code = 'E', priority = 'E', *prefix = "== ERROR";
    const int depth = OSKAR_LOG_INFO_PREFIX;
    oskar_log_line(log, priority, ' ');
    va_start(args, format);
    write_log(log, 0, priority, code, depth, prefix, format, args);
    va_end(args);
    va_start(args, format);
    write_log(log, 1, priority, code, depth, prefix, format, args);
    va_end(args);
    oskar_log_line(log, priority, ' ');
}


char* oskar_log_file_data(oskar_Log* log, size_t* size)
{
    char* data = 0;
    if (!size) return 0;
    if (!log) log = &log_;

    /* If log exists, then read the whole file. */
    if (log->file)
    {
        FILE* temp_handle = 0;

        /* Determine the current size of the file. */
        fflush(log->file);
        temp_handle = fopen(log->name, "rb");
        if (temp_handle)
        {
            fseek(temp_handle, 0, SEEK_END);
            *size = ftell(temp_handle);

            /* Read the file into memory. */
            if (*size != 0)
            {
                size_t bytes_read = 0;
                data = (char*) calloc(10 + *size * sizeof(char), 1);
                if (data != 0)
                {
                    rewind(temp_handle);
                    bytes_read = fread(data, 1, *size, temp_handle);
                    if (bytes_read != *size)
                    {
                        free(data);
                        data = 0;
                    }
                }
            }
            fclose(temp_handle);
        }
    }

    return data;
}


void oskar_log_free(oskar_Log* log)
{
    if (!log)
    {
        oskar_log_close(&log_);
    }
    else
    {
        oskar_log_close(log);
        free(log);
    }
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

void oskar_log_line(oskar_Log* log, char priority, char symbol)
{
    const char code = symbol, *format = 0, *prefix = 0;
    const int depth = OSKAR_LOG_LINE;
#ifdef OSKAR_OS_WIN
    {
        va_list vl = create_empty_va_list();
        write_log(log, 0, priority, code, depth, prefix, format, vl);
        va_end(vl);
    }
    va_list vl = create_empty_va_list();
    write_log(log, 1, priority, code, depth, prefix, format, vl);
    va_end(vl);
#else
    va_list vl;
    write_log(log, 0, priority, code, depth, prefix, format, vl);
    write_log(log, 1, priority, code, depth, prefix, format, vl);
#endif
}


void oskar_log_message(oskar_Log* log, char priority, int depth,
        const char* format, ...)
{
    va_list args;
    const char code = get_entry_code(priority), *prefix = 0;
    va_start(args, format);
    write_log(log, 0, priority, code, depth, prefix, format, args);
    va_end(args);
    va_start(args, format);
    write_log(log, 1, priority, code, depth, prefix, format, args);
    va_end(args);
}


void oskar_log_section(oskar_Log* log, char priority, const char* format, ...)
{
    va_list args;
    const char code = '=', *prefix = "== ";
    const int depth = OSKAR_LOG_SECTION;
    oskar_log_line(log, priority, ' ');
    va_start(args, format);
    write_log(log, 0, priority, code, depth, prefix, format, args);
    va_end(args);
    va_start(args, format);
    write_log(log, 1, priority, code, depth, prefix, format, args);
    va_end(args);
    oskar_log_line(log, priority, ' ');
}


void oskar_log_value(oskar_Log* log, char priority, int depth,
        const char* prefix, const char* format, ...)
{
    va_list args;
    const char code = get_entry_code(priority);
    /* Only depth codes > -1 are valid for value log entries */
    if (depth < -1) return;
    va_start(args, format);
    write_log(log, 0, priority, code, depth, prefix, format, args);
    va_end(args);
    va_start(args, format);
    write_log(log, 1, priority, code, depth, prefix, format, args);
    va_end(args);
}


void oskar_log_warning(oskar_Log* log, const char* format, ...)
{
    va_list args;
    const char code = 'W', priority = 'W', *prefix = "== WARNING";
    const int depth = OSKAR_LOG_INFO_PREFIX;
    oskar_log_line(log, priority, ' ');
    va_start(args, format);
    write_log(log, 0, priority, code, depth, prefix, format, args);
    va_end(args);
    va_start(args, format);
    write_log(log, 1, priority, code, depth, prefix, format, args);
    va_end(args);
    oskar_log_line(log, priority, ' ');
}


void oskar_log_set_keep_file(oskar_Log* log, int value)
{
    if (!log) log = &log_;
    log->keep_file = value;
}

void oskar_log_set_file_priority(oskar_Log* log, int value)
{
    if (!log) log = &log_;
    log->file_priority = value;
}

void oskar_log_set_term_priority(oskar_Log* log, int value)
{
    if (!log) log = &log_;
    log->term_priority = value;
}

void oskar_log_set_value_width(oskar_Log* log, int value)
{
    if (!log) log = &log_;
    log->value_width = value;
}


void init_log(oskar_Log* log)
{
    struct tm* timeinfo = 0;
    size_t buf_len = 0;
    char *current_dir = 0, fname1[120], time_str[120];
    int i = 0, n = 0;
    log->init = 1;

    /* Construct log file name root. */
    const time_t unix_time = time(NULL);
    timeinfo = localtime(&unix_time);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)", timeinfo);
    timeinfo->tm_mon += 1;
    timeinfo->tm_year += 1900;
    n = SNPRINTF(fname1, sizeof(fname1),
            "oskar_%.4d-%.2d-%.2d_%.2d%.2d%.2d",
            timeinfo->tm_year, timeinfo->tm_mon, timeinfo->tm_mday,
            timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
    if (n < 0 || n >= (int)sizeof(fname1))
    {
        return;
    }

    /* Construct a unique log file name. */
    do
    {
        ++i;
        if (i == 1)
        {
            n = SNPRINTF(log->name, sizeof(log->name), "%s.log", fname1);
        }
        else
        {
            n = SNPRINTF(log->name, sizeof(log->name), "%s_%d.log", fname1, i);
        }
        if (n < 0 || n >= (int)sizeof(log->name))
        {
            return;
        }
        if (i > 1000)
        {
            log->name[0] = 0;
            break;
        }
    }
    while (!oskar_lock_file(log->name));

    /* Open or remove the log file if required. */
    if (strlen(log->name) > 0)
    {
        if (log->file_priority > OSKAR_LOG_NONE)
        {
            log->file = fopen(log->name, "a+");
        }
        else
        {
            FILE* f = fopen(log->name, "rb");
            if (f)
            {
                fclose(f);
                remove(log->name);
            }
        }
    }

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

    /* Write standard header. */
    log->timestamp_start = oskar_log_timestamp();
    if (log->write_header)
    {
        oskar_log_section(log, 'M', "OSKAR-%s starting at %s.",
                OSKAR_VERSION_STR, time_str);
        oskar_log_message(log, 'M', 0, "Current dir is %s", current_dir);
    }
    free(current_dir);
    if (log->file)
    {
        oskar_log_message(log, 'M', 0, "Logging to file %s", log->name);
    }
    else if (log->file_priority > OSKAR_LOG_NONE)
    {
        oskar_log_warning(log, "Log file could not be created.");
    }
}


static void write_log(oskar_Log* log, int to_file, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args)
{
    if (!log) log = &log_;

    /* If both strings are NULL and not printing a line the entry is invalid */
    if (!format && !prefix && depth != OSKAR_LOG_LINE) return;

    /* Check if the log needs to be initialised. */
    if (!log->init) init_log(log);
    const int priority_level = log_priority_level(priority);

    /* Write the entry to the terminal or log file. */
    if (!to_file && (priority_level <= log->term_priority))
    {
        FILE* stream = (priority == 'E' ? stderr : stdout);
        print_entry(log, stream,
                priority, code, depth, prefix, format, args);
        fflush(stream);
    }
    else if (to_file && (priority_level <= log->file_priority))
    {
        print_entry(log, log->file,
                priority, code, depth, prefix, format, args);
        fflush(log->file);
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

static void print_entry(oskar_Log* log, FILE* stream, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args)
{
    int i = 0;
    if (!stream) return;
    const int width = log->value_width;

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

#if WRITE_TIMESTAMP
    /* Print the timestamp. */
    fprintf(stream, "%6.1f ", oskar_log_timestamp() - log->timestamp_start);
#endif

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
    case 'a':
    case 'A':
        return OSKAR_LOG_ADVICE;
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

#ifdef __cplusplus
}
#endif
