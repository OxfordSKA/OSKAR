/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#include <oskar_global.h>
#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Log;
#ifndef OSKAR_LOG_TYPEDEF_
#define OSKAR_LOG_TYPEDEF_
typedef struct oskar_Log oskar_Log;
#endif /* OSKAR_LOG_TYPEDEF_ */

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
    OSKAR_LOG_ADVICE  =  2,  /* == 'W' */
    OSKAR_LOG_MESSAGE =  3,  /* == 'M' */
    OSKAR_LOG_STATUS  =  4,  /* == 'S' */
    OSKAR_LOG_DEBUG   =  5   /* == 'D' */
};

/**
 * @brief Writes an advisory message to the log.
 *
 * @details
 * This function writes an advisory message to the log.
 * This is for a lower-priority warning message.
 *
 * @param[in]     format Format string for printf().
 */
OSKAR_EXPORT
void oskar_log_advice(oskar_Log* log, const char* format, ...);

/**
 * @brief
 * Closes the current log but does not destroy the handle.
 *
 * @details
 * This function closes the log but does not destroy its handle.
 * Subsequent log messages will be written to a new log file.
 */
OSKAR_EXPORT
void oskar_log_close(oskar_Log* log);

/**
 * @brief
 * Starts the log, creating a log file if necessary.
 *
 * @details
 * This function starts the log and creates a new log file if necessary.
 * The filename is generated based on the current date and time.
 */
OSKAR_EXPORT
oskar_Log* oskar_log_create(int file_priority, int term_priority);

/**
 * @brief Writes an error message to the log.
 *
 * @details
 * This function writes an error message to the log.
 *
 * @param[in]     format Format string for printf().
 */
OSKAR_EXPORT
void oskar_log_error(oskar_Log* log, const char* format, ...);

/**
 * @brief Returns current contents of log file.
 */
OSKAR_EXPORT
char* oskar_log_file_data(oskar_Log* log, size_t* size);

/**
 * @brief
 * Frees memory held in a log structure.
 *
 * @details
 * This function frees memory held in a log structure and closes the log file.
 */
OSKAR_EXPORT
void oskar_log_free(oskar_Log* log);

/**
 * @brief Writes a character symbol line to the log.
 *
 * @details
 * This function writes a character symbol line to the log.
 *
 * The priority code takes one of the following values:
 *   - 'E' : Error
 *   - 'W' : Warning
 *   - 'M' : Message
 *   - 'D' : Debug
 *
 * @param[in]     priority Priority of log entry.
 * @param[in]     symbol   Symbol to use for the line.
 */
OSKAR_EXPORT
void oskar_log_line(oskar_Log* log, char priority, char symbol);

/**
 * @brief Writes a message log entry.
 *
 * @details
 * This function writes a message log entry.
 *
 * Message log entries have a depth which will indent the message
 * by 2 spaces per depth value and, if positive, prefix the message with a
 * list bullet. If the depth value is negative the list bullet will be omitted.
 *
 * Log entries are created with a priority code which is used to determine
 * which log are printed in the log and also sets a letter code with which
 * the message is tagged.
 *
 * The priority code takes one of the following values:
 *   - 'E' : Error
 *   - 'W' : Warning
 *   - 'M' : Message
 *   - 'D' : Debug
 *
 * @param[in]     priority Priority of log entry.
 * @param[in]     depth    Level of nesting of log entry.
 * @param[in]     format   Format string for printf().
 */
OSKAR_EXPORT
void oskar_log_message(oskar_Log* log, char priority, int depth,
        const char* format, ...);

/**
 * @brief Writes a section-level message to the log.
 *
 * @details
 * This function writes a section-level message to the log.
 *
 * The priority code takes one of the following values:
 *   - 'E' : Error
 *   - 'W' : Warning
 *   - 'M' : Message
 *   - 'D' : Debug
 *
 * @param[in]     priority Priority code of the message.
 * @param[in]     format   Format string for printf().
 */
OSKAR_EXPORT
void oskar_log_section(oskar_Log* log, char priority, const char* format, ...);

/**
 * @brief Writes a key-value pair to the log.
 *
 * @details
 * This function writes a key-value pair to the log, where the value is
 * written starting at a fixed column offset from the start of the entry.
 *
 * The separation of the key and value is specified by the log width parameter
 * which is set by the oskar_log_set_value_width() method. The default
 * width for value messages places the value of the message is determined
 * by the OSKAR_LOG_DEFAULT_VALUE_WIDTH macro in the oskar_log.h header.
 *
 * Value log entries have a depth which, if positive, will indent the message
 * by 2 spaces per depth value and prefix the message with a list bullet.
 *
 * If the depth value is set to -1 the list bullet will be omitted and
 * no indent is applied.
 *
 * Log entries are created with a priority code which is used to determine
 * which log are printed in the log and also sets a letter code with which
 * the message is tagged.
 *
 * The priority code takes one of the following values:
 *   - 'E' : Error
 *   - 'W' : Warning
 *   - 'M' : Message
 *   - 'D' : Debug
 *
 * @param[in]     Depth    List depth of the log entry.
 * @param[in]     priority Priority of nesting of log entry.
 * @param[in]     prefix   String prefix (key).
 * @param[in]     format   Format string for printf().
 */
OSKAR_EXPORT
void oskar_log_value(oskar_Log* log, char priority, int depth,
        const char* prefix, const char* format, ...);

/**
 * @brief Writes a warning message to the log.
 *
 * @details
 * This function writes a warning message to the log.
 *
 * @param[in]     format Format string for printf().
 */
OSKAR_EXPORT
void oskar_log_warning(oskar_Log* log, const char* format, ...);

OSKAR_EXPORT
void oskar_log_set_keep_file(oskar_Log* log, int value);

/**
 * @brief Sets the logging verbosity level for log files.
 *
 * @param[in] value Entries with priority below this will not be written.
 */
OSKAR_EXPORT
void oskar_log_set_file_priority(oskar_Log* log, int value);

/**
 * @brief Sets the logging verbosity level when logging to the terminal.
 *
 * @param[in] value Entries with priority below this will not be printed.
 */
OSKAR_EXPORT
void oskar_log_set_term_priority(oskar_Log* log, int value);

OSKAR_EXPORT
void oskar_log_set_value_width(oskar_Log* log, int value);

OSKAR_EXPORT
double oskar_log_timestamp(void);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_LOG_H_ */
