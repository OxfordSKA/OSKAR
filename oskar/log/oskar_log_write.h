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

#ifndef OSKAR_LOG_WRITE_H_
#define OSKAR_LOG_WRITE_H_

/**
 * @file oskar_log_write.h
 */

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes a generic log entry.
 *
 * @details
 * This function writes a log entry to the to the terminal and, if the log
 * structure is defined, to a log file.
 *
 * @param[in,out] log    Pointer to a log structure.
 * @param[in]     priority Priority level of the log entry.
 * @param[in]     code   Code (group) of log message.
 * @param[in]     depth  Level of nesting of log message.
 * @param[in]     prefix Description of key (set blank or NULL if not required).
 * @param[in]     args   Variable argument list for printf().
 */
OSKAR_EXPORT
void oskar_log_write(oskar_Log* log, FILE* stream, char priority, char code,
        int depth, const char* prefix, const char* format, va_list args);


/**
 * @brief
 * Converts a log priority char into a message code.
 *
 * @param[in] priority An OSKAR log priority code.
 *
 * @return A code representing the priority.
 */
OSKAR_EXPORT
char oskar_log_get_entry_code(char priority);


#ifdef __cplusplus
}
#endif

#endif /* OSKAR_LOG_WRITE_H_ */
