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

#ifndef OSKAR_LOG_VALUE_H_
#define OSKAR_LOG_VALUE_H_

/**
 * @file oskar_log_value.h
 */

#include <oskar_global.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes a key-value pair to the log.
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
 * @param[in,out] log      Pointer to a log structure.
 * @param[in]     Depth    List depth of the log entry.
 * @param[in]     priority Priority of nesting of log entry.
 * @param[in]     prefix   String prefix (key).
 * @param[in]     format   Format string for printf().
 */
OSKAR_EXPORT
void oskar_log_value(oskar_Log* log, char priority, int depth,
        const char* prefix, const char* format, ...);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_LOG_VALUE_H_ */
