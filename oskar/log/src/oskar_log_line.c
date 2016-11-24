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

#include "log/private_log.h"
#include "log/oskar_log.h"
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    char code = symbol;
    int depth = OSKAR_LOG_LINE;
    const char* prefix = 0;
    const char* format = 0;
    FILE* stream = (priority == 'E') ? stderr : stdout;

#ifdef OSKAR_OS_WIN
    /* Print to stdout or stderr */
    {
        va_list vl = create_empty_va_list();
        oskar_log_write(log, stream, priority, code, depth, prefix, format, vl);
        va_end(vl);
    }
    /* Print to the log file */
    if (log && log->file)
    {
        va_list vl = create_empty_va_list();
        oskar_log_write(log, log->file, priority, code, depth, prefix, format, vl);
        va_end(vl);
    }
#else
    va_list vl;
    /* Print to stdout or stderr */
    oskar_log_write(log, stream, priority, code, depth, prefix, format, vl);

    /* Print to the log file */
    if (log && log->file)
    {
        oskar_log_write(log, log->file, priority, code, depth, prefix, format, vl);
    }
#endif
}

#ifdef __cplusplus
}
#endif
