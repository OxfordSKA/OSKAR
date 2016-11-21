/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_binary_write_metadata.h>
#include <oskar_version.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_write_metadata(oskar_Binary* handle, int* status)
{
    const char* str;
    size_t len;
    static char time_str[80];
    time_t unix_time;
    struct tm* timeinfo;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Write the system date and time. */
    unix_time = time(0);
    timeinfo = localtime(&unix_time);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)", timeinfo);
    len = 1 + strlen(time_str);
    oskar_binary_write(handle, OSKAR_CHAR,
            OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_DATE_TIME_STRING,
            0, len, time_str, status);

    /* Write the OSKAR version string. */
    len = 1 + strlen(OSKAR_VERSION_STR);
    oskar_binary_write(handle, OSKAR_CHAR,
            OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_OSKAR_VERSION_STRING,
            0, len, OSKAR_VERSION_STR, status);

    /* Write the current working directory. */
    str = getenv("PWD");
    if (!str) str = getenv("CD");
    if (str)
    {
        len = 1 + strlen(str);
        oskar_binary_write(handle, OSKAR_CHAR,
                OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_CWD,
                0, len, str, status);
    }

    /* Write the username. */
    str = getenv("USERNAME");
    if (!str)
        str = getenv("USER");
    if (str)
    {
        len = 1 + strlen(str);
        oskar_binary_write(handle, OSKAR_CHAR,
                OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_USERNAME,
                0, len, str, status);
    }
}

#ifdef __cplusplus
}
#endif
