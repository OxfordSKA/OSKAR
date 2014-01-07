/*
 * Copyright (c) 2013, The University of Oxford
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


#ifndef OSKAR_MATLAB_COMMON_H_
#define OSKAR_MATLAB_COMMON_H_

/**
 * @file oskar_matlab_common.h
 */

#include <mex.h>
#include <cstring>
#include <cstdarg>

/**
 * @brief Prints a formatted usage message for OSKAR mex functions.
 *
 * @param rtns      String containing formatted return values.
 * @param package   OSKAR MATLAB package name.
 * @param function  OSAKR MATLAB function name.
 * @param args      String containing formatted function arguments.
 * @param desc      Optional string containg a function description.
 */
void oskar_matlab_usage(const char* rtns, const char* package, const char* function,
        const char* args, const char* desc = NULL)
{
    if (!rtns || (int)strlen(rtns) == 0)
    {
        if (!desc || (int)strlen(desc) == 0)
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR",
                    "\n"
                    "ERROR:\n\tFunction called with invalid arguments or return values.\n"
                    "\n"
                    "Usage:\n"
                    "\toskar.%s.%s(%s)\n"
                    "\n",
                    package, function, args);
        }
        else
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR",
                    "\n"
                    "ERROR:\n\tFunction called with invalid arguments or return values.\n"
                    "\n"
                    "Usage:\n"
                    "\toskar.%s.%s(%s)\n"
                    "\n"
                    "Description:\n"
                    "\t%s\n"
                    "\n",
                    package, function, args, desc);
        }
    }
    else
    {
        if (!desc || (int)strlen(desc) == 0)
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR",
                    "\n"
                    "ERROR:\n\tFunction called with invalid arguments or return values.\n"
                    "\n"
                    "Usage:\n"
                    "\t%s = oskar.%s.%s(%s)\n"
                    "\n",
                    rtns, package, function, args);
        }
        else
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR",
                    "\n"
                    "ERROR:\n\tFunction called with invalid arguments or return values.\n"
                    "\n"
                    "Usage:\n"
                    "\t%s = oskar.%s.%s(%s)\n"
                    "\n"
                    "Description:\n"
                    "\t%s\n"
                    "\n",
                    rtns, package, function, args, desc);
        }
    }
}

/**
 * @brief Prints a formatted error message for use in OSKAR mex functions.
 *
 * @param format Format specifier string
 * @param arg    Variable arguments list.
 */
void oskar_matlab_error(const char* format, ...)
{
    char buffer[1024];
    int len = 0;
    va_list args;
    va_start(args, format);
    len = vsprintf(buffer, format, args);
    va_end(args);
    mexErrMsgIdAndTxt("OSKAR:ERROR", "\nERROR:\n\t%s.\n", buffer);
}


#endif /* OSKAR_MATLAB_COMMON_H_ */
