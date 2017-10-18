/*
 * Copyright (c) 2017, The University of Oxford
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

#include "utility/oskar_get_num_procs.h"

#if defined(OSKAR_OS_WIN)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <process.h>
#else
    #include <unistd.h>
    #ifndef _SC_NPROCESSORS_ONLN
        #ifdef _SC_NPROC_ONLN
            #define _SC_NPROCESSORS_ONLN _SC_NPROC_ONLN
        #elif defined _SC_CRAY_NCPU
            #define _SC_NPROCESSORS_ONLN _SC_CRAY_NCPU
        #endif
    #endif
    #if defined(OSKAR_OS_MAC)
        #include <sys/sysctl.h>
        #include <stdio.h>
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

int oskar_get_num_procs(void)
{
    int cores = 1;
#if defined(OSKAR_OS_WIN)
    SYSTEM_INFO sysinfo;
    GetNativeSystemInfo(&sysinfo);
    cores = sysinfo.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (cores == -1)
        cores = 1;
#elif defined(OSKAR_OS_MAC)
    size_t len = sizeof(cores);
    int mib[2];
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    if (sysctl(mib, 2, &cores, &len, NULL, 0) != 0)
        perror("sysctl error");
#endif
    return cores;
}

#ifdef __cplusplus
}
#endif
