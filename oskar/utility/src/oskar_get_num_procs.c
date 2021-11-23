/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    if (cores == -1) cores = 1;
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
