/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_get_memory_usage.h"

#include <stdio.h>
#include <stddef.h>

#if defined(OSKAR_OS_LINUX)
#   include <sys/types.h>
#   include <sys/sysinfo.h>
#   include <stdlib.h>
#   include <stdio.h>
#   include <string.h>
#elif defined(OSKAR_OS_MAC)
#   include <sys/param.h>
#   include <sys/mount.h>
#   include <sys/types.h>
#   include <sys/sysctl.h>
#   include <mach/mach.h>
#   include <mach/vm_statistics.h>
#   include <mach/mach_types.h>
#   include <mach/mach_init.h>
#   include <mach/mach_host.h>
#   include <sys/sysctl.h>
#elif defined(OSKAR_OS_WIN)
#   include <windows.h>
#   include <psapi.h>
#endif

/* http://goo.gl/iKnmd */

#ifdef __cplusplus
extern "C" {
#endif

size_t oskar_get_total_physical_memory(void)
{
#ifdef OSKAR_OS_LINUX
    size_t totalPhysMem = 0;
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;
    return totalPhysMem;
#elif defined(OSKAR_OS_MAC)
    int mib[2];
    int64_t physical_memory;
    size_t length;
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    return (size_t)physical_memory;
#elif defined(OSKAR_OS_WIN)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    DWORDLONG totalPhysMem = memInfo.ullTotalPhys;
    return (size_t)totalPhysMem;
#endif
}

size_t oskar_get_free_physical_memory(void)
{
#ifdef OSKAR_OS_LINUX
    size_t freePhysMem = 0;
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    freePhysMem = memInfo.freeram;
    freePhysMem *= memInfo.mem_unit;
    return freePhysMem;
#elif defined(OSKAR_OS_MAC)
    vm_size_t page_size;
    mach_port_t mach_port;
    mach_msg_type_number_t count;
    vm_statistics64_data_t vm_stats;
    mach_port = mach_host_self();
    count = sizeof(vm_stats) / sizeof(natural_t);
    if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
            KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
                    (host_info64_t)&vm_stats, &count))
    {
        /* Note on OS X since 10.9, compressed memory makes this value somewhat
         * flexible. */
        return (size_t)((int64_t)vm_stats.free_count*(int64_t)page_size);
    }
    return 0;
#elif defined(OSKAR_OS_WIN)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    DWORDLONG freePhysMem = memInfo.ullAvailPhys;
    return (size_t)freePhysMem;
#endif
}

#ifdef OSKAR_OS_LINUX
static size_t parse_line(char* line)
{
    size_t i = strlen(line);
    while (*line < '0' || *line > '9') line++;
    line[i-3] = '\0';
#if __cplusplus >= 201103L || __STDC_VERSION__ >= 199901L
    i = (size_t) strtoull(line, NULL, 10);
#else
    i = (size_t) strtoul(line, NULL, 10);
#endif
    return i;
}
#endif

size_t oskar_get_memory_usage(void)
{
#ifdef OSKAR_OS_LINUX
    FILE* file = fopen("/proc/self/status", "r");
    size_t result = 0;
    char line[128];
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            result = parse_line(line);
            break;
        }
    }
    fclose(file);
    /* Value in /proc/self/status is in kB. */
    return result * 1024;
#elif defined(OSKAR_OS_MAC)
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO,
                                  (task_info_t)&t_info, &t_info_count))
        return 0L;
    return t_info.resident_size;
#elif defined(OSKAR_OS_WIN)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc,
            sizeof(pmc));
    SIZE_T physMemUsedByMe = pmc.WorkingSetSize;
    return (size_t)physMemUsedByMe;
#else
    return 0L;
#endif
}

void oskar_log_mem(oskar_Log* log)
{
    const size_t gigabyte = 1024 * 1024 * 1024;
    const size_t mem_total = oskar_get_total_physical_memory();
    const size_t mem_resident = oskar_get_memory_usage();
    const size_t mem_free = oskar_get_free_physical_memory();
    const size_t mem_used = mem_total - mem_free;
    oskar_log_message(log, 'M', 0,
            "System memory is %.1f%% (%.1f GB/%.1f GB) used.",
            100. * (double) mem_used / mem_total,
            (double) mem_used / gigabyte,
            (double) mem_total / gigabyte);
    oskar_log_message(log, 'M', 0,
            "System memory used by current process: %.1f MB.",
            (double) mem_resident / (1024. * 1024.));
}

#ifdef __cplusplus
}
#endif
