/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_device.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_thread.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifndef OSKAR_OS_WIN
#include <sys/time.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef OSKAR_HAVE_OPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Timer
{
    oskar_Mutex* mutex;
#ifdef OSKAR_HAVE_CUDA
    cudaEvent_t start_cuda, end_cuda;
#endif
    double start, elapsed;
#ifdef OSKAR_OS_WIN
    double freq;
#endif
    int type, paused;
};

static double oskar_get_wtime(oskar_Timer* timer)
{
#if defined(OSKAR_OS_WIN)
    /* Windows-specific version. */
    LARGE_INTEGER cntr;
    QueryPerformanceCounter(&cntr);
    return (double)(cntr.QuadPart) / timer->freq;
#elif _POSIX_MONOTONIC_CLOCK > 0
    /* Use monotonic clock if available. */
    struct timespec ts;
    (void)timer;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#else
    /* Use gettimeofday() as fallback. */
    struct timeval tv;
    (void)timer;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec / 1e6;
#endif
}

oskar_Timer* oskar_timer_create(int type)
{
    oskar_Timer* timer = 0;
#ifdef OSKAR_OS_WIN
    LARGE_INTEGER freq;
#endif
    timer = (oskar_Timer*) calloc(1, sizeof(oskar_Timer));
    timer->mutex = oskar_mutex_create();
#ifdef OSKAR_OS_WIN
    QueryPerformanceFrequency(&freq);
    timer->freq = (double)(freq.QuadPart);
#endif
    timer->type = type;
    timer->paused = 1;
#ifdef OSKAR_HAVE_CUDA
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        cudaEventCreate(&timer->start_cuda);
        cudaEventCreate(&timer->end_cuda);
    }
#endif
    return timer;
}

void oskar_timer_free(oskar_Timer* timer)
{
    if (!timer) return;
#ifdef OSKAR_HAVE_CUDA
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        cudaEventDestroy(timer->start_cuda);
        cudaEventDestroy(timer->end_cuda);
    }
#endif
    oskar_mutex_free(timer->mutex);
    free(timer);
}

double oskar_timer_elapsed(oskar_Timer* timer)
{
    /* If timer is paused, return immediately with current elapsed time. */
    if (timer->paused) return timer->elapsed;

#ifdef OSKAR_HAVE_CUDA
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        float millisec = 0.0f;
        oskar_mutex_lock(timer->mutex);

        /* Get elapsed time since start. */
        cudaEventRecord(timer->end_cuda, 0);
        cudaEventSynchronize(timer->end_cuda);
        cudaEventElapsedTime(&millisec, timer->start_cuda, timer->end_cuda);

        /* Increment elapsed time and restart. */
        timer->elapsed += millisec / 1000.0;
        cudaEventRecord(timer->start_cuda, 0);
        oskar_mutex_unlock(timer->mutex);
        return timer->elapsed;
    }
#endif
#ifdef OSKAR_HAVE_OPENCL
    if (timer->type == OSKAR_TIMER_CL)
    {
        /* Get elapsed time since start. */
        /* Note that clGetEventProfilingInfo() seems to be broken
         * in various ways, at least on macOS. So just enqueue a marker,
         * wait on its event and use the native timer instead. */
        cl_event event = 0;
        clEnqueueMarkerWithWaitList(oskar_device_queue_cl(), 0, NULL, &event);
        clWaitForEvents(1, &event);

        /* Increment elapsed time and restart. */
        oskar_mutex_lock(timer->mutex);
        const double now = oskar_get_wtime(timer);
        timer->elapsed += (now - timer->start);
        timer->start = now;
        oskar_mutex_unlock(timer->mutex);
        return timer->elapsed;
    }
#endif
    if (timer->type == OSKAR_TIMER_NATIVE)
    {
        oskar_mutex_lock(timer->mutex);
        const double now = oskar_get_wtime(timer);

        /* Increment elapsed time and restart. */
        timer->elapsed += (now - timer->start);
        timer->start = now;
        oskar_mutex_unlock(timer->mutex);
        return timer->elapsed;
    }
    return timer->elapsed;
}

void oskar_timer_pause(oskar_Timer* timer)
{
    if (timer->paused) return;
    (void)oskar_timer_elapsed(timer);
    timer->paused = 1;
}

void oskar_timer_reset(oskar_Timer* timer)
{
    oskar_mutex_lock(timer->mutex);
    timer->paused = 1;
    timer->start = 0.0;
    timer->elapsed = 0.0;
    oskar_mutex_unlock(timer->mutex);
}

void oskar_timer_resume(oskar_Timer* timer)
{
    if (!timer->paused) return;
    oskar_timer_restart(timer);
}

void oskar_timer_restart(oskar_Timer* timer)
{
    timer->paused = 0;
#ifdef OSKAR_HAVE_CUDA
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        cudaEventRecord(timer->start_cuda, 0);
        return;
    }
#endif
    timer->start = oskar_get_wtime(timer);
}

void oskar_timer_start(oskar_Timer* timer)
{
    timer->elapsed = 0.0;
    oskar_timer_restart(timer);
}

#ifdef __cplusplus
}
#endif
