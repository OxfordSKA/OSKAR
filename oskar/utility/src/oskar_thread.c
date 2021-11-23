/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_thread.h"
#include <stdlib.h>

#ifdef OSKAR_OS_WIN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 *  MUTEX
 * =========================================================================*/

struct oskar_Mutex
{
#ifdef OSKAR_OS_WIN
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif
};

static void oskar_mutex_init(oskar_Mutex* mutex)
{
#ifdef OSKAR_OS_WIN
    InitializeCriticalSectionAndSpinCount(&mutex->lock, 0x0800);
#else
    pthread_mutex_init(&mutex->lock, NULL);
#endif
}

static void oskar_mutex_uninit(oskar_Mutex* mutex)
{
#ifdef OSKAR_OS_WIN
    DeleteCriticalSection(&mutex->lock);
#else
    pthread_mutex_destroy(&mutex->lock);
#endif
}

oskar_Mutex* oskar_mutex_create(void)
{
    oskar_Mutex* mutex = 0;
    mutex = (oskar_Mutex*) calloc(1, sizeof(oskar_Mutex));
    oskar_mutex_init(mutex);
    return mutex;
}

void oskar_mutex_free(oskar_Mutex* mutex)
{
    if (!mutex) return;
    oskar_mutex_uninit(mutex);
    free(mutex);
}

void oskar_mutex_lock(oskar_Mutex* mutex)
{
#ifdef OSKAR_OS_WIN
    EnterCriticalSection(&mutex->lock);
#else
    pthread_mutex_lock(&mutex->lock);
#endif
}

void oskar_mutex_unlock(oskar_Mutex* mutex)
{
#ifdef OSKAR_OS_WIN
    LeaveCriticalSection(&mutex->lock);
#else
    pthread_mutex_unlock(&mutex->lock);
#endif
}


/* =========================================================================
 *  CONDITION VARIABLE
 * =========================================================================*/

struct oskar_ConditionVar
{
    oskar_Mutex lock;
#if defined(OSKAR_OS_WIN)
    CONDITION_VARIABLE var;
#else
    pthread_cond_t var;
#endif
};
typedef struct oskar_ConditionVar oskar_ConditionVar;

static void oskar_condition_init(oskar_ConditionVar* var)
{
    oskar_mutex_init(&var->lock);
#if defined(OSKAR_OS_WIN)
    InitializeConditionVariable(&var->var);
#else
    pthread_cond_init(&var->var, NULL);
#endif
}

static void oskar_condition_uninit(oskar_ConditionVar* var)
{
    oskar_mutex_uninit(&var->lock);
#if defined(OSKAR_OS_WIN)
    /* No equivalent to pthread_cond_destroy(). */
#else
    pthread_cond_destroy(&var->var);
#endif
}

static void oskar_condition_lock(oskar_ConditionVar* var)
{
    oskar_mutex_lock(&var->lock);
}

static void oskar_condition_unlock(oskar_ConditionVar* var)
{
    oskar_mutex_unlock(&var->lock);
}

static void oskar_condition_notify_all(oskar_ConditionVar* var)
{
#if defined(OSKAR_OS_WIN)
    WakeAllConditionVariable(&var->var);
#else
    pthread_cond_broadcast(&var->var);
#endif
}

static void oskar_condition_wait(oskar_ConditionVar* var)
{
#if defined(OSKAR_OS_WIN)
    SleepConditionVariableCS(&var->var, &(var->lock.lock), INFINITE);
#else
    pthread_cond_wait(&var->var, &(var->lock.lock));
#endif
}


/* =========================================================================
 *  THREAD
 * =========================================================================*/

struct oskar_Thread
{
    void *(*start_routine)(void*);
    void *arg;
#ifdef OSKAR_OS_WIN
    HANDLE thread;
    unsigned int thread_id;
#else
    pthread_t thread;
#endif
};

#ifdef OSKAR_OS_WIN
unsigned __stdcall thread_func_win(void* arg)
{
    oskar_Thread* thread = (oskar_Thread*)arg;
    thread->start_routine(thread->arg);
    _endthreadex(0);
    return 0;
}
#endif

oskar_Thread* oskar_thread_create(void *(*start_routine)(void*), void* arg,
        int detached)
{
#ifndef OSKAR_OS_WIN
    pthread_attr_t attr;
#endif
    oskar_Thread* thread = 0;
    thread = (oskar_Thread*) calloc(1, sizeof(oskar_Thread));
    thread->start_routine = start_routine;
    thread->arg = arg;

    /* Create the thread and run it. */
#ifdef OSKAR_OS_WIN
    thread->thread = (HANDLE) _beginthreadex(NULL, 0, thread_func_win, thread,
            (unsigned int) CREATE_SUSPENDED, &(thread->thread_id));
    if (thread->thread != 0)
        ResumeThread(thread->thread);
#else
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,
            detached ? PTHREAD_CREATE_DETACHED : PTHREAD_CREATE_JOINABLE);
    pthread_create(&thread->thread, &attr, start_routine, arg);
    pthread_attr_destroy(&attr);
#endif
    return thread;
}

void oskar_thread_free(oskar_Thread* thread)
{
    if (!thread) return;
#ifdef OSKAR_OS_WIN
    CloseHandle(thread->thread);
#endif
    free(thread);
}

void oskar_thread_join(oskar_Thread* thread)
{
    if (!thread) return;
#ifdef OSKAR_OS_WIN
    WaitForSingleObject(thread->thread, INFINITE);
#else
    pthread_join(thread->thread, NULL);
#endif
}


/* =========================================================================
 *  BARRIER
 * =========================================================================*/

struct oskar_Barrier
{
    oskar_ConditionVar var;
    unsigned int num_threads, count, iter;
};

oskar_Barrier* oskar_barrier_create(int num_threads)
{
    oskar_Barrier* barrier = 0;
    barrier = (oskar_Barrier*) calloc(1, sizeof(oskar_Barrier));
    oskar_condition_init(&barrier->var);
    barrier->num_threads = num_threads;
    barrier->count = num_threads;
    barrier->iter = 0;
    return barrier;
}

void oskar_barrier_free(oskar_Barrier* barrier)
{
    if (!barrier) return;
    oskar_condition_uninit(&barrier->var);
    free(barrier);
}

void oskar_barrier_set_num_threads(oskar_Barrier* barrier, int num_threads)
{
    barrier->num_threads = num_threads;
    barrier->count = num_threads;
    barrier->iter = 0;
}

int oskar_barrier_wait(oskar_Barrier* barrier)
{
    oskar_condition_lock(&barrier->var);
    {
        const unsigned int i = barrier->iter;
        if (--(barrier->count) == 0)
        {
            (barrier->iter)++;
            barrier->count = barrier->num_threads;
            oskar_condition_notify_all(&barrier->var);
            oskar_condition_unlock(&barrier->var);
            return 1;
        }
        /* Release lock and block this thread until notified/woken. */
        /* Allow for spurious wake-ups. */
        do {
            oskar_condition_wait(&barrier->var);
        } while (i == barrier->iter);
    }
    oskar_condition_unlock(&barrier->var);
    return 0;
}

#ifdef __cplusplus
}
#endif
