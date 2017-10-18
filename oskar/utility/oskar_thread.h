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

#ifndef OSKAR_THREAD_H_
#define OSKAR_THREAD_H_

/**
 * @file oskar_thread.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Mutex;
struct oskar_Thread;
struct oskar_Barrier;
typedef struct oskar_Mutex oskar_Mutex;
typedef struct oskar_Thread oskar_Thread;
typedef struct oskar_Barrier oskar_Barrier;

/**
 * @brief Creates a mutex.
 *
 * @details
 * Creates a mutex.
 *
 * The mutex is created in an unlocked state.
 */
OSKAR_EXPORT
oskar_Mutex* oskar_mutex_create(void);

/**
 * @brief Destroys the mutex.
 *
 * @details
 * Destroys the mutex.
 *
 * @param[in,out] mutex Pointer to mutex.
 */
OSKAR_EXPORT
void oskar_mutex_free(oskar_Mutex* mutex);

/**
 * @brief Locks the mutex.
 *
 * @details
 * Locks the mutex.
 *
 * @param[in,out] mutex Pointer to mutex.
 */
OSKAR_EXPORT
void oskar_mutex_lock(oskar_Mutex* mutex);

/**
 * @brief Unlocks the mutex.
 *
 * @details
 * Unlocks the mutex.
 *
 * @param[in,out] mutex Pointer to mutex.
 */
OSKAR_EXPORT
void oskar_mutex_unlock(oskar_Mutex* mutex);

/**
 * @brief Creates and starts a thread.
 *
 * @details
 * Creates and starts a thread.
 */
OSKAR_EXPORT
oskar_Thread* oskar_thread_create(void *(*start_routine)(void*), void* arg,
        int detached);

/**
 * @brief Deallocates thread resources.
 *
 * @brief
 * Deallocates thread resources.
 */
OSKAR_EXPORT
void oskar_thread_free(oskar_Thread* thread);

/**
 * @brief Joins a thread with the caller.
 *
 * @brief
 * Blocks the caller until the specified thread has exited.
 */
OSKAR_EXPORT
void oskar_thread_join(oskar_Thread* thread);

/**
 * @brief Creates a barrier.
 *
 * @details
 * Creates a barrier.
 *
 * Rationale: Neither pthread_barrier functions or OpenMP are
 * supported natively on macOS.
 */
OSKAR_EXPORT
oskar_Barrier* oskar_barrier_create(int num_threads);

/**
 * @brief Destroys the barrier.
 *
 * @details
 * Destroys the barrier.
 *
 * @param[in,out] barrier Pointer to barrier.
 */
OSKAR_EXPORT
void oskar_barrier_free(oskar_Barrier* barrier);

/**
 * @brief Sets the number of threads the barrier must work for.
 *
 * @details
 * Sets the number of threads the barrier must work for.
 *
 * @param[in,out] barrier Pointer to barrier.
 */
OSKAR_EXPORT
void oskar_barrier_set_num_threads(oskar_Barrier* barrier, int num_threads);

/**
 * @brief Make all threads wait at the barrier.
 *
 * @details
 * Make all threads wait at the barrier.
 *
 * @param[in,out] barrier Pointer to barrier.
 */
OSKAR_EXPORT
int oskar_barrier_wait(oskar_Barrier* barrier);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_THREAD_H_ */
