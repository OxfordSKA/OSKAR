/*
 * Copyright (c) 2016, The University of Oxford
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

#ifndef OSKAR_MUTEX_H_
#define OSKAR_MUTEX_H_

/**
 * @file oskar_mutex.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Mutex;
#ifndef OSKAR_MUTEX_TYPEDEF_
#define OSKAR_MUTEX_TYPEDEF_
typedef struct oskar_Mutex oskar_Mutex;
#endif /* OSKAR_MUTEX_TYPEDEF_ */

/**
 * @brief Creates a mutex.
 *
 * @details
 * Creates a mutex.
 * The mutex will use the OpenMP mutex if it is available.
 *
 * The mutex is created in an unlocked state.
 *
 * @param[in,out] timer Pointer to timer.
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

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MUTEX_H_ */
