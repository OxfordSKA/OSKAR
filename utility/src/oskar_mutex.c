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

#include <oskar_mutex.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


struct oskar_Mutex
{
#ifdef _OPENMP
    omp_lock_t lock;
#endif
};

oskar_Mutex* oskar_mutex_create(void)
{
    oskar_Mutex* mutex;

    /* Create the structure. */
    mutex = (oskar_Mutex*) calloc(1, sizeof(oskar_Mutex));

#ifdef _OPENMP
    omp_init_lock(&mutex->lock);
#endif
    return mutex;
}


void oskar_mutex_free(oskar_Mutex* mutex)
{
    if (!mutex) return;
#ifdef _OPENMP
    omp_destroy_lock(&mutex->lock);
#endif
    free(mutex);
}


void oskar_mutex_lock(oskar_Mutex* mutex)
{
    if (!mutex) return;
#ifdef _OPENMP
    omp_set_lock(&mutex->lock);
#endif
}


void oskar_mutex_unlock(oskar_Mutex* mutex)
{
    if (!mutex) return;
#ifdef _OPENMP
    omp_unset_lock(&mutex->lock);
#endif
}


#ifdef __cplusplus
}
#endif
