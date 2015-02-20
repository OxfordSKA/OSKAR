/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#ifndef OSKAR_PRIVATE_TIMER_H_
#define OSKAR_PRIVATE_TIMER_H_

/**
 * @file private_timer.h
 */

#include <oskar_global.h>

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @struct oskar_Timer
 *
 * @brief Structure to hold data for a timer.
 *
 * @details
 * The structure holds data for a single timer.
 */
struct oskar_Timer
{
    int type;
    int paused;
    double elapsed;
    double start;
#ifdef OSKAR_HAVE_CUDA
    cudaEvent_t start_cuda, end_cuda;
#endif
#ifdef OSKAR_OS_WIN
    double freq;
#endif
#ifdef _OPENMP
    omp_lock_t mutex;
#endif
};

#ifndef OSKAR_TIMER_TYPEDEF_
#define OSKAR_TIMER_TYPEDEF_
typedef struct oskar_Timer oskar_Timer;
#endif /* OSKAR_TIMER_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_TIMER_H_ */
