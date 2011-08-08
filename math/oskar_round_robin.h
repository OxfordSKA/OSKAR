/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_MATH_ROUND_ROBIN_H_
#define OSKAR_MATH_ROUND_ROBIN_H_

/**
 * @file oskar_math_round_robin.h
 */

/**
 * @details
 * Distributes a number of \p items among a number of \p resources
 * (e.g. threads, or processes), and returns the \p number of items
 * and \p start item index (zero-based) for the current resource \p rank
 * (also zero-based).
 *
 * @param[in] items The number of items to distribute.
 * @param[in] resources The number of resources available.
 * @param[in] rank The index of this resource.
 * @param[out] number The number of items that this resource must process.
 * @param[out] start The start item index that this resource must process.
 */
static inline void oskar_round_robin(unsigned items, unsigned resources,
        unsigned rank, unsigned& number, unsigned& start)
{
    number = items / resources;
    unsigned remainder = items % resources;
    if (rank < remainder) number++;
    start = number * rank;
    if (rank >= remainder) start += remainder;
}

#endif // OSKAR_MATH_ROUND_ROBIN_H_
