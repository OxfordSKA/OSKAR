/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_JONES_H_
#define OSKAR_JONES_H_

/**
 * @file oskar_Jones.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"

/**
 * @brief Structure to hold Jones matrix and meta-data.
 *
 * @details
 * This structure holds the memory pointer and meta-data
 * for a type of Jones matrix.
 *
 * The fastest-varying dimension is the source dimension; the slowest varying
 * is the station dimension.
 */
struct OSKAR_EXPORT oskar_Jones
{
    int num_stations; /**< Slowest varying dimension. */
    int num_sources;  /**< Fastest varying dimension. */
    int cap_stations; /**< Slowest varying dimension. */
    int cap_sources;  /**< Fastest varying dimension. */
    oskar_Mem data;   /**< Pointer to the matrix data. */

#ifdef __cplusplus
    /* If C++, then provide constructors, a destructor and methods. */
    /**
     * @brief Constructs and allocates data for an oskar_Jones data structure.
     *
     * @details
     * Constructs a new oskar_Jones data structure, allocating memory for it
     * in the specified location.
     *
     * @param[in] type Enumerated data type of memory contents (magic number).
     * @param[in] location Enumerated memory location (magic number).
     * @param[in] num_stations Number of elements in the station dimension.
     * @param[in] num_sources Number of elements in the source dimension.
     */
    oskar_Jones(int type = OSKAR_DOUBLE, int location = OSKAR_LOCATION_CPU,
            int num_stations = 0, int num_sources = 0);

    /**
     * @brief Copies an oskar_Jones data structure.
     *
     * @details
     * Constructs a copy of the given oskar_Jones data structure, allocating
     * memory for it in the specified location.
     *
     * @param[in] other Pointer to the oskar_Jones structure to copy.
     * @param[in] location Specify 0 for host memory, 1 for device memory.
     */
    oskar_Jones(const oskar_Jones* other, int location);

    /**
     * @brief Destroys the structure and frees memory held by it.
     *
     * @details
     * Destroys the structure and frees memory held by it.
     */
    ~oskar_Jones();

    int type() const {return data.type;}
    int location() const {return data.location;}
#endif
};

typedef struct oskar_Jones oskar_Jones;

#endif /* OSKAR_JONES_H_ */
