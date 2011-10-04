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

#ifndef OSKAR_JONES_H_
#define OSKAR_JONES_H_

/**
 * @file oskar_Jones.h
 */

/**
 * @brief Structure to hold Jones matrix meta-data.
 *
 * @details
 * This is a valid C-structure that holds the memory pointer and meta-data
 * for a type of Jones matrix.
 *
 * If using C++, then the meta-data is made private, and read-only accessor
 * functions are also provided.
 */
#ifdef __cplusplus
extern "C"
#endif
struct oskar_Jones
{
    // If C++, then make the meta-data private.
#ifdef __cplusplus
private:
#endif
    int private_type; // Magic number.
    int private_n_sources;  ///< Fastest varying dimension.
    int private_n_stations; ///< Slowest varying dimension.
    int private_location; // 0 for host, 1 for device.

    // If C++, then make the remaining members public.
#ifdef __cplusplus
public:
#endif
    void* data; ///< Pointer to the matrix data.

    // If C++, then provide a constructor and destructor for the structure.
#ifdef __cplusplus
    /// Constructs and allocates data for an oskar_Jones data structure.
    oskar_Jones(int type, int n_sources, int n_stations, int location);

    /// Destroys the structure and frees memory held by it.
    ~oskar_Jones();
#endif

    // If C++, then provide methods on the structure.
#ifdef __cplusplus
    /// Copies the memory contents and meta-data of this structure to another.
    int copy_to(oskar_Jones* other);
#endif

    // If C++, then provide read-only accessor functions for the meta-data.
#ifdef __cplusplus
    int type() const {return private_type;}
    int n_sources() const {return private_n_sources;}
    int n_stations() const {return private_n_stations;}
    int location() const {return private_location;}
#endif
};

// Define an enumerator for the type.
enum {
    OSKAR_JONES_FLOAT_SCALAR  = 0x01CF, // scalar, complex float
    OSKAR_JONES_DOUBLE_SCALAR = 0x01CD, // scalar, complex double
    OSKAR_JONES_FLOAT_MATRIX  = 0x04CF, // matrix, complex float
    OSKAR_JONES_DOUBLE_MATRIX = 0x04CD, // matrix, complex double
};

typedef struct oskar_Jones oskar_Jones;

#endif // OSKAR_JONES_H_
