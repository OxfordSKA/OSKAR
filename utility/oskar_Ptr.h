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

#ifndef OSKAR_PTR_H_
#define OSKAR_PTR_H_

/**
 * @file oskar_Ptr.h
 */

/**
 * @brief Structure to wrap a pointer.
 *
 * @details
 * This is a valid C-structure that holds a pointer to memory either on the CPU
 * or GPU, and defines the type of the data that it points to.
 *
 * If using C++, then the meta-data is made private, and read-only accessor
 * functions are also provided.
 */
#ifdef __cplusplus
extern "C"
#endif
struct oskar_Ptr
{
    // If C++, then make the meta-data private.
#ifdef __cplusplus
private:
#endif
    int private_type; // Magic number.
    int private_location; // 0 for host, 1 for device.

    // If C++, then make the remaining members public.
#ifdef __cplusplus
public:
#endif
    void* data; ///< Data pointer.

    // If C++, then provide a constructor, a destructor.
#ifdef __cplusplus
    /**
     * @brief Constructs and allocates data for an oskar_Ptr data structure.
     *
     * @details
     * Constructs a new oskar_Ptr data structure, allocating memory for it
     * in the specified location.
     *
     * @param[in] type Enumerated data type of memory contents (magic number).
     * @param[in] location Specify 0 for host memory, 1 for device memory.
     */
    oskar_Ptr(int type, int location)
    : private_type(type), private_location(location) {}

    /**
     * @brief Destroys the structure.
     *
     * @details
     * Destroys the structure.
     */
    ~oskar_Ptr() {}
#endif

    // If C++, then provide read-only accessor functions for the meta-data.
#ifdef __cplusplus
    int type() const {return private_type;}
    int location() const {return private_location;}
#endif
};

// Define an enumerator for the type.
enum {
    OSKAR_SINGLE                 = 0x010F, // (float)    scalar, float
    OSKAR_DOUBLE                 = 0x010D, // (double)   scalar, double
    OSKAR_SINGLE_COMPLEX         = 0x01CF, // (float2)   scalar, complex float
    OSKAR_DOUBLE_COMPLEX         = 0x01CD, // (double2)  scalar, complex double
    OSKAR_SINGLE_MATRIX          = 0x040F, // (float4)   matrix, float
    OSKAR_DOUBLE_MATRIX          = 0x040D, // (double4)  matrix, double
    OSKAR_SINGLE_COMPLEX_MATRIX  = 0x04CF, // (float4c)  matrix, complex float
    OSKAR_DOUBLE_COMPLEX_MATRIX  = 0x04CD  // (double4c) matrix, complex double
};

typedef struct oskar_Ptr oskar_Ptr;

#endif // OSKAR_PTR_H_
