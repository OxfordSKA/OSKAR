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

#ifndef OSKAR_MEM_H_
#define OSKAR_MEM_H_

/**
 * @file oskar_Mem.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
/* Forward declarations. */
struct float2;
struct float4c;
struct double2;
struct double4c;
#endif

/**
 * @brief Structure to wrap a memory pointer either on the CPU or GPU.
 *
 * @details
 * This structure holds a pointer to memory either on the CPU
 * or GPU, and defines the type of the data that it points to.
 *
 * In C++, the structure can take ownership of the memory: If the value of the
 * \p owner variable is set to true, the memory will be released
 * automatically when the structure is deleted.
 */
struct OSKAR_EXPORT oskar_Mem
{
    int type;         /**< Enumerated element type of memory block. */
    int location;     /**< Address space of data pointer. */
    int num_elements; /**< Number of elements in memory block. */
    int owner;        /**< Flag set if the structure owns the memory. */
    void* data;       /**< Data pointer. */

#ifdef __cplusplus
    /* If C++, then provide constructors and a destructor. */
    /**
     * @brief Constructs an oskar_Mem data structure.
     *
     * @details
     * Constructs a new oskar_Mem data structure.
     * The pointer and data types are all set to 0.
     *
     * @param[in] owner_ Bool flag specifying if the structure should take
     *                   ownership of the memory (default = true).
     */
    oskar_Mem(int owner_ = 1);

    /**
     * @brief Constructs and allocates data for an oskar_Mem data structure.
     *
     * @details
     * Constructs a new oskar_Mem data structure, allocating memory for it in
     * the specified location.
     *
     * @param[in] mem_type     Enumerated data type of memory contents (magic number).
     * @param[in] mem_location Specify 0 for host memory, 1 for device memory.
     * @param[in] size         Number of elements of type \p type in the array.
     * @param[in] owner_       Bool flag specifying if the structure should take
     *                         ownership of the memory (default = true).
     */
    oskar_Mem(int mem_type, int mem_location, int size = 0, int owner_ = 1);

    /**
     * @brief Constructs and allocates data for an oskar_Mem data structure.
     *
     * @details
     * Constructs a new oskar_Mem data structure, allocating memory for it in
     * the specified location.
     *
     * @param[in] other        Pointer to another oskar_Mem data structure.
     * @param[in] mem_location Specify 0 for host memory, 1 for device memory.
     * @param[in] owner_       Bool flag specifying if the structure should
     *                         take ownership of the memory (default = true).
     */
    oskar_Mem(const oskar_Mem* other, int mem_location, int owner_ = 1);

    /**
     * @brief Destroys the structure.
     *
     * @details
     * If the pointer is not NULL and the ownership flag is set to true,
     * then the memory is also freed.
     */
    ~oskar_Mem();

    /* Convenience pointer casts. */
    operator char*() {return (char*)data;}
    operator int*() {return (int*)data;}
    operator float*() {return (float*)data;}
    operator double*() {return (double*)data;}
    operator float2*() {return (float2*)data;}
    operator double2*() {return (double2*)data;}
    operator float4c*() {return (float4c*)data;}
    operator double4c*() {return (double4c*)data;}
    operator const char*() const {return (const char*)data;}
    operator const int*() const {return (const int*)data;}
    operator const float*() const {return (const float*)data;}
    operator const double*() const {return (const double*)data;}
    operator const float2*() const {return (const float2*)data;}
    operator const double2*() const {return (const double2*)data;}
    operator const float4c*() const {return (const float4c*)data;}
    operator const double4c*() const {return (const double4c*)data;}
#endif
};
typedef struct oskar_Mem oskar_Mem;

/* Define an enumerator for the type.
 *
 * IMPORTANT:
 * 1. All these must be small enough to fit into one byte (8 bits) only.
 * 2. To maintain binary data compatibility, do not modify any numbers
 *    that appear in the list below!
 */
enum OSKAR_MEM_TYPE
{
    /* Byte (char): bit 0 set. */
    OSKAR_CHAR                   = 0x01,

    /* Integer (int): bit 1 set. */
    OSKAR_INT                    = 0x02,

    /* Scalar single (float): bit 2 set. */
    OSKAR_SINGLE                 = 0x04,

    /* Scalar double (double): bit 3 set. */
    OSKAR_DOUBLE                 = 0x08,

    /* Complex flag: bit 5 set. */
    OSKAR_COMPLEX                = 0x20,

    /* Matrix flag: bit 6 set. */
    OSKAR_MATRIX                 = 0x40,

    /* Scalar complex single (float2). */
    OSKAR_SINGLE_COMPLEX         = OSKAR_SINGLE | OSKAR_COMPLEX,

    /* Scalar complex double (double2). */
    OSKAR_DOUBLE_COMPLEX         = OSKAR_DOUBLE | OSKAR_COMPLEX,

    /* Matrix complex float (float4c). */
    OSKAR_SINGLE_COMPLEX_MATRIX  = OSKAR_SINGLE | OSKAR_COMPLEX | OSKAR_MATRIX,

    /* Matrix complex double (double4c). */
    OSKAR_DOUBLE_COMPLEX_MATRIX  = OSKAR_DOUBLE | OSKAR_COMPLEX | OSKAR_MATRIX
};
typedef enum OSKAR_MEM_TYPE oskar_mem_type;

enum OSKAR_MEM_LOCATION
{
    OSKAR_LOCATION_CPU = 0,
    OSKAR_LOCATION_GPU = 1
};
typedef enum OSKAR_MEM_LOCATION oskar_mem_location;

#endif /* OSKAR_MEM_H_ */
