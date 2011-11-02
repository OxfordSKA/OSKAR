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
 * This is a valid C-structure that holds a memory pointer either on the CPU
 * or GPU, and defines the type of the data that it points to.
 *
 * If using C++, then the meta-data is made private, and accessor methods
 * are also provided. The C++ interface also provides facility for the structure
 * to take ownership of the memory. If the value of the private_ower member
 * variable flag is set to true the memory will automatically be released
 * when the structure is deleted.
 */
#ifdef __cplusplus
extern "C"
#endif
struct oskar_Mem
{
#ifdef __cplusplus
/* If C++, then make the meta-data private. */
private:
#endif
    int private_type;
    int private_location;
    int private_n_elements;
    int private_owner;      // Bool flag specifying if the C++ interface should
                            // take ownership of the memory.

#ifdef __cplusplus
/* If C++, then make the remaining members public. */
public:
#endif
    void* data; /**< Data pointer.  */

#ifdef __cplusplus
    /* If C++, then provide constructors and a destructor. */
    /**
     * @brief Constructs an oskar_Mem data structure.
     *
     * @details
     * Constructs a new oskar_Mem data structure.
     * The pointer and data types are all set to 0.
     *
     * @param[in] owner Bool flag specifying if the structure should take
     *                  ownership of the memory (default = true).
     */
    oskar_Mem(int ower = 1);

    /**
     * @brief Constructs and allocates data for an oskar_Mem data structure.
     *
     * @details
     * Constructs a new oskar_Mem data structure, allocating memory for it in
     * the specified location.
     *
     * @param[in] type       Enumerated data type of memory contents (magic number).
     * @param[in] location   Specify 0 for host memory, 1 for device memory.
     * @param[in] n_elements Number of elements of type \p type in the array.
     * @param[in] owner      Bool flag specifying if the structure should take
     *                       ownership of the memory (default = true).
     */
    oskar_Mem(int type, int location, int n_elements = 0, int ower = 1);

    /**
     * @brief Constructs and allocates data for an oskar_Mem data structure.
     *
     * @details
     * Constructs a new oskar_Mem data structure, allocating memory for it in
     * the specified location.
     *
     * @param[in] other    Enumerated data type of memory contents (magic number).
     * @param[in] location Specify 0 for host memory, 1 for device memory.
     * @param[in] owner    Bool flag specifying if the structure should take
     *                     ownership of the memory (default = true).
     */
    oskar_Mem(const oskar_Mem* other, int location, int ower = 1);

    /**
     * @brief Destroys the structure.
     *
     * @details
     * If the pointer is not NULL and the ownership flag is set to true,
     * then the memory is also freed.
     */
    ~oskar_Mem();

    /**
     * @brief Copies the memory contents of this structure to another.
     *
     * @details
     * Copies the memory contents and meta-data of this structure to another.
     *
     * @param[in] other Pointer to the oskar_Mem structure to copy.
     *
     * @return A CUDA or OSKAR error code.
     */
    int copy_to(oskar_Mem* other);

    /**
     * @brief
     * Resizes the memory block.
     *
     * @details
     * Resizes the memory to the specified number of elements. The
     * memory type and location are preserved.
     *
     * @param[in] num_elements The required number of elements.
     *
     * @return A CUDA or OSKAR error code.
     */
    int resize(int num_elements);

    /**
     * @brief Appends to the memory by copying num_elements of memory from the
     * specified array with the specified memory location.
     *
     * @param[in] from          Location from which to append to the current memory.
     * @param[in] from_location Location to append from.
     * @param[in] num_elements  Number of elements to append.
     *
     * @return A CUDA or OSKAR error code.
     */
    int append(const void* from, int type, int from_location, int num_elements);
#endif

#ifdef __cplusplus
    /* If C++, then provide read-only accessor functions for the meta-data. */
    int type() const {return private_type;}
    int location() const {return private_location;}
    int n_elements() const {return private_n_elements;}
    bool owner() const { return private_owner; }
    // Warning: using this method can be dangerous if used to allow more than one
    // oskar_Mem structure to own the memory.
    void set_owner(bool value) { private_owner = (int)value; }
    bool is_double() const;
    bool is_complex() const;
    bool is_null() const {return (data == 0);}
    bool is_scalar() const;
    static bool is_double(const int mem_type);
    static bool is_complex(const int mem_type);
    static bool is_scalar(const int mem_type);

    /* Convenience pointer casts. */
    operator float*() {return (float*)data;}
    operator double*() {return (double*)data;}
    operator float2*() {return (float2*)data;}
    operator double2*() {return (double2*)data;}
    operator float4c*() {return (float4c*)data;}
    operator double4c*() {return (double4c*)data;}
    operator const float*() const {return (const float*)data;}
    operator const double*() const {return (const double*)data;}
    operator const float2*() const {return (const float2*)data;}
    operator const double2*() const {return (const double2*)data;}
    operator const float4c*() const {return (const float4c*)data;}
    operator const double4c*() const {return (const double4c*)data;}
#endif
};
typedef struct oskar_Mem oskar_Mem;

/* Define an enumerator for the type. */
enum {
    OSKAR_SINGLE                 = 0x010F, /* (float)    scalar, float */
    OSKAR_DOUBLE                 = 0x010D, /* (double)   scalar, double */
    OSKAR_SINGLE_COMPLEX         = 0x01CF, /* (float2)   scalar, complex float */
    OSKAR_DOUBLE_COMPLEX         = 0x01CD, /* (double2)  scalar, complex double */
    OSKAR_SINGLE_COMPLEX_MATRIX  = 0x04CF, /* (float4c)  matrix, complex float */
    OSKAR_DOUBLE_COMPLEX_MATRIX  = 0x04CD  /* (double4c) matrix, complex double */
};

enum {
    OSKAR_LOCATION_CPU = 0,
    OSKAR_LOCATION_GPU = 1
};

#endif /* OSKAR_MEM_H_ */
