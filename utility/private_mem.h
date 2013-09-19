/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#ifndef OSKAR_PRIVATE_MEM_H_
#define OSKAR_PRIVATE_MEM_H_

/**
 * @file private_mem.h
 */

#include <oskar_global.h>

/**
 * @brief Structure to wrap a memory pointer either on the CPU or GPU.
 *
 * @details
 * This structure holds a pointer to memory either on the CPU
 * or GPU, and defines the type of the data to which it points.
 *
 * The structure will normally take ownership of the memory:
 * If the value of the \p owner variable is set to true, the memory will be
 * released when the structure is freed.
 */
struct oskar_Mem
{
    int type;         /**< Enumerated element type of memory block. */
    int location;     /**< Address space of data pointer. */
    int num_elements; /**< Number of elements in memory block. */
    int owner;        /**< Flag set if the structure owns the memory. */
    void* data;       /**< Data pointer. */

    /* ALL THE FOLLOWING METHODS ARE DEPRECATED */
#ifdef __cplusplus
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
    OSKAR_EXPORT
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
    OSKAR_EXPORT
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
    OSKAR_EXPORT
    oskar_Mem(const oskar_Mem* other, int mem_location, int owner_ = 1);

    /**
     * @brief Destroys the structure.
     *
     * @details
     * If the pointer is not NULL and the ownership flag is set to true,
     * then the memory is also freed.
     */
    OSKAR_EXPORT
    ~oskar_Mem();
#endif
};

#ifndef OSKAR_MEM_TYPEDEF_
#define OSKAR_MEM_TYPEDEF_
typedef struct oskar_Mem oskar_Mem;
#endif /* OSKAR_MEM_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_MEM_H_ */
