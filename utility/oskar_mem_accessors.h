/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_MEM_ACCESSORS_H_
#define OSKAR_MEM_ACCESSORS_H_

/**
 * @file oskar_mem_accessors.h
 */

#include <oskar_global.h>

#include <oskar_vector_types.h>

#include <stdlib.h> /* For size_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Returns true if the structure holds a valid pointer; false if not.
 *
 * @details
 * Returns true if the structure holds a valid pointer; false if not.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return True if the pointer is not NULL.
 */
OSKAR_EXPORT
int oskar_mem_allocated(const oskar_Mem* mem);

/**
 * @brief
 * Returns the number of elements in the memory block.
 *
 * @details
 * This accessor function returns the number of elements in the memory block.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return The number of elements.
 */
OSKAR_EXPORT
size_t oskar_mem_length(const oskar_Mem* mem);

/**
 * @brief
 * Returns the enumerated location of the memory block.
 *
 * @details
 * This accessor function returns the enumerated location of the memory block.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return The memory location (OSKAR_CPU or OSKAR_GPU).
 */
OSKAR_EXPORT
int oskar_mem_location(const oskar_Mem* mem);

/**
 * @brief
 * Returns the enumerated type of the memory block.
 *
 * @details
 * This accessor function returns the enumerated type of the memory block.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return The full enumerated memory element type.
 */
OSKAR_EXPORT
int oskar_mem_type(const oskar_Mem* mem);

/**
 * @brief
 * Returns the base type (precision) of the data.
 *
 * @details
 * Returns the base type (precision) of the data.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return The base type (OSKAR_CHAR, OSKAR_INT, OSKAR_SINGLE, OSKAR_DOUBLE).
 */
OSKAR_EXPORT
int oskar_mem_precision(const oskar_Mem* mem);

/**
 * @brief
 * Returns true if the data is in double precision.
 *
 * @details
 * Returns true if the data is in double precision.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return True (1) if double, else false (0).
 */
OSKAR_EXPORT
int oskar_mem_is_double(const oskar_Mem* mem);

/**
 * @brief
 * Returns true if the data is in single precision.
 *
 * @details
 * Returns true if the data is in single precision.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return True (1) if single, else false (0).
 */
OSKAR_EXPORT
int oskar_mem_is_single(const oskar_Mem* mem);

/**
 * @brief
 * Returns true if the data is a complex type.
 *
 * @details
 * Returns true if the data is a complex type.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return True (1) if complex, else false (0).
 */
OSKAR_EXPORT
int oskar_mem_is_complex(const oskar_Mem* mem);

/**
 * @brief
 * Returns true if the data is not a complex type.
 *
 * @details
 * Returns true if the data is not a complex type.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return True (1) if purely real, else false (0).
 */
OSKAR_EXPORT
int oskar_mem_is_real(const oskar_Mem* mem);

/**
 * @brief
 * Returns true if the data is a 2x2 matrix type.
 *
 * @details
 * Returns true if the data is a 2x2 matrix type.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return True (1) if a 2x2 matrix type, else false (0).
 */
OSKAR_EXPORT
int oskar_mem_is_matrix(const oskar_Mem* mem);

/**
 * @brief
 * Returns true if the data is a scalar (non-matrix) type.
 *
 * @details
 * Returns true if the data is a scalar (non-matrix) type.
 *
 * @param[in] mem Pointer to the memory block.
 *
 * @return True (1) if a scalar type, else false (0).
 */
OSKAR_EXPORT
int oskar_mem_is_scalar(const oskar_Mem* mem);

/**
 * @brief
 * Converts the pointer held in the structure to a void*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * void*.
 *
 * @param[in]     mem    Pointer to data structure.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
void* oskar_mem_void(oskar_Mem* mem);

/**
 * @brief
 * Converts the pointer held in the structure to a const void*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const void*.
 *
 * @param[in]     mem    Pointer to data structure.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const void* oskar_mem_void_const(const oskar_Mem* mem);

/**
 * @brief
 * Converts the pointer held in the structure to a char*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * char*.
 *
 * @param[in]     mem    Pointer to data structure.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
char* oskar_mem_char(oskar_Mem* mem);

/**
 * @brief
 * Converts the pointer held in the structure to a const char*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const char*.
 *
 * @param[in]     mem    Pointer to data structure.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const char* oskar_mem_char_const(const oskar_Mem* mem);

/**
 * @brief
 * Converts the pointer held in the structure to a int*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * int*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
int* oskar_mem_int(oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a const int*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const int*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const int* oskar_mem_int_const(const oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a float*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * float*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
float* oskar_mem_float(oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a const float*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const float*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const float* oskar_mem_float_const(const oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a float2*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * float2*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
float2* oskar_mem_float2(oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a const float2*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const float2*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const float2* oskar_mem_float2_const(const oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a float4c*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * float4c*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
float4c* oskar_mem_float4c(oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a const float4c*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const float4c*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const float4c* oskar_mem_float4c_const(const oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a double*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * double*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
double* oskar_mem_double(oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a const double*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const double*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const double* oskar_mem_double_const(const oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a double2*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * double2*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
double2* oskar_mem_double2(oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a const double2*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const double2*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const double2* oskar_mem_double2_const(const oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a double4c*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * double4c*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
double4c* oskar_mem_double4c(oskar_Mem* mem, int* status);

/**
 * @brief
 * Converts the pointer held in the structure to a const double4c*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const double4c*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
OSKAR_EXPORT
const double4c* oskar_mem_double4c_const(const oskar_Mem* mem, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_ACCESSORS_H_ */
