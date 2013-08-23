/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_MEM_TO_TYPE_H_
#define OSKAR_MEM_TO_TYPE_H_

/**
 * @file oskar_mem_to_type.h
 */

#include "oskar_global.h"

#ifdef OSKAR_HAVE_CUDA
#include <vector_types.h>
#endif
#include <oskar_vector_types.h>

#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

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
void* oskar_mem_to_void(oskar_Mem* mem);

/**
 * @brief
 * Converts the pointer held in the structure to a char*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * char*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
char* oskar_mem_to_char(oskar_Mem* mem, int* status);

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
int* oskar_mem_to_int(oskar_Mem* mem, int* status);

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
float* oskar_mem_to_float(oskar_Mem* mem, int* status);

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
float2* oskar_mem_to_float2(oskar_Mem* mem, int* status);

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
float4c* oskar_mem_to_float4c(oskar_Mem* mem, int* status);

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
double* oskar_mem_to_double(oskar_Mem* mem, int* status);

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
double2* oskar_mem_to_double2(oskar_Mem* mem, int* status);

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
double4c* oskar_mem_to_double4c(oskar_Mem* mem, int* status);

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
const void* oskar_mem_to_const_void(const oskar_Mem* mem);

/**
 * @brief
 * Converts the pointer held in the structure to a const char*.
 *
 * @details
 * This function returns the pointer held in the memory structure as a
 * const char*.
 *
 * If the base type of the memory is not correct, the status code will be set
 * to OSKAR_ERR_TYPE_MISMATCH.
 *
 * @param[in]     mem    Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer to access the data as the required type.
 */
const char* oskar_mem_to_const_char(const oskar_Mem* mem, int* status);

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
const int* oskar_mem_to_const_int(const oskar_Mem* mem, int* status);

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
const float* oskar_mem_to_const_float(const oskar_Mem* mem, int* status);

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
const float2* oskar_mem_to_const_float2(const oskar_Mem* mem, int* status);

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
const float4c* oskar_mem_to_const_float4c(const oskar_Mem* mem, int* status);

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
const double* oskar_mem_to_const_double(const oskar_Mem* mem, int* status);

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
const double2* oskar_mem_to_const_double2(const oskar_Mem* mem, int* status);

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
const double4c* oskar_mem_to_const_double4c(const oskar_Mem* mem, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_TO_TYPE_H_ */
