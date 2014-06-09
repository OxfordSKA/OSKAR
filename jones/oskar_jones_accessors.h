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

#ifndef OSKAR_JONES_ACCESSORS_H_
#define OSKAR_JONES_ACCESSORS_H_

/**
 * @file oskar_jones_accessors.h
 */

#include <oskar_global.h>

#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Returns the number of sources in the Jones matrix block.
 *
 * @details
 * Returns the number of sources in the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The number of stations.
 */
OSKAR_EXPORT
int oskar_jones_num_sources(const oskar_Jones* jones);

/**
 * @brief
 * Returns the number of stations in the Jones matrix block.
 *
 * @details
 * Returns the number of stations in the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The number of stations.
 */
OSKAR_EXPORT
int oskar_jones_num_stations(const oskar_Jones* jones);

/**
 * @brief
 * Returns the enumerated data type of the Jones matrix block.
 *
 * @details
 * Returns the enumerated data type of the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The enumerated data type.
 */
OSKAR_EXPORT
int oskar_jones_type(const oskar_Jones* jones);

/**
 * @brief
 * Returns the enumerated location of the Jones matrix block.
 *
 * @details
 * Returns the enumerated location (OSKAR_CPU or OSKAR_GPU)
 * of the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The enumerated location.
 */
OSKAR_EXPORT
int oskar_jones_mem_location(const oskar_Jones* jones);

/**
 * @brief
 * Returns a pointer to the matrix block memory.
 *
 * @details
 * Returns a pointer to the matrix block memory.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return A pointer to the memory structure.
 */
OSKAR_EXPORT
oskar_Mem* oskar_jones_mem(oskar_Jones* jones);

/**
 * @brief
 * Returns a read-only pointer to the matrix block memory.
 *
 * @details
 * Returns a read-only pointer to the matrix block memory.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return A pointer to the memory structure.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_jones_mem_const(const oskar_Jones* jones);

/**
 * @brief
 * Returns a pointer to the matrix block as a float2.
 *
 * @details
 * Returns a pointer to the matrix block as a float2.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
float2* oskar_jones_float2(oskar_Jones* jones, int* status);

/**
 * @brief
 * Returns a pointer to the matrix block as a const float2.
 *
 * @details
 * Returns a pointer to the matrix block as a const float2.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
const float2* oskar_jones_float2_const(const oskar_Jones* jones, int* status);

/**
 * @brief
 * Returns a pointer to the matrix block as a float4c.
 *
 * @details
 * Returns a pointer to the matrix block as a float4c.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
float4c* oskar_jones_float4c(oskar_Jones* jones, int* status);

/**
 * @brief
 * Returns a pointer to the matrix block as a const float4c.
 *
 * @details
 * Returns a pointer to the matrix block as a const float4c.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
const float4c* oskar_jones_float4c_const(const oskar_Jones* jones, int* status);

/**
 * @brief
 * Returns a pointer to the matrix block as a double2.
 *
 * @details
 * Returns a pointer to the matrix block as a double2.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
double2* oskar_jones_double2(oskar_Jones* jones, int* status);

/**
 * @brief
 * Returns a pointer to the matrix block as a const double2.
 *
 * @details
 * Returns a pointer to the matrix block as a const double2.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
const double2* oskar_jones_double2_const(const oskar_Jones* jones, int* status);

/**
 * @brief
 * Returns a pointer to the matrix block as a double4c.
 *
 * @details
 * Returns a pointer to the matrix block as a double4c.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
double4c* oskar_jones_double4c(oskar_Jones* jones, int* status);

/**
 * @brief
 * Returns a pointer to the matrix block as a const double4c.
 *
 * @details
 * Returns a pointer to the matrix block as a const double4c.
 *
 * @param[in]     jones  Pointer to data structure.
 * @param[in,out] status Status return code.
 *
 * @return A pointer of the requested type.
 */
OSKAR_EXPORT
const double4c* oskar_jones_double4c_const(const oskar_Jones* jones,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_JONES_ACCESSORS_H_ */
