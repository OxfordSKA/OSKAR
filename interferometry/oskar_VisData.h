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

#ifndef OSKAR_VIS_DATA_H_
#define OSKAR_VIS_DATA_H_

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// DEPRECATED
struct oskar_VisData_d
{
    int      num_samples;
    double*  u;
    double*  v;
    double*  w;
    double2* amp;
};
typedef struct oskar_VisData_d oskar_VisData_d;

// DEPRECATED
struct oskar_VisData_f
{
    int      num_samples;
    float*   u;
    float*   v;
    float*   w;
    float2*  amp;
};
typedef struct oskar_VisData_f oskar_VisData_f;



/**
 * @brief Allocate memory for the specified oskar_VisData_d structure.
 *
 * @param[in] num_samples   Number of visibility samples to allocate. This is
 *                          usually num_baselines * num_snapshots
 * @param[in] vis           Pointer to a oskar_VisData_d structure.
 */
OSKAR_EXPORT
void oskar_allocate_vis_data_d(const unsigned num_samples, oskar_VisData_d* vis);

OSKAR_EXPORT
void oskar_allocate_vis_data_f(const unsigned num_samples, oskar_VisData_f* vis);


/**
 * @brief Free memory held in the specified oskar_VisData_d structure.
 *
 * @param[in] vis          Pointer to a oskar_VisData_d structure.
 */
OSKAR_EXPORT
void oskar_free_vis_data_d(oskar_VisData_d* vis);

OSKAR_EXPORT
void oskar_free_vis_data_f(oskar_VisData_f* vis);

/**
 * @brief Writes a oskar_VisData_d structure to the specified file.
 *
 * @param[in] filename    Filename to write to.
 * @param[in] vis         Pointer to oskar_VisData_d structure to be written.
 */
OSKAR_EXPORT
void oskar_write_vis_data_d(const char* filename, const oskar_VisData_d* vis);

OSKAR_EXPORT
void oskar_write_vis_data_f(const char* filename, const oskar_VisData_f* vis);


/**
 * @brief Loads a oskar_VisData_d structure from the specified file.
 *
 * @details
 * note: This function will allocate memory for the oskar_VisData_d structure
 * internally.
 *
 * @param[in]  filename  Filename holding the oskar_VisData_d data to load.
 * @param[out] vis       Structure holding the loaded data.
 */
OSKAR_EXPORT
void oskar_load_vis_data_d(const char* filename, oskar_VisData_d* vis);

OSKAR_EXPORT
void oskar_load_vis_data_f(const char* filename, oskar_VisData_f* vis);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_VIS_DATA_H_
