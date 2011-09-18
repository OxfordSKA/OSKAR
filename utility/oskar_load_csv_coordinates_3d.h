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

#ifndef OSKAR_LOAD_CSV_COORDINATES_3D_H_
#define OSKAR_LOAD_CSV_COORDINATES_3D_H_

/**
 * @file oskar_load_csv_coordinates_3d.h
 */

#include "oskar_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads a comma-separated list of 3-dimensional coordinates from a text file
 * (single precision).
 *
 * @details
 * This function loads a plain text file containing a comma-separated list of
 * 3D coordinates, and returns pointers to data held in memory.
 *
 * WARNING: This function allocates memory to hold the data, which must be
 * freed by the caller when no longer required.
 *
 * Usage:
 * To read a text file containing 4 sets of coordinates represented as pairs of
 * comma-separated values, e.g.
 * \verbatim
 *      1.0, 2.0, 3.0
 *      1.1, 2.1, 3.1
 *      1.2, 2.2, 3.2
 *      1.3, 2.3, 3.3
 * \endverbatim
 *
 * The following code will load the coordinates into memory:
 * \code
 *      float* x = NULL;
 *      float* y = NULL;
 *      float* z = NULL;
 *      unsigned num = 0;
 *      oskar_load_csv_coordinates_3d_f("filename.csv", &num, &x, &y, &z);
 * \endcode
 *
 * @param[in]  filename Filename of the CSV coordinates file.
 * @param[out] n        The number of coordinates loaded.
 * @param[out] x        Pointer to array of x coordinates loaded.
 * @param[out] y        Pointer to array of y coordinates loaded.
 * @param[out] z        Pointer to array of z coordinates loaded.
 *
 * @return The number of coordinates loaded.
 */
DllExport
int oskar_load_csv_coordinates_3d_f(const char* filename, unsigned* n,
        float** x, float** y, float** z);

/**
 * @brief
 * Loads a comma-separated list of 3-dimensional coordinates from a text file
 * (double precision).
 *
 * @details
 * This function loads a plain text file containing a comma-separated list of
 * 3D coordinates, and returns pointers to data held in memory.
 *
 * WARNING: This function allocates memory to hold the data, which must be
 * freed by the caller when no longer required.
 *
 * Usage:
 * To read a text file containing 4 sets of coordinates represented as pairs of
 * comma-separated values, e.g.
 * \verbatim
 *      1.0, 2.0, 3.0
 *      1.1, 2.1, 3.1
 *      1.2, 2.2, 3.2
 *      1.3, 2.3, 3.3
 * \endverbatim
 *
 * The following code will load the coordinates into memory:
 * \code
 *      double* x = NULL;
 *      double* y = NULL;
 *      double* z = NULL;
 *      unsigned num = 0;
 *      oskar_load_csv_coordinates_3d_d("filename.csv", &num, &x, &y, &z);
 * \endcode
 *
 * @param[in]  filename Filename of the CSV coordinates file.
 * @param[out] n        The number of coordinates loaded.
 * @param[out] x        Pointer to array of x coordinates loaded.
 * @param[out] y        Pointer to array of y coordinates loaded.
 * @param[out] z        Pointer to array of z coordinates loaded.
 *
 * @return The number of coordinates loaded. */
DllExport
int oskar_load_csv_coordinates_3d_d(const char* filename, unsigned* n,
        double** x, double** y, double** z);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_LOAD_CSV_COORDINATES_3D_H_
