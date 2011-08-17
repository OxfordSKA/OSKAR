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

#ifndef OSKAR_LOAD_CSV_COORDINATES_H_
#define OSKAR_LOAD_CSV_COORDINATES_H_

/**
 * @file oskar_load_csv_coordinates.h
 */

/**
 * @brief
 * Loads a comma separated list of 2-dimensional co-ordinates from a plain text
 * file.
 *
 * @details
 * A plain text file containing a comma separated list of co-ordinates (x,y)
 * is loaded into the provided pointers.
 *
 * *************************************************************************
 * !WARNING! This function will allocate memory for loaded arrays internally.
 * **************************************************************************
 *
 * Usage:
 * To read a text file containing 4 sets of coordinates represented as pairs of
 * comma separated values. e.g.
 * \verbatim
 *      1.0, 2.0
 *      1.1, 2.1
 *      1.2, 2.2
 *      1.3, 2.3
 * \endverbatim
 *
 * The following code will load the coordinates into memory.
 * \code
 *      double* x = NULL;
 *      double* y = NULL;
 *      unsigned num = 0;
 *      oskar_load_csv_coordinates("filename.dat", &num, &x, &y);
 * \endcode
 *
 * @param[in]  filename Filename of the csv coordinates file.
 * @param[out] n        Pointer to number of coordinates loaded.
 * @param[out] x        Pointer to array of x coordinates loaded.
 * @param[out] y        Pointer to array of y coordinates loaded.
 * @return Number of coordinates loaded.
 */
int oskar_load_csv_coordinates(const char* filename, unsigned* n,
        double** x, double** y);

#endif // OSKAR_LOAD_CSV_COORDINATES_H_
