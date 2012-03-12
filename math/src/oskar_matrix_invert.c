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


#include "math/oskar_matrix_invert.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* calculate the cofactor of element (row, col) */
int get_minor(oskar_Mem* src, oskar_Mem* dst, int row, int col, int order);

/* Calculate the determinant recursively */
double calc_det(oskar_Mem* M, int order);


int oskar_matrix_invert(oskar_Mem* M, int order)
{
    int j, i, type, location;
    double det;
    oskar_Mem minor, Y;

    /* FIXME assumes double and CPU */
    if (M == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (M->type != OSKAR_DOUBLE || M->location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;

    type = OSKAR_DOUBLE;
    location = OSKAR_LOCATION_CPU;

    /* Get the determinant of M */
    det = 1.0/calc_det(M, order);

    /* Allocate memory */
    oskar_mem_init(&minor, type, location, (order-1)*(order-1), OSKAR_TRUE);
    oskar_mem_init(&Y, type, location, order*order, OSKAR_TRUE);

    for (j = 0; j < order; ++j)
    {
        for (i = 0; i < order; ++i)
        {
            get_minor(M, &minor, j, i, order);
            ((double*)Y.data)[i * order + j] = det * calc_det(&minor, order-1);
            if ((i+j)%2 == 1)
                ((double*)Y.data)[i * order + j] *= -1;
        }
    }
    for (j = 0; j < order; ++j)
    {
        for (i = 0; i < order; ++i)
        {
            ((double*)M->data)[j * order + i] = ((double*)Y.data)[j * order + i];
        }
    }

    oskar_mem_free(&minor);
    oskar_mem_free(&Y);

    return OSKAR_SUCCESS;
}


int get_minor(oskar_Mem* src, oskar_Mem* dst, int row, int col, int order)
{
    /* FIXME assumes double and CPU */

    /* indicate which col and row is being copied to dst */
    int col_count, row_count, i, j;
    col_count = 0;
    row_count = 0;

    for (i = 0; i < order; ++i)
    {
        if (i != row)
        {
            col_count = 0;
            for (j = 0; j < order; ++j)
            {
                /* when j is not the element */
                if (j != col)
                {
                    ((double*)dst->data)[row_count * (order-1) + col_count] =
                            ((double*)src->data)[i * (order) + j];
                    col_count++;
                }
            }
            row_count ++;
        }
    }
    return 1;
}


double calc_det(oskar_Mem* M, int order)
{
    /* FIXME assumes double and CPU */

    int i, type, location;
    double det;
    oskar_Mem minor;

    det = 0.0;
    type = OSKAR_DOUBLE;
    location = OSKAR_LOCATION_CPU;

    if (order == 1)
        return ((double*)M->data)[0];

    /* Allocate the co-factor matrix */
    oskar_mem_init(&minor, type, location, (order-1)*(order-1), OSKAR_TRUE);

    for (i = 0; i < order; ++i)
    {
        /* Get minor element of (0, i) */
        get_minor(M, &minor, 0, i, order);

        /* The recursion is here */
        det += (i%2==1?-1.0:1.0) * ((double*)M->data)[i] * calc_det(&minor, order-1);
        /*det += pow(-1.0, i) * ((double*)M->data)[i] * det(&minor, order-1);*/
    }


    /* Clean up */
    oskar_mem_free(&minor);

    return det;
}


#ifdef __cplusplus
}
#endif
