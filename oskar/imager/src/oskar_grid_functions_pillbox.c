/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/oskar_grid_functions_pillbox.h"

#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_convolution_function_pillbox(const int support,
        const int oversample, double* fn)
{
    int i = 0;
    const int gcf_size = oversample * (support + 1);
    const int extent = support * oversample;
    for (i = 0; i < gcf_size; ++i)
    {
        const double nu = (double)i / (double)extent;
        fn[i] = (nu < 0.5) ? 1.0 : 0.0;
    }
}

void oskar_grid_correction_function_pillbox(const int image_size,
        double* fn)
{
    int i = 0;
    for (i = 0; i < image_size; ++i) fn[i] = 1.0; /* No correction! */
}

#ifdef __cplusplus
}
#endif
