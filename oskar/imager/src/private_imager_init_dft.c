/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/private_imager_init_dft.h"
#include "imager/oskar_imager_accessors.h"
#include "math/oskar_evaluate_image_lmn_grid.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_init_dft(oskar_Imager* h, int* status)
{
    size_t num_pixels = 0;
    if (*status) return;

    /* Calculate pixel coordinate grid required for the DFT imager. */
    num_pixels = (size_t) h->image_size;
    num_pixels *= h->image_size;
    oskar_mem_free(h->l, status);
    oskar_mem_free(h->m, status);
    oskar_mem_free(h->n, status);
    h->l = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_pixels, status);
    h->m = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_pixels, status);
    h->n = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_pixels, status);
    oskar_evaluate_image_lmn_grid(h->image_size, h->image_size,
            h->fov_deg * M_PI/180, h->fov_deg * M_PI/180, 0,
            h->l, h->m, h->n, status);
    oskar_mem_add_real(h->n, -1.0, status); /* n-1 */

    /* Expand the number of devices to the number of selected GPUs,
     * if required. */
    if (h->num_devices < h->num_gpus)
    {
        oskar_imager_set_num_devices(h, h->num_gpus);
    }
}

#ifdef __cplusplus
}
#endif
