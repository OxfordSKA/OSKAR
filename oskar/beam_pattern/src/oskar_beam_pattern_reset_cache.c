/*
 * Copyright (c) 2016-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdio.h>
#include <stdlib.h>

#include "beam_pattern/oskar_beam_pattern.h"
#include "beam_pattern/private_beam_pattern.h"
#include "beam_pattern/private_beam_pattern_free_device_data.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_beam_pattern_reset_cache(oskar_BeamPattern* h, int* status)
{
    int i = 0;
    oskar_beam_pattern_free_device_data(h, status);
    oskar_mem_free(h->lon_rad, status);
    oskar_mem_free(h->lat_rad, status);
    oskar_mem_free(h->x, status);
    oskar_mem_free(h->y, status);
    oskar_mem_free(h->z, status);
    oskar_mem_free(h->pix, status);
    oskar_mem_free(h->ctemp, status);
    h->lon_rad = h->lat_rad = h->x = h->y = h->z = h->pix = h->ctemp = NULL;

    /* Close files and free data products. */
    for (i = 0; i < h->num_data_products; ++i)
    {
        if (h->data_products[i].text_file)
        {
            (void) fclose(h->data_products[i].text_file);
        }
        if (h->data_products[i].fits_file)
        {
            ffclos(h->data_products[i].fits_file, status);
        }
        if (h->data_products[i].hdf5_file)
        {
            oskar_hdf5_close(h->data_products[i].hdf5_file);
        }
    }
    free(h->data_products);
    h->data_products = NULL;
    h->num_data_products = 0;
}

#ifdef __cplusplus
}
#endif
