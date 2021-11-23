/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/private_imager_init_dft.h"
#include "imager/private_imager_init_fft.h"
#include "imager/private_imager_init_wproj.h"
#include "utility/oskar_timer.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_check_init(oskar_Imager* h, int* status)
{
    if (*status) return;

    /* Allocate empty weights grids if required. */
    if (!h->weights_grids && h->num_planes > 0)
    {
        int i = 0;
        h->weights_grids = (oskar_Mem**)
                calloc(h->num_planes, sizeof(oskar_Mem*));
        h->weights_guard = (oskar_Mem**)
                calloc(h->num_planes, sizeof(oskar_Mem*));
        for (i = 0; i < h->num_planes; ++i)
        {
            h->weights_grids[i] = oskar_mem_create(h->imager_prec,
                    OSKAR_CPU, 0, status);
            h->weights_guard[i] = oskar_mem_create(h->imager_prec,
                    OSKAR_CPU, 0, status);
        }
    }

    /* Don't continue if we're in "coords only" mode. */
    if (h->coords_only || h->init) return;

    oskar_log_section(h->log, 'M', "Initialising algorithm...");
    oskar_timer_resume(h->tmr_init);
    switch (h->algorithm)
    {
    case OSKAR_ALGORITHM_DFT_2D:
    case OSKAR_ALGORITHM_DFT_3D:
        oskar_imager_init_dft(h, status);
        break;
    case OSKAR_ALGORITHM_FFT:
        oskar_imager_init_fft(h, status);
        break;
    case OSKAR_ALGORITHM_WPROJ:
        oskar_imager_init_wproj(h, status);
        break;
    default:
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }
    oskar_timer_pause(h->tmr_init);
    h->init = 1;
}

#ifdef __cplusplus
}
#endif
