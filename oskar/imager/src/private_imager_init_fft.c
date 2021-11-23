/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/private_imager_init_fft.h"
#include "imager/oskar_grid_functions_spheroidal.h"
#include "imager/oskar_grid_functions_pillbox.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_init_fft(oskar_Imager* h, int* status)
{
    oskar_Mem* tmp = 0;
    if (*status) return;

    /* Generate the convolution function. */
    tmp = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            h->oversample * (h->support + 1), status);
    switch (h->kernel_type)
    {
    case 'S':
        oskar_grid_convolution_function_spheroidal(h->support, h->oversample,
                oskar_mem_double(tmp, status));
        break;
    case 'P':
        oskar_grid_convolution_function_pillbox(h->support, h->oversample,
                oskar_mem_double(tmp, status));
        break;
    default:
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        break;
    }

    /* Save the convolution function with appropriate numerical precision. */
    oskar_mem_free(h->conv_func, status);
    h->conv_func = oskar_mem_convert_precision(tmp, h->imager_prec, status);
    oskar_mem_free(tmp, status);

    /* Copy to device memory if required. */
    if (h->grid_on_gpu && h->num_devices > 0)
    {
        int i = 0;
        if (h->num_devices < h->num_gpus)
        {
            oskar_imager_set_num_devices(h, h->num_gpus);
        }
        for (i = 0; i < h->num_gpus; ++i)
        {
            DeviceData* d = &h->d[i];
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
            if (*status) break;
            oskar_mem_free(d->conv_func, status);
            d->conv_func = oskar_mem_create_copy(h->conv_func,
                    h->dev_loc, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
