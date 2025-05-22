/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

static void realloc_arrays(oskar_Element* e, int size);

void oskar_element_resize_freq_data(oskar_Element* model, int size,
        int* status)
{
    int i = 0;
    if (*status) return;
    const int old_size = model->num_freq;
    if (size > old_size)
    {
        /* Enlarge the arrays and create new structures. */
        realloc_arrays(model, size);
        for (i = old_size; i < size; ++i)
        {
            model->filename_x[i] = 0;
            model->filename_y[i] = 0;
            model->sph_wave[i] = 0;
            model->sph_wave_feko[i] = 0;
            model->sph_wave_galileo[i] = 0;
            model->l_max[i] = 0;
            model->common_phi_coords[i] = 0;
        }
    }
    else if (size < old_size)
    {
        /* Free old structures and shrink the arrays. */
        for (i = size; i < old_size; ++i)
        {
            oskar_mem_free(model->filename_x[i], status);
            oskar_mem_free(model->filename_y[i], status);
            oskar_mem_free(model->sph_wave[i], status);
            oskar_mem_free(model->sph_wave_feko[i], status);
            oskar_mem_free(model->sph_wave_galileo[i], status);
        }
        realloc_arrays(model, size);
    }
    model->num_freq = size;
}

static void realloc_arrays(oskar_Element* e, int size)
{
    const size_t sz = size * sizeof(void*);
    e->freqs_hz = (double*) realloc(e->freqs_hz, size * sizeof(double));
    e->l_max = (int*) realloc(e->l_max, size * sizeof(int));
    e->common_phi_coords = (int*) realloc(
            e->common_phi_coords, size * sizeof(int));
    e->filename_x = (oskar_Mem**) realloc(e->filename_x, sz);
    e->filename_y = (oskar_Mem**) realloc(e->filename_y, sz);
    e->sph_wave = (oskar_Mem**) realloc(e->sph_wave, sz);
    e->sph_wave_feko = (oskar_Mem**) realloc(e->sph_wave_feko, sz);
    e->sph_wave_galileo = (oskar_Mem**) realloc(e->sph_wave_galileo, sz);
}

#ifdef __cplusplus
}
#endif
