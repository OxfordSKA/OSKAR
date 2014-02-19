/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <private_element.h>
#include <oskar_element.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Element* oskar_element_create(int type, int location, int* status)
{
    oskar_Element* data = 0;

    /* Check all inputs. */
    if (!status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Allocate and initialise the structure. */
    data = (oskar_Element*) malloc(sizeof(oskar_Element));
    if (!data)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }

    /* Initialise variables. */
    data->data_type = type;
    data->data_location = location;
    data->element_type = OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE;
    data->taper_type = OSKAR_ELEMENT_TAPER_NONE;
    data->cos_power = 0.0;
    data->gaussian_fwhm_rad = 0.0;

    /* Check type. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Initialise memory. */
    data->filename_x = oskar_mem_create(OSKAR_CHAR, location, 0, status);
    data->filename_y = oskar_mem_create(OSKAR_CHAR, location, 0, status);
    data->phi_re_x = oskar_splines_create(type, location, status);
    data->phi_im_x = oskar_splines_create(type, location, status);
    data->theta_re_x = oskar_splines_create(type, location, status);
    data->theta_im_x = oskar_splines_create(type, location, status);
    data->phi_re_y = oskar_splines_create(type, location, status);
    data->phi_im_y = oskar_splines_create(type, location, status);
    data->theta_re_y = oskar_splines_create(type, location, status);
    data->theta_im_y = oskar_splines_create(type, location, status);

    /* Return pointer to the structure. */
    return data;
}

#ifdef __cplusplus
}
#endif
