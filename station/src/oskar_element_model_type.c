/*
 * Copyright (c) 2012, The University of Oxford
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

#include "math/oskar_spline_data_type.h"
#include "station/oskar_element_model_type.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_type_check.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_element_model_is_type(const oskar_ElementModel* element, int type)
{
    return (oskar_spline_data_type(&element->phi_re_x) == type &&
            oskar_spline_data_type(&element->theta_re_x) == type &&
            oskar_spline_data_type(&element->phi_re_y) == type &&
            oskar_spline_data_type(&element->theta_re_y) == type &&
            oskar_spline_data_type(&element->phi_im_x) == type &&
            oskar_spline_data_type(&element->theta_im_x) == type &&
            oskar_spline_data_type(&element->phi_im_y) == type &&
            oskar_spline_data_type(&element->theta_im_y) == type);
}

int oskar_element_model_type(const oskar_ElementModel* element)
{
    if (element == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_element_model_is_type(element, OSKAR_DOUBLE))
        return OSKAR_DOUBLE;
    else if (oskar_element_model_is_type(element, OSKAR_SINGLE))
        return OSKAR_SINGLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;
}

#ifdef __cplusplus
}
#endif
