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

#include "station/oskar_ElementModel.h"
#include "station/oskar_element_model_copy.h"
#include "station/oskar_element_model_init.h"
#include "station/oskar_element_model_free.h"
#include "station/oskar_element_model_type.h"

oskar_ElementModel::oskar_ElementModel(int type, int location)
{
    if (oskar_element_model_init(this, type, location))
        throw "Error in oskar_element_model_init";
}

oskar_ElementModel::oskar_ElementModel(const oskar_ElementModel* other,
        int location)
{
    if (oskar_element_model_init(this, other->type(), location))
        throw "Error in oskar_element_model_init";
    if (oskar_element_model_copy(this, other))
        throw "Error in oskar_station_model_copy";
}

oskar_ElementModel::~oskar_ElementModel()
{
    if (oskar_element_model_free(this))
        throw "Error in oskar_element_model_free";
}

int oskar_ElementModel::type() const
{
    return oskar_element_model_type(this);
}
