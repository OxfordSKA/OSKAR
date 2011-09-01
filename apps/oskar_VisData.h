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

#ifndef OSKAR_VISDATA_CLASS_H_
#define OSKAR_VISDATA_CLASS_H_

#include "utility/oskar_vector_types.h"
#include <vector>

class oskar_VisData
{
    public:
        oskar_VisData(const unsigned num_stations, const unsigned num_vis_dumps);
        ~oskar_VisData();

    public:
        void write(const char* filename);

        void load(const char* filename);

        double* u() { return &_u[0]; }
        double* v() { return &_v[0]; }
        double* w() { return &_w[0]; }
        double2* vis() { return &_vis[0]; }

        unsigned size() const { return _num_vis_coordinates; }
        unsigned num_baselines() const { return _num_baselines; }

    private:
        unsigned _num_baselines;
        unsigned _num_vis_coordinates;
        std::vector<double>  _u;
        std::vector<double>  _v;
        std::vector<double>  _w;
        std::vector<double2> _vis;
};


#endif // OSKAR_VISDATA_CLASS_H_
