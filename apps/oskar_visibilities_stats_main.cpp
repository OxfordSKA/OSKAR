/*
 * Copyright (c) 2013, The University of Oxford
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


#include <oskar_global.h>

#include <apps/lib/oskar_OptionParser.h>

#include "utility/oskar_vector_types.h"
#include <utility/oskar_Log.h>
#include <utility/oskar_log_settings.h>
#include <utility/oskar_get_error_string.h>

#include <interferometry/oskar_visibilities_read.h>
#include <interferometry/oskar_visibilities_write.h>
#include <interferometry/oskar_visibilities_init.h>
#include <interferometry/oskar_visibilities_copy.h>

#include <string>
#include <vector>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <cstdio>

using namespace std;

//------------------------------------------------------------------------------
void set_options(oskar_OptionParser& opt);
bool check_options(oskar_OptionParser& opt, int argc, char** argv);
void check_error(int status);
char *doubleToRawString(double x) {
    // Assumes sizeof(long long) == 8.

    char *buffer = new char[32];
    sprintf(buffer, "%llx", *(unsigned long long *)&x);  // Evil!
    return buffer;
}
//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    typedef float2 Complex;
    typedef double2 DComplex;
    typedef float4c Jones;
    typedef double4c DJones;

    int status = OSKAR_SUCCESS;

    // Register options ========================================================
    oskar_OptionParser opt("oskar_visibility_stats");
    set_options(opt);
    if (!check_options(opt, argc, argv))
        return OSKAR_FAIL;

    // Retrieve options ========================================================
    vector<string> vis_filename = opt.getArgs();
    bool verbose = opt.isSet("-v") ? true : false;

    if (verbose)
    {
        cout << endl;
        cout << "-----------------------------------------------------------" << endl;
        cout << "Number of visibility files = " << vis_filename.size() << endl;
        for (int i = 0; i < (int)vis_filename.size(); ++i)
            cout << "  --> " << vis_filename[i] << endl;
        cout << "-----------------------------------------------------------" << endl;
        cout << endl;
    }


    for (int i = 0; i < (int)vis_filename.size(); ++i)
    {
        if (verbose)
            cout << "Loading visibility file: " << vis_filename[i] << endl;

        oskar_Visibilities vis;
        oskar_visibilities_read(&vis, vis_filename[i].c_str(), &status);
        check_error(status);
        double min = DBL_MAX, max = -DBL_MAX;
        double mean = 0.0, rms = 0.0, var = 0.0, std = 0.0;
        double sum = 0.0, sumsq = 0.0;
        if (verbose)
        {
            cout << "  No. of baselines: " << vis.num_baselines << endl;
            cout << "  No. of times: " << vis.num_times << endl;
            cout << "  No. of channels: " << vis.num_channels << endl;
        }

        switch (vis.amplitude.type)
        {
            case OSKAR_SINGLE_COMPLEX:
            {
                Complex* amps = (Complex*)vis.amplitude.data;
                break;
            }
            case OSKAR_SINGLE_COMPLEX_MATRIX:
            {
                Jones* amps = (Jones*)vis.amplitude.data;
                break;
            }
            case OSKAR_DOUBLE_COMPLEX:
            {
                DComplex* amps = (DComplex*)vis.amplitude.data;
                break;
            }
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
            {
                DJones* amp = (DJones*)vis.amplitude.data;
                int num_zero = 0;
                for (int i = 0, c = 0; c < vis.num_channels; ++c)
                {
                    for (int t = 0; t < vis.num_times; ++t)
                    {
                        for (int b = 0; b < vis.num_baselines; ++b, ++i)
                        {
                            DComplex xx = amp[i].a;
                            DComplex yy = amp[i].d;
                            DComplex I;
                            I.x = 0.5 * (xx.x + yy.x);
                            I.y = 0.5 * (xx.y + yy.y);
                            double absI = std::sqrt(I.x * I.x + I.y * I.y);
                            if (absI < DBL_MIN)
                                num_zero++;
                            //printf("%i (%e [%s] %e) %e %e => %e\n", i, xx.x, doubleToRawString(xx.x), xx.y, I.x, I.y, absI);
                            if (absI > max) max = absI;
                            if (absI < min) min = absI;
                            sum += absI;
                            sumsq += absI * absI;
                        }
                    }
                }
                int num_vis = vis.amplitude.num_elements;
                mean = sum / num_vis;
                rms = std::sqrt(sumsq / num_vis);
                var = sumsq/num_vis - mean*mean;
                std = std::sqrt(var);
                printf("min, max, mean, rms, std\n");
                printf("%e, %e, %e, %e, %e\n", min, max, mean, rms, std);
                printf("number of zero visibilities = %i\n", num_zero);
                printf("number of non-zero visibilities = %i\n", num_vis-num_zero);
                printf("percent zero visibilities = %f\n", (double)num_zero/num_vis * 100.0);
                break;
            }
            default:
                return OSKAR_ERR_BAD_DATA_TYPE;
                break;
        } // switch (vis.amplitude.type)

    } // end of loop over visibility files


    // XXX free memory visibilities structure...

    return status;
}


//------------------------------------------------------------------------------

void set_options(oskar_OptionParser& opt)
{
    opt.setDescription("Application to generate some stats from an OSKAR visibility file.");
    opt.addRequired("OSKAR visibility file(s)...");
    opt.addFlag("-v", "Verbose");
}

bool check_options(oskar_OptionParser& opt, int argc, char** argv)
{
    if (!opt.check_options(argc, argv))
        return false;
    return true;
}

void check_error(int status)
{
    if (status != OSKAR_SUCCESS) {
        cout << "ERROR: code[" << status << "] ";
        cout << oskar_get_error_string(status) << endl;
        exit(status);
    }
}

