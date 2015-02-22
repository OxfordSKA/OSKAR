/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <apps/lib/oskar_OptionParser.h>

#include <oskar_log.h>
#include <oskar_get_error_string.h>
#include <oskar_vis.h>
#include <oskar_version_string.h>

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
//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int status = OSKAR_SUCCESS;

    // Register options ========================================================
    oskar_OptionParser opt("oskar_vis_stats", oskar_version_string());
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

        oskar_Binary* h = oskar_binary_create(vis_filename[i].c_str(), 'r',
                &status);
        oskar_Vis* vis = oskar_vis_read(h, &status);
        oskar_binary_free(h);
        check_error(status);
        double min = DBL_MAX, max = -DBL_MAX;
        double mean = 0.0, rms = 0.0, var = 0.0, std = 0.0;
        double sum = 0.0, sumsq = 0.0;
        int num_vis = 0, num_zero = 0;
        if (verbose)
        {
            cout << "  No. baselines: " << oskar_vis_num_baselines(vis) << endl;
            cout << "  No. times: " << oskar_vis_num_times(vis) << endl;
            cout << "  No. channels: " << oskar_vis_num_channels(vis) << endl;
        }
        if (oskar_mem_type(oskar_vis_amplitude(vis)) ==
                OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float4c* amp = oskar_mem_float4c(oskar_vis_amplitude(vis), &status);
            for (int i = 0, c = 0; c < oskar_vis_num_channels(vis); ++c)
            {
                for (int t = 0; t < oskar_vis_num_times(vis); ++t)
                {
                    for (int b = 0; b < oskar_vis_num_baselines(vis); ++b, ++i)
                    {
                        float2 xx = amp[i].a;
                        float2 yy = amp[i].d;
                        float2 I;
                        I.x = 0.5 * (xx.x + yy.x);
                        I.y = 0.5 * (xx.y + yy.y);
                        float absI = sqrtf(I.x*I.x + I.y*I.y);
                        if (absI < DBL_MIN)
                            num_zero++;
                        if (absI > max) max = absI;
                        if (absI < min) min = absI;
                        sum += absI;
                        sumsq += absI * absI;
                    }
                }
            }
            num_vis = (int)oskar_mem_length(oskar_vis_amplitude(vis));
            mean = sum / num_vis;
            rms = sqrtf(sumsq / num_vis);
            var = sumsq/num_vis - mean*mean;
            std = sqrtf(var);
        }
        else if (oskar_mem_type(oskar_vis_amplitude(vis)) ==
                OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            double4c* amp = oskar_mem_double4c(oskar_vis_amplitude(vis), &status);
            for (int i = 0, c = 0; c < oskar_vis_num_channels(vis); ++c)
            {
                for (int t = 0; t < oskar_vis_num_times(vis); ++t)
                {
                    for (int b = 0; b < oskar_vis_num_baselines(vis); ++b, ++i)
                    {
                        double2 xx = amp[i].a;
                        double2 yy = amp[i].d;
                        double2 I;
                        I.x = 0.5 * (xx.x + yy.x);
                        I.y = 0.5 * (xx.y + yy.y);
                        double absI = std::sqrt(I.x * I.x + I.y * I.y);
                        if (absI < DBL_MIN)
                            num_zero++;
                        if (absI > max) max = absI;
                        if (absI < min) min = absI;
                        sum += absI;
                        sumsq += absI * absI;
                    }
                }
            }
            num_vis = (int)oskar_mem_length(oskar_vis_amplitude(vis));
            mean = sum / num_vis;
            rms = std::sqrt(sumsq / num_vis);
            var = sumsq/num_vis - mean*mean;
            std = std::sqrt(var);
        }
        else
        {
            return OSKAR_ERR_BAD_DATA_TYPE;
        }

        printf("min, max, mean, rms, std\n");
        printf("%e, %e, %e, %e, %e\n", min, max, mean, rms, std);
        printf("number of zero visibilities = %i\n", num_zero);
        printf("number of non-zero visibilities = %i\n", num_vis-num_zero);
        printf("percent zero visibilities = %f\n", (double)num_zero/num_vis * 100.0);

        // Free visibility data.
        oskar_vis_free(vis, &status);
    } // end of loop over visibility files

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

