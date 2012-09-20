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

#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_telescope_model_resize.h"
#include "sky/test/Test_SkyModel.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_append.h"
#include "sky/oskar_sky_model_copy.h"
#include "sky/oskar_sky_model_compute_relative_lmn.h"
#include "sky/oskar_sky_model_filter_by_flux.h"
#include "sky/oskar_sky_model_filter_by_radius.h"
#include "sky/oskar_sky_model_free.h"
#include "sky/oskar_sky_model_horizon_clip.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_load.h"
#include "sky/oskar_sky_model_resize.h"
#include "sky/oskar_sky_model_split.h"
#include "sky/oskar_sky_model_set_source.h"
#include "sky/oskar_evaluate_gaussian_source_parameters.h"
#include "sky/oskar_sky_model_append_to_set.h"
#include "sky/oskar_sky_model_insert.h"
#include "math/oskar_sph_to_lm.h"
#include "utility/oskar_get_error_string.h"

//#define TIMER_ENABLE
#include "utility/timer.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

void Test_SkyModel::test_resize()
{
    int status = 0;
    // Resizing on the GPU in single precision
    {
        oskar_SkyModel* sky = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_GPU, 10);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(10, sky->num_sources);
        oskar_sky_model_resize(sky, 1, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(1, sky->num_sources);
        oskar_sky_model_resize(sky, 20, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(20, sky->num_sources);
        delete sky;
    }
    // Resizing on the CPU in double precision
    {
        oskar_SkyModel* sky = new oskar_SkyModel(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 10);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(10, sky->num_sources);
        oskar_sky_model_resize(sky, 1, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(1, sky->num_sources);
        oskar_sky_model_resize(sky, 20, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(20, sky->num_sources);
        delete sky;
    }
}

void Test_SkyModel::test_set_source()
{
    // Construct a sky model on the GPU of zero size.
    oskar_SkyModel* sky = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_GPU, 0);
    CPPUNIT_ASSERT_EQUAL(0, sky->num_sources);

    // Try to set a source into the model - this should fail as the model is
    // still zero size.
    int error = 0;
    oskar_sky_model_set_source(sky, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            200.0e6, -0.7, 0.0, 0.0, 0.0, &error);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_OUT_OF_RANGE, error);
    error = 0;

    // Resize the model to 2 sources.
    oskar_sky_model_resize(sky, 2, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    CPPUNIT_ASSERT_EQUAL(2, sky->num_sources);

    // Set values of these 2 sources.
    oskar_sky_model_set_source(sky, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            200.0e6, -0.7, 0.0, 0.0, 0.0, &error);
    oskar_sky_model_set_source(sky, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
            250.0e6, -0.8, 0.0, 0.0, 0.0, &error);
    CPPUNIT_ASSERT_EQUAL(0, error);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky->location());

    // Copy back into temp. structure on the CPU to check the values were set
    // correctly.
    oskar_SkyModel sky_temp(sky, OSKAR_LOCATION_CPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky_temp.location());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky_temp.type());
    CPPUNIT_ASSERT_EQUAL(2, sky_temp.num_sources);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ((float*)sky_temp.RA.data)[0], 1.0e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0e6, ((float*)sky_temp.reference_freq.data)[0], 1.0e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.5, ((float*)sky_temp.Q.data)[1], 1.0e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.8, ((float*)sky_temp.spectral_index.data)[1], 1.0e-6);
    delete sky;
}


void Test_SkyModel::test_append()
{
    int status = 0;
    int sky1_num_sources = 2;
    oskar_SkyModel* sky1 = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_GPU,
            sky1_num_sources);
    for (int i = 0; i < sky1_num_sources; ++i)
    {
        double value = (double)i;
        oskar_sky_model_set_source(sky1, i, value, value,
                value, value, value, value,
                value, value, 0.0, 0.0, 0.0, &status);
    }
    int sky2_num_sorces = 3;
    oskar_SkyModel* sky2 = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_CPU,
            sky2_num_sorces);
    for (int i = 0; i < sky2_num_sorces; ++i)
    {
        double value = (double)i + 0.5;
        oskar_sky_model_set_source(sky2, i, value, value,
                value, value, value, value,
                value, value, 0.0, 0.0, 0.0, &status);
    }
    oskar_sky_model_append(sky1, sky2, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky1->location());
    oskar_SkyModel sky_temp(sky1, OSKAR_LOCATION_CPU);
    CPPUNIT_ASSERT_EQUAL(sky1_num_sources + sky2_num_sorces, sky_temp.num_sources);
    for (int i = 0; i < sky_temp.num_sources; ++i)
    {
        if (i < sky1_num_sources)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i, ((float*)sky_temp.RA.data)[i],
                    1.0e-6);
        }
        else
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)(i - sky1_num_sources) + 0.5, ((float*)sky_temp.RA.data)[i], 1.0e-6);
        }
    }
    delete sky2;
    delete sky1;
}


void Test_SkyModel::test_load()
{
    int err = 0;
    const double deg2rad = 0.0174532925199432957692;
    const char* filename = "temp_sources.osm";

    // Load sky file with all columns specified.
    {
        FILE* file = fopen(filename, "w");
        if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
        int num_sources = 1013;
        for (int i = 0; i < num_sources; ++i)
        {
            if (i % 10 == 0) fprintf(file, "# some comment!\n");
            fprintf(file, "%f %f %f %f %f %f %f %f\n",
                    i/10.0, i/20.0, 0.0, 1.0, 2.0, 3.0, 200.0e6, -0.7);
        }
        fclose(file);


        oskar_SkyModel* sky = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0);
        oskar_sky_model_load(sky, filename, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        // Cleanup.
        remove(filename);

        CPPUNIT_ASSERT_EQUAL_MESSAGE("oskar_SkyModel_load failed", 0, err);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(num_sources, sky->num_sources);

        // Check the data loaded correctly.
        for (int i = 0; i < num_sources; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0 * deg2rad, ((float*)sky->RA.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0 * deg2rad, ((float*)sky->Dec.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky->I.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ((float*)sky->Q.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, ((float*)sky->U.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, ((float*)sky->V.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0e6, ((float*)sky->reference_freq.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.7, ((float*)sky->spectral_index.data)[i], 1.0e-6);
        }

        delete sky;
    }


    // Load sky file with with just RA, Dec and I specified.
    {
        FILE* file = fopen(filename, "w");
        if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
        int num_sources = 1013;
        for (int i = 0; i < num_sources; ++i)
        {
            if (i % 10 == 0) fprintf(file, "# some comment!\n");
            fprintf(file, "%f, %f, %f\n", i/10.0, i/20.0, (float)i);
        }
        fclose(file);

        // Load the sky model onto the GPU.
        oskar_SkyModel* sky_gpu = new oskar_SkyModel(OSKAR_SINGLE,
                OSKAR_LOCATION_GPU, 0);
        oskar_sky_model_load(sky_gpu, filename, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        // Cleanup.
        remove(filename);

        CPPUNIT_ASSERT_EQUAL_MESSAGE("oskar_SkyModel_load failed", 0, err);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky_gpu->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky_gpu->location());
        CPPUNIT_ASSERT_EQUAL(num_sources, sky_gpu->num_sources);

        // Copy the sky model back to the CPU and free the GPU version.
        oskar_SkyModel sky_cpu(sky_gpu, OSKAR_LOCATION_CPU);
        delete sky_gpu;

        // Check the data is correct.
        for (int i = 0; i < num_sources; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0 * deg2rad, ((float*)sky_cpu.RA.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0 * deg2rad, ((float*)sky_cpu.Dec.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i, ((float*)sky_cpu.I.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.Q.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.U.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.V.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.reference_freq.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.spectral_index.data)[i], 1.0e-6);
        }
    }
}

void Test_SkyModel::test_compute_relative_lmn()
{
    const double deg2rad = 0.0174532925199432957692;

    // Create some sources.
    float ra[] = {30.0, 45.0};
    float dec[] = {50.0, 60.0};
    int n = sizeof(ra) / sizeof(float);
    for (int i = 0; i < n; ++i)
    {
        ra[i] *= deg2rad;
        dec[i] *= deg2rad;
    }

    // Define phase centre.
    float ra0 = 30.0 * deg2rad;
    float dec0 = 55.0 * deg2rad;

    // Construct a sky model on the GPU.
    oskar_SkyModel* sky = new oskar_SkyModel(OSKAR_SINGLE,
            OSKAR_LOCATION_GPU, n);
    CPPUNIT_ASSERT_EQUAL(n, sky->num_sources);

    // Set values of these sources.
    int error = 0;
    for (int i = 0; i < n; ++i)
    {
        oskar_sky_model_set_source(sky, i, ra[i], dec[i], 1.0, 2.0, 3.0, 4.0,
                200.0e6, -0.7, 0.0, 0.0, 0.0, &error);
    }
    CPPUNIT_ASSERT_EQUAL(0, error);

    // Compute l,m direction cosines.
    oskar_sky_model_compute_relative_lmn(sky, ra0, dec0, &error);

    // Copy data back to CPU.
    oskar_SkyModel sky_temp(sky, OSKAR_LOCATION_CPU);
    delete sky;

    // Check the data.
    for (int i = 0; i < n; ++i)
    {
        float l = sin(ra[i] - ra0) * cos(dec[i]);
        float m = cos(dec0) * sin(dec[i]) -
                sin(dec0) * cos(dec[i]) * cos(ra[i] - ra0);
        float p = sqrt(1.0 - l*l - m*m) - 1.0;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(l, ((float*)sky_temp.rel_l.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(m, ((float*)sky_temp.rel_m.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(p, ((float*)sky_temp.rel_n.data)[i], 1e-3);
    }
}

void Test_SkyModel::test_horizon_clip()
{
    // Create a sky model.
    int err = 0;
    oskar_SkyModel sky_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0);
    oskar_SkyModel sky_out(OSKAR_SINGLE, OSKAR_LOCATION_GPU, 0);

    // Constants.
    const double deg2rad = M_PI / 180.0;

    // Sky grid parameters.
    int n_lat = 256; // 8
    int n_lon = 256; // 12
    int n_sources = n_lat * n_lon;
    double lat_start = -90.0;
    double lon_start = 0.0;
    double lat_end = 90.0;
    double lon_end = 330.0;
    oskar_sky_model_resize(&sky_cpu, n_sources, &err);
    CPPUNIT_ASSERT_EQUAL(0, err);

    // Generate grid.
    for (int i = 0, k = 0; i < n_lat; ++i)
    {
        for (int j = 0; j < n_lon; ++j, ++k)
        {
            double ra = lon_start + j * (lon_end - lon_start) / (n_lon - 1);
            double dec = lat_start + i * (lat_end - lat_start) / (n_lat - 1);
            oskar_sky_model_set_source(&sky_cpu, k, ra * deg2rad, dec * deg2rad,
                    double(k), double(2*k), double(3*k), double(4*k),
                    double(5*k), double(6*k), 0.0, 0.0, 0.0, &err);
            CPPUNIT_ASSERT_EQUAL(0, err);
        }
    }

    // Create a telescope model.
    int n_stations = 25;
    oskar_TelescopeModel telescope(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
    oskar_telescope_model_resize(&telescope, n_stations, &err);
    CPPUNIT_ASSERT_EQUAL(0, err);

    // Create a work buffer.
    oskar_WorkStationBeam work(OSKAR_SINGLE, OSKAR_LOCATION_GPU);

    // Try calling compact: should fail.
    oskar_sky_model_horizon_clip(&sky_out, &sky_cpu, &telescope, 0.0, &work, &err);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_BAD_LOCATION, err);
    err = 0;

    {
        // Copy sky data to GPU.
        oskar_SkyModel* sky_gpu = new oskar_SkyModel(&sky_cpu, OSKAR_LOCATION_GPU);

        // Compact sky data at pole.
        for (int i = 0; i < n_stations; ++i)
        {
            telescope.station[i].latitude_rad = (90.0 - i * 0.01) * deg2rad;
            telescope.station[i].longitude_rad = 0.0;
        }

        TIMER_START
        oskar_sky_model_horizon_clip(&sky_out, sky_gpu, &telescope, 0.0, &work, &err);
        TIMER_STOP("Done sky model compaction (%d sources)", n_sources)
        CPPUNIT_ASSERT_EQUAL(0, err);
        CPPUNIT_ASSERT_EQUAL(n_sources / 2, sky_out.num_sources);

        // Check sky data.
        oskar_SkyModel* sky_temp = new oskar_SkyModel(&sky_out, OSKAR_LOCATION_CPU);
        CPPUNIT_ASSERT_EQUAL(n_sources / 2, sky_temp->num_sources);
        for (int i = 0, n = sky_temp->num_sources; i < n; ++i)
        {
            CPPUNIT_ASSERT(((float*)(sky_temp->Dec))[i] > 0.0f);
        }

        delete sky_temp;
        delete sky_gpu;
    }
}

void Test_SkyModel::test_split()
{
    int num_sources = 2139;
    int max_sources_per_subset = 510;
    int error = 0;

    oskar_SkyModel sky_full(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_sources);
    oskar_SkyModel* sky_subset = NULL;

    // Split the sky model into a number of subsets.
    int num_subsets = 0;
    error = oskar_sky_model_split(&sky_subset, &num_subsets,
            max_sources_per_subset, &sky_full);

    // Check if the split worked as expected.
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    CPPUNIT_ASSERT_EQUAL((int)ceil((double)num_sources/max_sources_per_subset),
            num_subsets);

    for (int i = 0; i < num_subsets; ++i)
    {
        CPPUNIT_ASSERT(sky_subset[i].num_sources <= max_sources_per_subset);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky_subset[i].type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky_subset[i].location());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].RA.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].Dec.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].I.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].Q.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].U.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].V.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].reference_freq.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].spectral_index.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].rel_l.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].rel_m.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].rel_n.num_elements);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].RA.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].Dec.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].I.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].Q.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].U.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].V.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].reference_freq.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].spectral_index.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].rel_l.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].rel_m.owner);
        CPPUNIT_ASSERT_EQUAL(0, sky_subset[i].rel_n.owner);
    }

    // Copy subsets to the GPU.
    oskar_SkyModel* sky_subset_gpu = NULL;
    sky_subset_gpu = (oskar_SkyModel*)malloc(num_subsets * sizeof(oskar_SkyModel));
    for (int i = 0; i < num_subsets; ++i)
    {
        oskar_sky_model_init(&sky_subset_gpu[i], sky_subset[i].type(),
                OSKAR_LOCATION_GPU, 0, &error);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        oskar_sky_model_copy(&sky_subset_gpu[i], &sky_subset[i], &error);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    }

    for (int i = 0; i < num_subsets; ++i)
    {
        CPPUNIT_ASSERT(sky_subset_gpu[i].num_sources <= max_sources_per_subset);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky_subset_gpu[i].type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky_subset_gpu[i].location());
        CPPUNIT_ASSERT_EQUAL(sky_subset_gpu[i].num_sources, sky_subset_gpu[i].RA.num_elements);
        CPPUNIT_ASSERT_EQUAL(sky_subset_gpu[i].num_sources, sky_subset_gpu[i].rel_m.num_elements);
        CPPUNIT_ASSERT_EQUAL(1, sky_subset_gpu[i].RA.owner);
        CPPUNIT_ASSERT_EQUAL(1, sky_subset_gpu[i].Q.owner);
        CPPUNIT_ASSERT_EQUAL(1, sky_subset_gpu[i].rel_m.owner);
    }

    // Cleanup.
    free(sky_subset_gpu);
    free(sky_subset);
}


void Test_SkyModel::test_filter_by_radius()
{
    // Generate 91 sources from dec = 0 to dec = 90 degrees.
    int err = 0;
    int num_sources = 91;
    oskar_SkyModel sky(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_sources);
    for (int i = 0; i < num_sources; ++i)
    {
        oskar_sky_model_set_source(&sky, i, 0.0, i * ((M_PI / 2) / (num_sources - 1)),
                (double)i, 1.0, 2.0, 3.0, i * 100.0, i * 200.0,
                0.0, 0.0, 0.0, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);
    }

    // Check that the data was set correctly.
    CPPUNIT_ASSERT_EQUAL(num_sources, sky.num_sources);
    for (int i = 0; i < num_sources; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((const double*)sky.RA)[i], 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i * ((M_PI / 2) / (num_sources - 1)),
                ((const double*)sky.Dec)[i], 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i,
                ((const double*)sky.I)[i], 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ((const double*)sky.Q)[i], 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, ((const double*)sky.U)[i], 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, ((const double*)sky.V)[i], 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 100,
                ((const double*)sky.reference_freq)[i], 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 200,
                ((const double*)sky.spectral_index)[i], 0.001);
    }

    // Filter the data.
    double inner = 4.5 * M_PI / 180;
    double outer = 10.5 * M_PI / 180;
    double ra0 = 0.0;
    double dec0 = M_PI / 2;
    err = oskar_sky_model_filter_by_radius(&sky, inner, outer, ra0, dec0);
    CPPUNIT_ASSERT_EQUAL(0, err);

    // Check the resulting sky model.
    CPPUNIT_ASSERT_EQUAL(6, sky.num_sources);
    for (int i = 0; i < sky.num_sources; ++i)
    {
        CPPUNIT_ASSERT(((const double*)sky.Dec)[i] > 79.5 * M_PI / 180.0);
        CPPUNIT_ASSERT(((const double*)sky.Dec)[i] < 85.5 * M_PI / 180.0);
//        printf("RA = %f, Dec = %f\n", ((const double*)sky.RA)[i],
//                ((const double*)sky.Dec)[i] * 180 / M_PI);
    }
}

void Test_SkyModel::test_gaussian_source()
{
    double ra0  = 0.0  * M_PI/180;
    double dec0 = 90.0 * M_PI/180;

    double ra       = 0.0  * (M_PI / 180.0);
    double dec      = 70.0 * (M_PI / 180.0);
    double fwhm_maj = 1.0  * (M_PI / 180.0);
    double fwhm_min = 1.0  * (M_PI / 180.0);
    double pa       = 0.0 * (M_PI / 180.0);

    double delta_ra_maj, delta_dec_maj, delta_ra_min, delta_dec_min;
    double lon[4], lat[4];

    delta_ra_maj  = (fwhm_maj / 2.0) * sin(pa);
    delta_dec_maj = (fwhm_maj / 2.0) * cos(pa);

    delta_ra_min  = (fwhm_min / 2.0) * cos(pa);
    delta_dec_min = (fwhm_min / 2.0) * sin(pa);

    lon[0] = ra - delta_ra_maj;
    lon[1] = ra + delta_ra_maj;
    lon[2] = ra - delta_ra_min;
    lon[3] = ra + delta_ra_min;

    lat[0] = dec - delta_dec_maj;
    lat[1] = dec + delta_dec_maj;
    lat[2] = dec - delta_dec_min;
    lat[3] = dec + delta_dec_min;

    double l[4], m[4];

    oskar_sph_to_lm_d(4, ra0, dec0, lon, lat, l, m);

    printf("\n");
    printf("ra0, dec0              = %f, %f\n", ra0*(180.0/M_PI), dec0*(180.0/M_PI));
    printf("ra, dec                = %f, %f\n", ra*180/M_PI, dec*180/M_PI);
    printf("fwhm_maj, fwhm_min, pa = %f, %f, %f\n", fwhm_maj*180/M_PI,
            fwhm_min*180/M_PI, pa*180/M_PI);
    printf("delta ra (maj, min)    = %f, %f\n",
            delta_ra_maj*180/M_PI, delta_ra_min*180/M_PI);
    printf("delta dec (maj, min)   = %f, %f\n",
            delta_dec_maj*180/M_PI, delta_dec_min*180/M_PI);
    printf("\n");


    double x_maj = l[1] - l[0];
    double y_maj = m[1] - m[0];
    double pa_lm_maj = M_PI/2.0 - atan2(y_maj, x_maj);
    double fwhm_lm_maj = sqrt(pow(fabs(x_maj), 2.0) + pow(fabs(y_maj), 2.0));

    double x_min = l[3] - l[2];
    double y_min = m[3] - m[2];
    double pa_lm_min = M_PI/2.0 - atan2(y_min, x_min);
    double fwhm_lm_min = sqrt(pow(fabs(x_min), 2.0) + pow(fabs(y_min), 2.0));


    printf("= major axis:\n");
    printf("    lon, lat = %f->%f, %f->%f\n",
            lon[0]*(180/M_PI), lon[1]*(180/M_PI),
            lat[0]*(180/M_PI), lat[1]*(180/M_PI));
    printf("    l,m      = %f->%f, %f->%f\n", l[0], l[1], m[0], m[1]);
    printf("    x,y      = %f, %f\n", x_maj, y_maj);
    printf("    pa_lm    = %f\n", pa_lm_maj * (180.0/M_PI));
    printf("    fwhm     = %f\n", asin(fwhm_lm_maj)*180/M_PI);

    printf("= minor axis:\n");
    printf("    lon, lat = %f->%f, %f->%f\n",
            lon[2]*(180/M_PI), lon[3]*(180/M_PI),
            lat[2]*(180/M_PI), lat[3]*(180/M_PI));
    printf("    l,m      = %f->%f, %f->%f\n", l[2], l[3], m[2], m[3]);
    printf("    x,y      = %f, %f\n", x_min, y_min);
    printf("    pa_lm    = %f\n", pa_lm_min * (180.0/M_PI));
    printf("    fwhm     = %f\n", asin(fwhm_lm_min)*180/M_PI);
}

void Test_SkyModel::test_evaluate_gaussian_source_parameters()
{
    const double asec2rad = M_PI / (180.0 * 3600.0);
    const double deg2rad  = M_PI / 180.0;

    int num_sources = 1;
    int status = 0;
    int zero_failed_sources = OSKAR_FALSE;

    oskar_SkyModel sky(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_sources);
    oskar_sky_model_set_source(&sky, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            20 * 60 * asec2rad,
            10 * 60 * asec2rad,
            30 * deg2rad, &status);
    oskar_evaluate_gaussian_source_parameters(NULL, num_sources, &sky.gaussian_a,
            &sky.gaussian_b, &sky.gaussian_c, &sky.FWHM_major, &sky.FWHM_minor,
            &sky.position_angle, &sky.RA, &sky.Dec, zero_failed_sources,
            &sky.I, 0, 40.0 * M_PI/180.0);
//    sky.write("temp_sky_gaussian.osm");

//    printf("\n");
//    for (int i = 0; i < num_sources; ++i)
//    {
//        printf("[%i] a = %e, b = %e, c = %e\n", i,
//                ((double*)sky.gaussian_a.data)[i],
//                ((double*)sky.gaussian_b.data)[i],
//                ((double*)sky.gaussian_c.data)[i]);
//    }
}


void Test_SkyModel::test_insert()
{
    int type     = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;
    int dst_size = 60;
    int src_size = 20;
    int status = 0;

    oskar_SkyModel dst(type, location, dst_size);

    oskar_SkyModel src(type, location, src_size);
    for (int i = 0; i < src_size; ++i)
    {
        ((double*)src.RA.data)[i]             = (double)i + 0.0;
        ((double*)src.Dec.data)[i]            = (double)i + 0.1;
        ((double*)src.I.data)[i]              = (double)i + 0.2;
        ((double*)src.Q.data)[i]              = (double)i + 0.3;
        ((double*)src.U.data)[i]              = (double)i + 0.4;
        ((double*)src.V.data)[i]              = (double)i + 0.5;
        ((double*)src.reference_freq.data)[i] = (double)i + 0.6;
        ((double*)src.spectral_index.data)[i] = (double)i + 0.7;
        ((double*)src.rel_l.data)[i]          = (double)i + 0.8;
        ((double*)src.rel_m.data)[i]          = (double)i * 2.0;
        ((double*)src.rel_n.data)[i]          = (double)i * 3.0;
        ((double*)src.FWHM_major.data)[i]     = (double)i * 4.0;
        ((double*)src.FWHM_minor.data)[i]     = (double)i * 5.0;
        ((double*)src.position_angle.data)[i] = (double)i * 6.0;
        ((double*)src.gaussian_a.data)[i]     = (double)i * 7.0;
        ((double*)src.gaussian_b.data)[i]     = (double)i * 8.0;
        ((double*)src.gaussian_c.data)[i]     = (double)i * 9.0;
    }

    oskar_sky_model_insert(&dst, &src, 0, &status);
    oskar_sky_model_insert(&dst, &src, 20, &status);
    oskar_sky_model_insert(&dst, &src, 40, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    double delta = 1.0e-10;

    for (int j = 0; j < 3; ++j)
    {
        for (int i = 0; i < src_size; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.0, ((double*)src.RA.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.1, ((double*)src.Dec.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.2, ((double*)src.I.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.3, ((double*)src.Q.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.4, ((double*)src.U.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.5, ((double*)src.V.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.6, ((double*)src.reference_freq.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.7, ((double*)src.spectral_index.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i + 0.8, ((double*)src.rel_l.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 2.0, ((double*)src.rel_m.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 3.0, ((double*)src.rel_n.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 4.0, ((double*)src.FWHM_major.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 5.0, ((double*)src.FWHM_minor.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 6.0, ((double*)src.position_angle.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 7.0, ((double*)src.gaussian_a.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 8.0, ((double*)src.gaussian_b.data)[i], delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i * 9.0, ((double*)src.gaussian_c.data)[i], delta);
        }
    }
}


void Test_SkyModel::test_sky_model_set()
{
    int number = 0;
    int error = OSKAR_SUCCESS;
    oskar_SkyModel* set = NULL;

    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;
    int max = 5;

    int model_size = 6;
    oskar_SkyModel model1(type, location, model_size);
    for (int i = 0; i < model_size; ++i)
    {
        ((double*)model1.RA.data)[i] = (double)i;
    }
    error = oskar_sky_model_append_to_set(&number, &set, max, &model1);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    model_size = 7;
    oskar_SkyModel model2(type, location, model_size);
    for (int i = 0; i < model_size; ++i)
    {
        ((double*)model2.RA.data)[i]         = (double)i + 0.5;
        ((double*)model2.FWHM_major.data)[i] = (double)i * 0.75;
    }
    error = oskar_sky_model_append_to_set(&number, &set, max, &model2);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

//    printf("\n");
//    printf("==========================\n");
//    for (int i = 0; i < number; ++i)
//    {
//        printf("++ set[%i] no. sources = %i, use extended = %s\n",
//                i, set[i].num_sources, set[i].use_extended ? "true" : "false");
//        for (int s = 0; s < set[i].num_sources; ++s)
//        {
//            printf("  RA = %f, FWHM_major = %f\n",
//                    ((double*)set[i].RA.data)[s],
//                    ((double*)set[i].FWHM_major.data)[s]);
//        }
//    }
//    printf("==========================\n");

    // Free the array of sky models.
    if (set)
    {
        for (int i = 0; i < number; ++i)
        {
            oskar_sky_model_free(&set[i], &error);
        }
        free(set);
    }
}

