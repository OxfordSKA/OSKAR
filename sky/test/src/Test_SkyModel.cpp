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

#include "interferometry/oskar_TelescopeModel.h"
#include "sky/test/Test_SkyModel.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_filter_by_flux.h"
#include "sky/oskar_sky_model_filter_by_radius.h"
#include "sky/oskar_sky_model_horizon_clip.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_load.h"
#include "sky/oskar_sky_model_split.h"
#include "sky/oskar_evaluate_sky_temperature.h"
#include "sky/oskar_evaluate_gaussian_source_parameters.h"
#include "sky/oskar_sky_model_append_to_set.h"
#include "sky/oskar_sky_model_insert.h"
#include "utility/oskar_Work.h"
#include "utility/oskar_get_error_string.h"

//#define TIMER_ENABLE
#include "utility/timer.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

void Test_SkyModel::test_resize()
{
    // Resizing on the GPU in single precision
    {
        oskar_SkyModel* sky = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_GPU, 10);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(10, sky->num_sources);
        sky->resize(1);
        CPPUNIT_ASSERT_EQUAL(1, sky->num_sources);
        sky->resize(20);
        CPPUNIT_ASSERT_EQUAL(20, sky->num_sources);
        delete sky;
    }
    // Resizing on the CPU in double precision
    {
        oskar_SkyModel* sky = new oskar_SkyModel(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 10);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(10, sky->num_sources);
        sky->resize(1);
        CPPUNIT_ASSERT_EQUAL(1, sky->num_sources);
        sky->resize(20);
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
    int error = sky->set_source(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 200.0e6, -0.7);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_OUT_OF_RANGE, error);

    // Resize the model to 2 sources.
    sky->resize(2);
    CPPUNIT_ASSERT_EQUAL(2, sky->num_sources);

    // Set values of these 2 sources.
    error = sky->set_source(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 200.0e6, -0.7);
    error = sky->set_source(1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 250.0e6, -0.8);
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
    int sky1_num_sources = 2;
    oskar_SkyModel* sky1 = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_GPU,
            sky1_num_sources);
    for (int i = 0; i < sky1_num_sources; ++i)
    {
        double value = (double)i;
        sky1->set_source(i, value, value, value, value, value, value, value, value);
    }
    int sky2_num_sorces = 3;
    oskar_SkyModel* sky2 = new oskar_SkyModel(OSKAR_SINGLE, OSKAR_LOCATION_CPU,
            sky2_num_sorces);
    for (int i = 0; i < sky2_num_sorces; ++i)
    {
        double value = (double)i + 0.5;
        sky2->set_source(i, value, value, value, value, value, value, value, value);
    }
    sky1->append(sky2);
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
        int err = oskar_sky_model_load(sky, filename);
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
        int err = oskar_sky_model_load(sky_gpu, filename);
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
        error = sky->set_source(i, ra[i], dec[i], 1.0, 2.0, 3.0, 4.0,
                200.0e6, -0.7);
    }
    CPPUNIT_ASSERT_EQUAL(0, error);

    // Compute l,m direction cosines.
    sky->compute_relative_lmn(ra0, dec0);

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
    sky_cpu.resize(n_sources);

    // Generate grid.
    for (int i = 0, k = 0; i < n_lat; ++i)
    {
        for (int j = 0; j < n_lon; ++j, ++k)
        {
            double ra = lon_start + j * (lon_end - lon_start) / (n_lon - 1);
            double dec = lat_start + i * (lat_end - lat_start) / (n_lat - 1);
            sky_cpu.set_source(k, ra * deg2rad, dec * deg2rad,
                    double(k), double(2*k), double(3*k), double(4*k),
                    double(5*k), double(6*k));
        }
    }

    // Create a telescope model.
    int n_stations = 25;
    oskar_TelescopeModel telescope(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
    err = telescope.resize(n_stations);
    CPPUNIT_ASSERT_EQUAL(0, err);

    // Create a work buffer.
    oskar_Work work(OSKAR_SINGLE, OSKAR_LOCATION_GPU);

    // Try calling compact: should fail.
    err = oskar_sky_model_horizon_clip(&sky_out, &sky_cpu, &telescope, 0.0, &work);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_BAD_LOCATION, err);

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
        err = oskar_sky_model_horizon_clip(&sky_out, sky_gpu, &telescope, 0.0, &work);
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

    oskar_SkyModel sky_full(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_sources);
    oskar_SkyModel* sky_subset = NULL;

    // Split the sky model into a number of subsets.
    int num_subsets = 0;
    int error = oskar_sky_model_split(&sky_subset, &num_subsets,
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
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].RA.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].Dec.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].I.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].Q.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].U.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].V.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].reference_freq.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].spectral_index.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].rel_l.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].rel_m.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset[i].num_sources, sky_subset[i].rel_n.num_elements());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].RA.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].Dec.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].I.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].Q.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].U.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].V.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].reference_freq.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].spectral_index.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].rel_l.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].rel_m.owner());
        CPPUNIT_ASSERT_EQUAL(false, sky_subset[i].rel_n.owner());
    }

    // Copy subsets to the GPU.
    // NOTE: this interface is not exactly ideal as it requires the user to use
    // the low level function oskar_sky_model_init().
    oskar_SkyModel* sky_subset_gpu = NULL;
    sky_subset_gpu = (oskar_SkyModel*)malloc(num_subsets * sizeof(oskar_SkyModel));
    for (int i = 0; i < num_subsets; ++i)
    {
        error = oskar_sky_model_init(&sky_subset_gpu[i], sky_subset[i].type(),
                OSKAR_LOCATION_GPU, 0);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        error = sky_subset[i].copy_to(&sky_subset_gpu[i]);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    }

    for (int i = 0; i < num_subsets; ++i)
    {
        CPPUNIT_ASSERT(sky_subset_gpu[i].num_sources <= max_sources_per_subset);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky_subset_gpu[i].type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky_subset_gpu[i].location());
        CPPUNIT_ASSERT_EQUAL(sky_subset_gpu[i].num_sources, sky_subset_gpu[i].RA.num_elements());
        CPPUNIT_ASSERT_EQUAL(sky_subset_gpu[i].num_sources, sky_subset_gpu[i].rel_m.num_elements());
        CPPUNIT_ASSERT_EQUAL(true, sky_subset_gpu[i].RA.owner());
        CPPUNIT_ASSERT_EQUAL(true, sky_subset_gpu[i].Q.owner());
        CPPUNIT_ASSERT_EQUAL(true, sky_subset_gpu[i].rel_m.owner());
    }

    // Cleanup.
    free(sky_subset_gpu);
    free(sky_subset);
}

void Test_SkyModel::test_evaluate_sky_temperature()
{
    int num_channels = 30;
    double start_freq = 10e6; // Hz
    double freq_inc   = 5e6;  // Hz
    double spectral_index = 0.75;
    double* temp = (double*)malloc(num_channels * sizeof(double));

    oskar_evaluate_sky_temperature(temp, num_channels, start_freq, freq_inc,
            spectral_index);

    free(temp);
}

void Test_SkyModel::test_filter_by_radius()
{
    // Generate 91 sources from dec = 0 to dec = 90 degrees.
    int err = 0;
    int num_sources = 91;
    oskar_SkyModel sky(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_sources);
    for (int i = 0; i < num_sources; ++i)
    {
        err = sky.set_source(i, 0.0, i * ((M_PI / 2) / (num_sources - 1)),
                i, 1.0, 2.0, 3.0, i * 100, i * 200);
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


void Test_SkyModel::test_evaluate_gaussian_source_parameters()
{
    const double asec2rad = M_PI / (180.0 * 3600.0);
    const double deg2rad  = M_PI / 180.0;

    int num_sources = 1;
    oskar_SkyModel sky(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_sources);
    sky.set_source(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            20 * 60 * asec2rad,
            10 * 60 * asec2rad,
            30 * deg2rad);
    oskar_evaluate_gaussian_source_parameters(num_sources, &sky.gaussian_a,
            &sky.gaussian_b, &sky.gaussian_c, &sky.FWHM_major, &sky.FWHM_minor,
            &sky.position_angle);
    //sky.write("temp_sky_gaussian.osm");

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

    oskar_sky_model_insert(&dst, &src, 0);
    oskar_sky_model_insert(&dst, &src, 20);
    oskar_sky_model_insert(&dst, &src, 40);

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


    if (set) free(set);
}

