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

#include "apps/lib/oskar_Settings.h"
#include "apps/lib/oskar_load_telescope.h"
#include "apps/lib/oskar_imager_dft.cu.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_VisData.h"
#include "utility/oskar_vector_types.h"

#include "widgets/plotting/oskar_PlotWidget.h"
#include "widgets/plotting/oskar_ImagePlot.h"

#include <QtGui/QApplication>
#include <QtCore/QTime>

#include "qwt_scale_widget.h"

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace oskar;

int imager_d(const oskar_Settings& settings);
int imager_f(const oskar_Settings& settings);

int main(int argc, char** argv)
{
    // $> oskar_sim1_scalar settings_file.txt
    if (argc != 2)
    {
        fprintf(stderr, "ERROR: missing command line arguments\n");
        fprintf(stderr, "Usage:  $ oskar_imager_scalar [settings file]\n");
        return EXIT_FAILURE;
    }

    // Load settings file.
    oskar_Settings settings;
    settings.load(QString(argv[1]));


    // Double precision.
    if (settings.double_precision())
    {
        imager_d(settings);
    }
    // Single precision.
    else
    {
        imager_f(settings);
    }




    // =========== PLOTTING ====================================================
    // FIXME images not in memory in this version....
//    std::vector<float> image_f(image_size * image_size);
//    for (unsigned i = 0; i < image_size * image_size; ++i)
//        image_f[i] = (float)image[i];
//    // plotting.
//    QApplication app(argc, argv);
//    PlotWidget plot1;
//    plot1.plotImage(&image_f[0], image_size, image_size, -fov_deg/2, fov_deg/2,
//            -fov_deg/2, fov_deg/2);
//    plot1.resize(700, 600);
//    plot1.setTitle("Dirty image");
//    plot1.setXLabel("Relative Right Ascension (deg)");
//    plot1.setYLabel("Relative Declination (deg)");
//    plot1.setScaleLabel("Jy/beam");
//    PlotWidget plot2;
//    plot2.plotCurve(vis.num_samples, vis.u, vis.v);
//    int status = app.exec();
    // =========== END PLOTTING ================================================

    int status = 0;
    return status;
}





int imager_d(const oskar_Settings& settings)
{
    const double fov_deg      = settings.image().fov_deg();
    const unsigned image_size = settings.image().size();

    oskar_TelescopeModel_d telescope;
    oskar_load_telescope_d(settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(), &telescope);

    // Image coordinates.
    std::vector<double> l(image_size);
    double lmax = sin((fov_deg / 2) * M_PI / 180.0);
    double inc  = (2.0 * lmax) / (image_size - 1);
    for (unsigned i = 0; i < image_size; ++i)
        l[i] = -lmax + i * inc;


    int num_channels  = settings.obs().num_channels();
    int num_baselines = telescope.num_antennas * (telescope.num_antennas - 1) / 2;
    int num_dumps_per_snapshot = settings.image().make_snapshots() ?
            settings.image().dumps_per_snapshot() : settings.obs().num_vis_dumps();
    int num_snapshots = (int)settings.obs().num_vis_dumps() / num_dumps_per_snapshot;
    if ((int)settings.obs().num_vis_dumps() % num_dumps_per_snapshot != 0)
        fprintf(stderr, "ERROR: eek!\n");

    // Array of peak amplitudes.
    vector<double> peak_amp(num_channels * num_snapshots);

    // Loop over freqs. and make images.
    for (int i = 0; i < num_channels; ++i)
    {
        double frequency = settings.obs().frequency(i);
        printf("== imaging simulation of freq %e\n", frequency);

        // Load data file for the frequency.
        QString vis_file = settings.obs().oskar_vis_filename()
                + "_channel_" + QString::number(i) + ".dat";
        oskar_VisData_d vis;
        oskar_load_vis_data_d(vis_file.toLatin1().data(), &vis);
        printf("== num vis = %i\n", vis.num_samples);

        for (int t = 0; t < num_snapshots; ++t)
        {
            unsigned num_samples = num_baselines * num_dumps_per_snapshot;
            double2* amp = &vis.amp[t * num_samples];
            double* u    = &vis.u[t * num_samples];
            double* v    = &vis.v[t * num_samples];

            std::vector<double> image(image_size * image_size, 0.0);
            int error = oskar_imager_dft_d(num_samples, amp, u, v,
                    frequency, image_size, &l[0], &image[0]);

            if (error != 0)
            {
                fprintf(stderr, "ERROR: CUDA DFT imager failed, error = %i\n", error);
                return EXIT_FAILURE;
            }

            // FIXME: ******** Pixel of peak *************
            int idx = (192) * image_size + (128-1);
            peak_amp[i * num_snapshots + t] = image[idx];
            image[idx] = 0.0;
            // FIXME: ******** Pixel of peak *************

            // Write out image file.
            QString image_file = settings.image().filename() +
                    "_channel_" + QString::number(i) +
                    "_t_" + QString::number(t) + ".dat";
            FILE* file;
            file = fopen(image_file.toLatin1().data(), "wb");
            if (file == NULL)
            {
                fprintf(stderr, "ERROR: Failed to open output image file.\n");
                return EXIT_FAILURE;
            }
            fwrite(&image[0], sizeof(double), image_size * image_size, file);
            fclose(file);
        } // end loop over snapshots.

        oskar_free_vis_data_d(&vis);

    } // end loop over frequency.


    FILE* file;
    QString peaks_file = settings.image().filename() + "_peaks.dat";
    file = fopen(peaks_file.toLatin1().data(), "w");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open amps file.\n");
        return EXIT_FAILURE;
    }
    fwrite(&peak_amp[0], sizeof(double), num_channels * num_snapshots, file);
    fclose(file);

    return EXIT_SUCCESS;
}




int imager_f(const oskar_Settings& settings)
{
    const float fov_deg       = settings.image().fov_deg();
    const unsigned image_size = settings.image().size();

    oskar_TelescopeModel_f telescope;
    oskar_load_telescope_f(settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(), &telescope);

    // Image coordinates.
    std::vector<float> l(image_size);
    float lmax = sin((fov_deg / 2.0f) * M_PI / 180.0f);
    float inc  = (2.0f * lmax) / (image_size - 1);
    for (unsigned i = 0; i < image_size; ++i)
        l[i] = -lmax + i * inc;

    int num_channels  = settings.obs().num_channels();
    int num_baselines = telescope.num_antennas * (telescope.num_antennas - 1) / 2;
    int num_dumps_per_snapshot = settings.image().make_snapshots() ?
            settings.image().dumps_per_snapshot() : settings.obs().num_vis_dumps();
    int num_snapshots = (int)settings.obs().num_vis_dumps() / num_dumps_per_snapshot;

    printf("creating %i image snapshots\n", num_snapshots);

    if ((int)settings.obs().num_vis_dumps() % num_dumps_per_snapshot != 0)
        fprintf(stderr, "ERROR: eek!\n");

    // Array of peak amplitudes.
    vector<float> peak_amp(num_channels * num_snapshots);

    // Loop over freqs. and make images.
    for (int i = 0; i < num_channels; ++i)
    {
        float frequency = settings.obs().frequency(i);
        printf("== imaging simulation of freq %e\n", frequency);

        // Load data file for the frequency.
        QString vis_file = settings.obs().oskar_vis_filename()
                + "_channel_" + QString::number(i) + ".dat";
        oskar_VisData_f vis;
        oskar_load_vis_data_f(vis_file.toLatin1().data(), &vis);
        if (vis.num_samples < 1)
        {
            fprintf(stderr, "ERROR: no visibility data points found in data file: %s.\n",
                    vis_file.toLatin1().data());
            continue;
        }
        printf("== num vis = %i\n", vis.num_samples);

        for (int t = 0; t < num_snapshots; ++t)
        {
            unsigned num_samples = num_baselines * num_dumps_per_snapshot;
            float2* amp = &vis.amp[t * num_samples];
            float* u    = &vis.u[t * num_samples];
            float* v    = &vis.v[t * num_samples];

            std::vector<float> image(image_size * image_size, 0.0);
            int error = oskar_imager_dft_f(num_samples, amp, u, v,
                    frequency, image_size, &l[0], &image[0]);

            if (error != 0)
            {
                fprintf(stderr, "ERROR: CUDA DFT imager failed, error = %i\n", error);
                return EXIT_FAILURE;
            }

            // FIXME: ******** Pixel of peak *************
            int idx = (192) * image_size + (128-1);
            peak_amp[i * num_snapshots + t] = image[idx];
            image[idx] = 0.0;
            // FIXME: ******** Pixel of peak *************

            // Write out image file.
            QString image_file = settings.image().filename() +
                    "_channel_" + QString::number(i) +
                    "_t_" + QString::number(t) + ".dat";
            FILE* file;
            file = fopen(image_file.toLatin1().data(), "wb");
            if (file == NULL)
            {
                fprintf(stderr, "ERROR: Failed to open output image file.\n");
                return EXIT_FAILURE;
            }
            fwrite(&image[0], sizeof(float), image_size * image_size, file);
            fclose(file);
        } // end loop over snapshots.

        oskar_free_vis_data_f(&vis);

    } // end loop over frequency.


    FILE* file;
    QString peaks_file = settings.image().filename() + "_peaks.dat";
    file = fopen(peaks_file.toLatin1().data(), "w");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open amps file.\n");
        return EXIT_FAILURE;
    }
    fwrite(&peak_amp[0], sizeof(float), num_channels * num_snapshots, file);
    fclose(file);

    return EXIT_SUCCESS;
}
