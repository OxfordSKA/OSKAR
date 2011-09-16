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

int main(int argc, char** argv)
{
    // $> oskar_sim1_scalar settings_file.txt
    if (argc != 4)
    {
        fprintf(stderr, "ERROR: missing command line arguments\n");
        fprintf(stderr, "Usage:  $ imager_scalar [settings file] [field of view (deg)] [num pixels]\n");
        return EXIT_FAILURE;
    }

    // Load settings file.
    oskar_Settings settings;
    settings.load(QString(argv[1]));

    double fov_deg      = settings.fov_deg();
    unsigned image_size = settings.image_size();
    double lmax         = sin((fov_deg / 2) * M_PI / 180.0);
    double inc          = (2.0 * lmax) / (image_size - 1);

    int error = 0;

    // Double precision.
    if (settings.double_precision())
    {
        oskar_TelescopeModel_d telescope;
        oskar_load_telescope_d(settings.telescope_file().toLatin1().data(),
                settings.longitude_rad(), settings.latitude_rad(), &telescope);

        // Load data file.
        oskar_VisData_d vis;
        oskar_load_vis_data_d(settings.output_file().toLatin1().data(), &vis);

        printf("- num vis samples     = %i\n", vis.num_samples);
        printf("- field of view (deg) = %f\n", fov_deg);
        printf("- image size          = %i\n", image_size);
        printf("- cellsize (arcsec)   = %f\n", asin(inc) * (180.0 / M_PI) * 3600.0);
        printf("\n");

        QTime timer;
        timer.start();
        if (settings.image_snapshots())
        {
            int num_baselines = telescope.num_antennas * (telescope.num_antennas - 1) / 2;
            int num_dumps_per_snapshot = 2;
            int num_snapshots = (int)settings.num_vis_dumps() / num_dumps_per_snapshot;


            for (int t = 0; t < num_snapshots; ++t)
            {
                unsigned num_samples = num_baselines * num_dumps_per_snapshot;
                double2* amp = &vis.amp[t * num_samples];
                double* u    = &vis.u[t * num_samples];
                double* v    = &vis.v[t * num_samples];

                std::vector<double> image(image_size * image_size, 0.0);
                error = oskar_imager_dft_d(num_samples, amp, u, v, frequency,
                        image_size, &l[0], &image[0]);

                if (error != 0)
                {
                    fprintf(stderr, "ERROR: CUDA dft imager failed with error = %i\n", err);
                    return EXIT_FAILURE;
                }

                int idx = (192) * image_size + (128-1);
                peak_amp[i * num_snapshots + t] = image[idx];
                image[idx] = 0.0;

                // Write out image file.
                QString outfile = "image_t" + QString::number(t) + "_f" + QString::number(i) + ".dat";
                FILE* file;
                file = fopen(outfile.toLatin1().data(), "wb");
                if (file == NULL)
                {
                    fprintf(stderr, "ERROR: Failed to open output image file.\n");
                    return EXIT_FAILURE;
                }
                fwrite(&image[0], sizeof(double), image_size * image_size, file);
                fclose(file);
            }
         }
        else
        {
            std::vector<double> l(image_size);
            for (unsigned i = 0; i < image_size; ++i)
                l[i] = -lmax + i * inc;
            std::vector<double> image(image_size * image_size);
            error = oskar_imager_dft_d(vis.num_samples, vis.amp, vis.u, vis.v,
                    settings.frequency(0), image_size, &l[0], &image[0]);
        }
        printf("= Completed imaging after %f seconds [error code: %i].\n",
                timer.elapsed() / 1.0e3, err);
        if (error != 0)
        {
            fprintf(stderr, "ERROR: CUDA dft imager failed with error = %i\n", error);
            return EXIT_FAILURE;
        }

        // Cleanup.
        oskar_free_vis_data_d(&vis);

    }
    // Single precision.
    else
    {



    }






//    std::vector<float> image_f(image_size * image_size);
//    for (unsigned i = 0; i < image_size * image_size; ++i)
//        image_f[i] = (float)image[i];
//
//    // plotting.
//    QApplication app(argc, argv);
//
//    PlotWidget plot1;
//    plot1.plotImage(&image_f[0], image_size, image_size, -fov_deg/2, fov_deg/2,
//            -fov_deg/2, fov_deg/2);
//    plot1.resize(700, 600);
//    plot1.setTitle("Dirty image");
//    plot1.setXLabel("Relative Right Ascension (deg)");
//    plot1.setYLabel("Relative Declination (deg)");
//    plot1.setScaleLabel("Jy/beam");
//
////    PlotWidget plot2;
////    plot2.plotCurve(vis.num_samples, vis.u, vis.v);
//
//    int status = app.exec();



    return status;
}



