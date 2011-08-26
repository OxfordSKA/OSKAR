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



#include "apps/oskar_VisData.h"
#include "apps/oskar_Settings.h"
#include "apps/oskar_load_telescope.h"
#include "apps/oskar_imager_dft.cu.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_vector_types.h"

#include "widgets/plotting/oskar_PlotWidget.h"
#include "widgets/plotting/oskar_ImagePlot.h"

#include <QtGui/QApplication>

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
//    settings.print();

    double fov_deg      = atof(argv[2]);
    unsigned image_size = atoi(argv[3]);

    printf("field of view (deg) = %f\n", fov_deg);
    printf("image size          = %i\n", image_size);
    printf("\n");

    oskar_TelescopeModel telescope;
    oskar_load_telescope(settings.telescope_file().toLatin1().data(),
            settings.longitude_rad(), settings.latitude_rad(), &telescope);

    // Load data file.
    oskar_VisData data(telescope.num_antennas, settings.num_vis_dumps());
    data.load(settings.output_file().toLatin1().data());

    std::vector<double> l(image_size);
    double lmax = sin((fov_deg / 2) * M_PI / 180.0);
    double inc  = (2.0 * lmax) / (image_size - 1);
    for (unsigned i = 0; i < image_size; ++i)
    {
        l[i] = -lmax + i * inc;
    }

    std::vector<double> image(image_size * image_size);

    printf("num_vis = %i\n", data.size());
    double2* vis = data.vis();
    double* u = data.u();
    double* v = data.v();

    int err = oskar_imager_dft_d(data.size(), vis, u, v,
            settings.frequency(), image_size, &l[0], &image[0]);

    if (err != 0)
    {
        fprintf(stderr, "ERROR: CUDA dft imager failed with error = %i\n", err);
        return EXIT_FAILURE;
    }

    std::vector<float> image_f(image_size * image_size);
    for (unsigned i = 0; i < image_size * image_size; ++i)
        image_f[i] = (float)image[i];

    // plot image
    QApplication app(argc, argv);
    PlotWidget plot2;
    plot2.resize(500, 500);
    plot2.show();
    plot2.plotCurve(data.size(), data.u(), data.v());

    PlotWidget plot1;
    plot1.resize(500, 500);
    plot1.show();
    try {
        plot1.plotImage(&image_f[0], image_size, image_size, 0, image_size, 0, image_size, true);
    }
    catch (const QString& error)
    {
        fprintf(stderr, "ERROR: plotting image failed: %s\n",
                error.toLatin1().data());
        return EXIT_FAILURE;
    }

    return app.exec();
}



