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

#include <QtGui/QApplication>
#include <QtCore/QTime>
#include <QtGui/QPalette>

#include "widgets/plotting/oskar_PlotWidget.h"


#include "qwt_scale_widget.h"

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace oskar; // FIXME: take plot widget out of oskar namespace.

int main(int argc, char** argv)
{
    // $> oskar_viewer_scalar settings file
    if (argc != 4)
    {
        fprintf(stderr, "ERROR: missing command line arguments\n");
        fprintf(stderr, "Usage:  $ oskar_viewer_scalar [settings file] [channel] [snapshot]\n");
        return EXIT_FAILURE;
    }

    // Load settings file.
    oskar_Settings settings;
    settings.load(QString(argv[1]));

    // Load image date from file... (animate in loop?)
    unsigned channel = (unsigned)atoi(argv[2]); // channel number
    unsigned t = (unsigned)atoi(argv[3]);  // snapshot number.
    unsigned image_size = settings.image().size();


    QString image_dir  = "./";
    //QString image_dir  = "./output/";
    QString image_file = image_dir + settings.image().filename() + "_channel_"
            + QString::number(channel) + "_t_" + QString::number(t) + ".dat";

    FILE* file;
    file = fopen(image_file.toLatin1().data(), "rb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open oskar image data file.\n");
        return EXIT_FAILURE;
    }

    // single precision
    unsigned num_pixels = image_size * image_size;
    float* image = (float*) malloc(num_pixels * sizeof(float));
    size_t read = fread(image, sizeof(float), num_pixels, file);
    if ((unsigned)read != num_pixels)
    {
        fprintf(stderr, "ERROR read the wrong number of bytes from image file. %i != %i\n",
                (int)read, num_pixels);
        return EXIT_FAILURE;
    }

    fclose(file);


    //====== Plotting. =======================================================
    QApplication app(argc, argv);

    double fov_deg = settings.image().fov_deg();

    PlotWidget plot1;
    plot1.setCanvasBackground(Qt::black);

    plot1.show();
    QPalette p;
    p.setColor(QPalette::Window, Qt::black);
    p.setColor(QPalette::WindowText, Qt::white);
    p.setColor(QPalette::Text, Qt::white);
    plot1.setBackgroundRole(QPalette::Window);
    plot1.setPalette(p);
    double extent = fov_deg / 2.0;
    plot1.plotImage(image, image_size, image_size, -extent, extent,
            -extent, extent);
    plot1.setYLabel("Relative Declination (deg)");
    plot1.setXLabel("Relative Right Ascension x cos(Dec) x -1 (deg)");
    plot1.setTitle("Dirty image (channel = " + QString::number(channel) + ", "
            " t = " + QString::number(t) + ")");
    plot1.setScaleLabel("Brightness");

    plot1.toggleGrid();
    int status = 0;
    status = app.exec();
    // ========================================================================

    // Free memory
    free(image);

    return status;
}
