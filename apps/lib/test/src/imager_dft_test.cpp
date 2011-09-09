#include "apps/lib/oskar_imager_dft.cu.h"
#include "interferometry/oskar_VisData.h"

#include "widgets/plotting/oskar_PlotWidget.h"
#include "widgets/plotting/oskar_ImagePlot.h"

#include <QtGui/QApplication>

#include <cmath>
#include <vector>
#include <cstdio>

using namespace oskar;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif


int main(int argc, char** argv)
{
    oskar_VisData_d vis;
    oskar_allocate_vis_data_d(2, &vis);
    double u = 0.0;
    double v = 5.0;
    vis.u[0]     = u;
    vis.v[0]     = v;
    vis.amp[0].x = 1.0;
    vis.u[1]     = -u;
    vis.v[1]     = -v;
    vis.amp[1].x = 1.0;

    double fov_deg = MAX(fabs(u), fabs(v));
    int image_size = 512;
    double l_max = sin((fov_deg / 2.0) * (M_PI / 180));
    double l_inc = (2.0 * l_max) / (image_size - 1);
    vector<double> l(image_size);
    for (int i = 0; i < image_size; ++i)
    {
        l[i] = -l_max + i * l_inc;
    }

    vector<double> image(image_size * image_size, 0.0);

    const double c_0 = 299792458;
    double frequency = c_0 / (2 * l_max);
    int err = oskar_imager_dft_d(vis.num_samples, vis.amp, vis.u, vis.v,
            frequency, image_size, &l[0], &image[0]);

    if (err != 0)
    {
        fprintf(stderr, "ERROR: CUDA dft imager failed with error = %i\n", err);
        return EXIT_FAILURE;
    }

    // plot image
    std::vector<float> image_f(image_size * image_size);
    for (int i = 0; i < image_size * image_size; ++i)
    {
        image_f[i] = (float)image[i];
    }

    QApplication app(argc, argv);
    PlotWidget plot1;
    plot1.plotImage(&image_f[0], image_size, image_size, -fov_deg/2, fov_deg/2, -fov_deg/2, fov_deg/2);

    int status = app.exec();

    oskar_free_vis_data_d(&vis);
    return status;
}



