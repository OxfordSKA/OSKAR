#include "imaging/oskar_GridKernel.h"
#include "imaging/oskar_evaluate_gridding_kernels.h"

#include "widgets/plotting/oskar_PlotWidget.h"

#include <QtGui/QApplication>

#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace std;
using namespace oskar;

int main(int argc, char** argv)
{
    oskar_GridKernel_d kernel;
    //oskar_evaluate_pillbox_d(&kernel);
    //oskar_evaluate_exp_sinc_d(&kernel);
    oskar_evaluate_spheroidal_d(&kernel);

    // Generate indices for curve plot.
    vector<double> x(kernel.size);
    for (int i = 0; i < kernel.size; ++i) x[i] = i;

    // Convert 1d kernel to image.
    vector<float> image1(kernel.size * kernel.size);
    for (int j = 0; j < kernel.size; ++j)
    {
        for (int i = 0; i <  kernel.size; ++i)
        {
            image1[j * kernel.size + i] = kernel.values[j] * kernel.values[i];
        }
    }

    QApplication app(argc, argv);

    PlotWidget plot1;
    plot1.plotImage(&image1[0], kernel.size, kernel.size, 0.0, kernel.size, 0.0, kernel.size);

    PlotWidget plot2;
    plot2.plotCurve(kernel.size, &x[0], kernel.values);

    return app.exec();
}



