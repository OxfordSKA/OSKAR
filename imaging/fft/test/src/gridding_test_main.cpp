#include "imaging/oskar_GridKernel.h"
#include "imaging/oskar_evaluate_gridding_kernels.h"
#include "imaging/oskar_gridding.h"

#include "interferometry/oskar_VisData.h"
#include "imaging/oskar_GridKernel.h"
#include "imaging/oskar_VisGrid.h"

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
    oskar_evaluate_exp_sinc_d(&kernel);

    oskar_VisData_d vis;
    oskar_allocate_vis_data_d(1, &vis);
    vis.u[0]     = 2.0;
    vis.v[0]     = 2.0;
    vis.w[0]     = 0.0;
    vis.amp[0].x = 1.0;
    vis.amp[0].y = 0.0;

    oskar_VisGrid_d grid;
    int grid_size = 20;
    oskar_allocate_vis_grid_d(grid_size, &grid);
    grid.pixel_separation = 1.0;

    double sum = oskar_grid_standard(&vis, &kernel, &grid);
    printf("grid sum = %f\n", sum);

    vector<float> image(grid_size * grid_size, 0.0);
    for (int i = 0; i < grid_size * grid_size; ++i)
    {
        image[i] = (float) grid.amp[i].x;
    }

    QApplication app(argc, argv);

    PlotWidget plot1;
    plot1.plotImage(&image[0], grid_size, grid_size, 0, grid_size, 0, grid_size);

    int status = 0;
    status = app.exec();

    printf("cleaning up...\n");
    oskar_free_vis_data_d(&vis);
    oskar_free_vis_grid_d(&grid);

    return status;
}
