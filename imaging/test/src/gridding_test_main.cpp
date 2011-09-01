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

void allocate_vis_data_d(const unsigned num_samples, oskar_VisData_d* vis);
void free_vis_data_d(oskar_VisData_d* vis);

void allocate_vis_grid_d(const unsigned grid_dim, oskar_VisGrid_d* grid);
void free_vis_grid_d(oskar_VisGrid_d* vis);

int main(int argc, char** argv)
{
    oskar_GridKernel_d kernel;
    //oskar_evaluate_pillbox_d(&kernel);
    oskar_evaluate_exp_sinc_d(&kernel);

    oskar_VisData_d vis;
    allocate_vis_data_d(1, &vis);
    vis.u[0]     = 3.0;
    vis.v[0]     = 0.0;
    vis.w[0]     = 0.0;
    vis.amp[0].x = 1.0;
    vis.amp[0].y = 0.0;

    oskar_VisGrid_d grid;
    int grid_dim = 10;
    allocate_vis_grid_d(grid_dim, &grid);
    grid.pixel_separation = 1.0;

    double sum = oskar_grid_standard(&vis, &kernel, &grid);
    printf("grid sum = %f\n", sum);

    vector<float> image(grid_dim * grid_dim, 0.0);
    for (int i = 0; i < grid_dim * grid_dim; ++i)
    {
        image[i] = (float) grid.amp[i].x;
    }

    QApplication app(argc, argv);

    PlotWidget plot1;
    plot1.plotImage(&image[0], grid_dim, grid_dim, 0, grid_dim, 0, grid_dim);

    int status = 0;
    status = app.exec();

    printf("cleaning up...\n");
    free_vis_data_d(&vis);
    free_vis_grid_d(&grid);

    return status;
}


void allocate_vis_data_d(const unsigned num_samples, oskar_VisData_d* vis)
{
    vis->num_samples = num_samples;
    vis->u   = (double*)  malloc(vis->num_samples * sizeof(double));
    vis->v   = (double*)  malloc(vis->num_samples * sizeof(double));
    vis->w   = (double*)  malloc(vis->num_samples * sizeof(double));
    vis->amp = (double2*) malloc(vis->num_samples * sizeof(double2));

    memset(vis->u,   0, vis->num_samples * sizeof(double));
    memset(vis->v,   0, vis->num_samples * sizeof(double));
    memset(vis->w,   0, vis->num_samples * sizeof(double));
    memset(vis->amp, 0, vis->num_samples * sizeof(double2));
}

void free_vis_data_d(oskar_VisData_d* vis)
{
    vis->num_samples = 0;
    free(vis->u);
    free(vis->v);
    free(vis->w);
    free(vis->amp);
}

void allocate_vis_grid_d(const unsigned grid_dim, oskar_VisGrid_d* grid)
{
    grid->grid_dim = grid_dim;
    grid->amp = (double2*) malloc(grid_dim * grid_dim * sizeof(double2));
    memset(grid->amp, 0, grid_dim * grid_dim * sizeof(double2));
}


void free_vis_grid_d(oskar_VisGrid_d* grid)
{
    grid->grid_dim = 0;
    free(grid->amp);
}
