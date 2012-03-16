#include "oskar_global.h"

#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_vector_types.h"

//#include "imaging/fft/oskar_GridKernel.h"
//#include "imaging/fft/oskar_VisGrid.h"
//#include "imaging/fft/oskar_evaluate_gridding_kernels.h"
//#include "imaging/fft/oskar_gridding.h"
//#include "imaging/fft/oskar_fft_utility.h"
//#include "imaging/fft/oskar_grid_correct.h"
//
//#include "widgets/plotting/oskar_PlotWidget.h"
//
//#include <QtGui/QApplication>
//#include <QtCore/QString>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int /*argc*/, char** /*argv*/)
{
    int status = OSKAR_SUCCESS;

    try {
        int type = OSKAR_DOUBLE_COMPLEX;
        int location = OSKAR_LOCATION_CPU;
        int num_channels = 1;
        int num_times = 1;
        int num_baselines = 2;
        oskar_Visibilities vis(type, location, num_channels, num_times, num_baselines);

        double u = 5.0;
        ((double*)vis.uu_metres.data)[0] = -u;
        ((double*)vis.vv_metres.data)[0] = 0.0;
        ((double*)vis.ww_metres.data)[0] = 0.0;
        ((double2*)vis.amplitude.data)[0].x = 1.0;
        ((double2*)vis.amplitude.data)[0].y = 0.0;

        /* FIXME should really not need the hermitian copy to be in the
         * vis data! */
        ((double*)vis.uu_metres.data)[1] = u;
        ((double*)vis.vv_metres.data)[1] = 0.0;
        ((double*)vis.ww_metres.data)[1] = 0.0;
        ((double2*)vis.amplitude.data)[1].x = 1.0;
        ((double2*)vis.amplitude.data)[1].y = 0.0;

//        oskar_GridKernel_d kernel;
//        oskar_evaluate_pillbox_d(&kernel);
//        //oskar_evaluate_exp_sinc_d(&kernel);
//        //oskar_evaluate_spheroidal_d(&kernel);
//
//        oskar_VisGrid_d grid;
//        int grid_size = 512;
//        oskar_allocate_vis_grid_d(grid_size, &grid);
//        grid.pixel_separation = 1.0;
//
//        double sum = oskar_grid_standard(&vis, &kernel, &grid);
//        printf("grid sum = %f\n", sum);
//
//        oskar_free_vis_grid_d(&grid);
    }
    catch (const char* error)
    {
        fprintf(stderr, "ERROR %s.\n", error);
    }

    //    for (int i = 0; i < grid.size * grid.size; ++i)
    //    {
    //        grid.amp[i].x /= sum;
    //        grid.amp[i].y /= sum;
    //    }
    //
    //    vector<double> image(grid.size * grid.size, 0.0);
    //    oskar_fft_z2r_2d(grid.size, grid.amp, &image[0]);
    //
    //    double* correction = NULL;
    //    oskar_evaluate_grid_correction_d(&kernel, grid.size, &correction);
    //
    //    for (int i = 0; i < grid.size * grid.size; ++i)
    //        image[i] /= correction[i];
    //
    //    vector<float> image_f(grid.size * grid.size, 0.0);
    //    vector<float> grid_f(grid.size * grid.size, 0.0);
    //    vector<float> correction_f(grid.size * grid.size, 0.0);
    //    vector<float> kernel_f(kernel.size * kernel.size, 0.0);
    //    for (int i = 0; i < grid.size * grid.size; ++i)
    //    {
    //        grid_f[i]       = (float) grid.amp[i].x;
    //        image_f[i]      = (float) image[i];
    //        correction_f[i] = (float) correction[i];
    ////        correction_f[i] = (float) (image[i] / correction[i] - 1);
    //        if (isinf(correction_f[i])) correction_f[i] = NAN;
    //    }
    //    for (int j = 0; j < kernel.size; ++j)
    //    {
    //        for (int i = 0; i < kernel.size; ++i)
    //            kernel_f[j * kernel.size + i] = (float)(kernel.amp[j] * kernel.amp[i]);
    //    }
    //
    //
    //    int hc_size = (grid.size / 2) + 1;
    //    vector<double2> hc_grid(grid.size * hc_size);
    //    vector<float> hc_grid_f(grid.size * hc_size);
    //    for (int j = 0; j < grid.size; ++j)
    //    {
    //        const void* from = &(grid.amp[j * grid.size]);
    //        void* to = &(hc_grid[j * hc_size]);
    //        memcpy(to, from, hc_size * sizeof(double2));
    //    }
    //    for (int i = 0; i < grid.size * hc_size; ++i)
    //        hc_grid_f[i] = (float)hc_grid[i].x;
    //
    //
    //    QApplication app(argc, argv);
    //
    ////    PlotWidget plot0;
    ////    plot0.setWindowTitle("kernel");
    ////    plot0.plotImage(&kernel_f[0], kernel.size, kernel.size, -kernel.num_cells/2.0, kernel.num_cells/2.0, -kernel.num_cells/2.0, kernel.num_cells/2.0);
    //
    //    PlotWidget plot1;
    //    plot1.setWindowTitle("image");
    //    plot1.plotImage(&image_f[0], grid.size, grid.size, 0, grid.size, 0, grid.size);
    //
    ////    PlotWidget plot2;
    ////    plot2.setWindowTitle("grid");
    ////    plot2.plotImage(&grid_f[0], grid.size, grid.size, 0, grid.size, 0, grid.size);
    //
    //    PlotWidget plot3;
    //    plot3.setWindowTitle("correction");
    //    plot3.plotImage(&correction_f[0], grid.size, grid.size, 0, grid.size, 0, grid.size);
    //
    ////    PlotWidget plot4;
    ////    plot4.setWindowTitle("hc grid");
    ////    plot4.plotImage(&hc_grid_f[0], hc_size, grid.size, 0, hc_size, 0, grid.size);
    //
    //    int status = app.exec();
    //
    //    free(correction);

    return status;
}


