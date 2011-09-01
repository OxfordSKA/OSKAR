#ifndef QTEST_GRIDDING_H_
#define QTEST_GRIDDING_H_

/**
 * @file QTest_Gridding.h
 */

#include "widgets/plotting/oskar_PlotWidget.h"
#include "imaging/oskar_ConvFunc.h"
#include "imaging/oskar_Gridder.h"

#include <QtGui/QApplication>
#include <QtCore/QObject>
#include <QtTest/QtTest>
#include <QtGui/QMainWindow>
#include <QtGui/QSplashScreen>

#include <vector>
#include <cmath>
#include <iostream>
#include <complex>

using namespace oskar;
using namespace std;

/**
 * @class QTestGridding
 *
 * @brief
 *
 * @details
 */

class QTestGridding : public QObject
{
        Q_OBJECT

    public:
        QTestGridding() {}
        ~QTestGridding()
        {
            for (unsigned i = 0; i < _p.size(); ++i)
                delete _p[i];
        }

    private slots:

//        void grid_distance()
//        {
//            const float pixel_size = 1.0f;
//            const unsigned oversample = 99;
//            int x_grid;
//            int x_conv_func;
//            Gridder g;
//
//            printf("---------------\n");
//            {
//                const float x = 5.5f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
////                printf("x conv func = %d\n", x_conv_func);
//            }
//
//            printf("---------------\n");
//            {
//                const float x = -5.5f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = 5.49f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = -5.49f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = 2.00f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = -2.00f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = 1.99f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = -1.99f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = 1.501f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//            {
//                const float x = -1.501f;
//                g.calculate_offset(x, pixel_size, oversample,
//                        &x_grid, &x_conv_func);
//                printf("x data = %f\n", x);
//                printf("x grid = %d\n", x_grid);
//                printf("x conv func = %d\n", x_conv_func);
//            }
//            printf("---------------\n");
//        }


        void griddingStandard()
        {
            const unsigned support = 3;
            const unsigned oversample = 99;

            // Convolution function.
            ConvFunc c;
//            c.pillbox(support, oversample);
//            c.exp(support, oversample);
            //c.expSinc(support, oversample);
            //c.sinc(support, oversample);

            typedef std::complex<float> Complex;

            // Data.
            const unsigned num_data = 3;
            vector<float> x(num_data, 0.0f);
            vector<float> y(num_data, 0.0f);
            vector<Complex> amp(num_data, Complex(0.0f, 0.0f));

            x[0]   = 0.0f;
            y[0]   = 0.0f;
            amp[0] = Complex(1.0f, 0.0f);

            x[1]   = 5.5f;
            y[1]   = 0.0f;
            amp[1] = Complex(1.0f, 0.0f);

            x[2]   = -5.5f;
            y[2]   = 0.0f;
            amp[2] = Complex(1.0f, 0.0f);

            // Grid.
            const unsigned grid_size = 40;
            vector<Complex> grid(grid_size * grid_size, Complex(0.0f, 0.0f));
            double grid_sum = 0.0;
            const float pixel_size = 1.0;

            // Gridding.
            Gridder g;
            g.grid_standard(num_data, &x[0], &y[0], &amp[0], support, oversample,
                    c.values(), grid_size, pixel_size, &grid[0], &grid_sum);

            _p.push_back(new PlotWidget);
            _p.back()->plotImage(grid_size, &grid[0], PlotWidget::RE, "grid");
        }



    private:
        std::vector<PlotWidget*> _p;
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QTestGridding test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}


#endif // QTEST_GRIDDING_H_
