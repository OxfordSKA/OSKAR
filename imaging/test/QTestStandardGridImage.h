#ifndef QTEST_STANDARD_GRID_IMAGE_H_
#define QTEST_STANDARD_GRID_IMAGE_H_

/**
 * @file QTestStandardGridImage.h
 */

#include "widgets/plotting/PlotWidget.h"
#include "imaging/ConvFunc.h"
#include "imaging/Gridder.h"
#include "imaging/FFTUtility.h"
#include "imaging/oskar_types.h"

#include <QtGui/QApplication>
#include <QtCore/QObject>
#include <QtTest/QtTest>
#include <QtGui/QMainWindow>
#include <QtGui/QSplashScreen>

#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

using namespace oskar;
using namespace std;

/**
 * @class QTestStandardGridImage.h
 *
 * @brief
 *
 * @details
 */

class QTestStandardGridImage : public QObject
{
        Q_OBJECT

    public:
        QTestStandardGridImage() {}
        ~QTestStandardGridImage()
        {
            for (unsigned i = 0; i < _p.size(); ++i)
                delete _p[i];
        }

    private slots:


        void image1()
        {
            const unsigned support = 3;
            const unsigned oversample = 99;

            // Convolution function.
            ConvFunc c;
            //c.pillbox(support, oversample);
            c.exp(support, oversample);
            //c.expSinc(support, oversample);
//            c.sinc(support, oversample);

            // Data.
            vector<float> x;
            vector<float> y;
            vector<Complex> amp;

            {
                x.push_back(0.0f);
                y.push_back(0.0f);
                amp.push_back(Complex(1.0f, 0.0f));
            }

//            {
//                x.push_back(2.0f);
//                y.push_back(0.0f);
//                amp.push_back(Complex(1.0f, 0.0f));
//
//                x.push_back(-2.0f);
//                y.push_back(0.0f);
//                amp.push_back(Complex(1.0f, 0.0f));
//            }

            const unsigned num_data = x.size();

            // Grid.
            const unsigned grid_size = 128;
            vector<Complex> grid(grid_size * grid_size, Complex(0.0f, 0.0f));
            float grid_sum = 0.0;
            const float pixel_size = 1.0;

            // Gridding.
            Gridder g;
            g.grid_standard(num_data, &x[0], &y[0], &amp[0], support, oversample,
                    c.values(), grid_size, pixel_size, &grid[0], &grid_sum);
            for (unsigned i = 0; i < grid_size * grid_size; ++i)
            {
                grid[i] /= grid_sum;
            }
            printf("grid sum = %f\n", grid_sum);


            _p.push_back(new PlotWidget);
            _p.back()->plotImage(grid_size, &grid[0], PlotWidget::RE, "grid");

            // FFT to form the image.
            vector<float> image(grid_size * grid_size, 0.0f);
            FFTUtility::fft_c2r_2d(grid_size, &grid[0], &image[0]);
            _p.push_back(new PlotWidget);
            _p.back()->plotImage(grid_size, &image[0], "image");

            float image_max = -numeric_limits<float>::max();
            for (unsigned i = 0; i < grid_size * grid_size; ++i)
            {
                image_max = max<float>(image_max, image[i]);
            }
            printf("image max = %f\n", image_max);

            // Grid correct (use a 1d dft?)
            // C = C(x) * C(y)
            // FT(C) = ...
            // see AIPS Q/SUB/NOTST/GRDTAB.FOR ...
            // see CASA code/synthesis/MeasurementComponents/GridFT.cc
            //                       ----   line 318
            //    casacore/scimath/Mathmatrics/ConvolveGridder.h
            //    casacore/scimath/Mathmatrics/Gridder.h


        }



    private:
        std::vector<PlotWidget*> _p;
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QTestStandardGridImage test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}


#endif // QTEST_STANDARD_GRID_IMAGE_H_
