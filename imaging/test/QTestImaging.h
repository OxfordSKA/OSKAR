#ifndef QTEST_IMAGING_H_
#define QTEST_IMAGING_H_

/**
 * @file QTestImaging.h
 */

#include "widgets/plotting/PlotWidget.h"
#include "imaging/ConvFunc.h"
#include "imaging/WProjConvFunc.h"
#include "imaging/oskar_types.h"

#include <QtGui/QApplication>
#include <QtCore/QObject>
#include <QtTest/QtTest>
#include <QtGui/QMainWindow>
#include <QtGui/QSplashScreen>

#include <vector>
#include <cmath>
#include <iostream>

using namespace oskar;
using namespace std;

/**
 * @class QTestimaging
 *
 * @brief
 *
 * @details
 */

class QTestImaging : public QObject
{
        Q_OBJECT

    public:
        QTestImaging() {}
        ~QTestImaging()
        {
            for (unsigned i = 0; i < _p.size(); ++i)
                delete _p[i];
        }

    private slots:

        void convFunc()
        {
            const unsigned support = 5;
            const unsigned oversample = 100;
            const unsigned size = (2 * support + 1) * oversample;
            vector<float> convFunc(size * size);

            // Generate the convolution function.
            ConvFunc::expSinc2D(support, oversample, &convFunc[0]);

            // Plot the convolution function.
            _p.push_back(new PlotWidget);
            _p.back()->plotImage(size, &convFunc[0], "expSinc2D");
        }


        void wprojConvFuncLM()
        {
            const unsigned sizeLM = 512;
            const unsigned padding = 3;
            const unsigned size = sizeLM * padding;
            const unsigned iStart = sizeLM * (unsigned)floor(float(padding) / 2.0f);
            const float imageWidthLM_rads = 10.0f * (M_PI / 180.0f);
            const float pixelSizeLM_rads = imageWidthLM_rads / (float)sizeLM;
            const float w = 1000.0f;
            const float taperWidth = 0.2;
            const float sigma = taperWidth * (float)sizeLM;
            const float sigma2 = sigma * sigma;
            const float taperFactor = 1.0f / (2.0f * sigma2);

            // Generate the convolution function.
            WProjConvFunc c;
            c.generateLM(size, sizeLM, iStart, pixelSizeLM_rads, w, taperFactor);

            // Plot the convolution function.
            _p.push_back(new PlotWidget);
            _p.back()->plotImage(size, c.values(), PlotWidget::RE, "WProjConvFunc LMm");
        }


        void wprojConvFuncUV()
        {
            const unsigned sizeLM = 512;
            const unsigned padding = 3;
            const unsigned size = sizeLM * padding;
            const unsigned iStart = sizeLM * (unsigned)floor(float(padding) / 2.0f);
            const float imageWidthLM_rads = 10.0f * (M_PI / 180.0f);
            const float pixelSizeLM_rads = imageWidthLM_rads / (float)sizeLM;
            const float w = 1000.0f;
            const float taperWidth = 0.2;
            const float sigma = taperWidth * (float)sizeLM;
            const float sigma2 = sigma * sigma;
            const float taperFactor = 1.0f / (2.0f * sigma2);
            const float cutoff = 0.001; // cut at the 0.1% level.

            // Generate the convolution function.
            WProjConvFunc c;
            c.generateUV(size, sizeLM, iStart, pixelSizeLM_rads, w, taperFactor,
                    cutoff);

            // Plot the convolution function.
            _p.push_back(new PlotWidget);
            _p.back()->plotImage(size, c.values(), PlotWidget::RE, "WProjConvFunc UV");
        }

    private:
        std::vector<PlotWidget*> _p;
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QTestImaging test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}


#endif // QTEST_IMAGING_H_
