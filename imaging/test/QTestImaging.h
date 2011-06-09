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

        //        void convFunc()
        //        {
        //            const unsigned support = 5;
        //            const unsigned oversample = 100;
        //            const unsigned size = (2 * support + 1) * oversample;
        //            vector<float> convFunc(size * size);
        //
        //            // Generate the convolution function.
        //            ConvFunc::expSinc2D(support, oversample, &convFunc[0]);
        //
        //            // Plot the convolution function.
        //            _p.push_back(new PlotWidget);
        //            _p.back()->plotImage(size, &convFunc[0], "expSinc2D");
        //        }

//        void wprojConvFuncLM()
//        {
//            const unsigned innerSize = 5;
//            const unsigned padding = 3;
//            const float imageWidthLM_rads = 10.0f * (M_PI / 180.0f);
//            const float pixelSizeLM_rads = imageWidthLM_rads / (float)innerSize;
//
//            const float w = 50.0f;
//            const float taperWidth = 0.15;
//            const float sigma = taperWidth * (float)innerSize;
//            const float sigma2 = sigma * sigma;
//            const float taperFactor = 1.0f / (2.0f * sigma2);
//
//            // Generate the convolution function.
//            WProjConvFunc c;
//            c.generateLM(innerSize, padding, pixelSizeLM_rads, w, taperFactor);
//
//            // Plot the convolution function.
//            _p.push_back(new PlotWidget);
//            _p.back()->plotImage(c.size(), c.values(), PlotWidget::RE, "WProjConvFunc LM Real");
//        }
//
//
//        void wprojConvFuncUV()
//        {
//            const unsigned innerSize = 5;
//            const unsigned padding = 3;
//            const float imageWidthLM_rads = 10.0f * (M_PI / 180.0f);
//            const float pixelSizeLM_rads = imageWidthLM_rads / (float)innerSize;
//
//            const float w = 50.0f;
//            const float taperWidth = 0.15;
//            const float sigma = taperWidth * (float)innerSize;
//            const float sigma2 = sigma * sigma;
//            const float taperFactor = 1.0f / (2.0f * sigma2);
//
//            const float cutoff = 0.001; // 0.1%
//
//            // Generate the convolution function.
//            WProjConvFunc c;
//            c.generateUV(innerSize, padding, pixelSizeLM_rads, w, taperFactor,
//                    cutoff);
//
//            // Plot the convolution function.
//            _p.push_back(new PlotWidget);
//            _p.back()->plotImage(c.size(), c.values(), PlotWidget::RE, "WProjConvFunc UV Real");
//        }


        void wprojConvFunc()
        {
            const unsigned innerSize = 128; // needs to be even
            const unsigned padding = 2;   // can be even or odd.
            const float imageWidthLM_rads = 10.0f * (M_PI / 180.0f);
            const float pixelSizeLM_rads = imageWidthLM_rads / (float)innerSize;

            const float w = 500.0f;
            const float taperWidth = 0.2;
            const float sigma = taperWidth * (float)innerSize;
            const float sigma2 = sigma * sigma;
            const float taperFactor = 1.0f / (2.0f * sigma2);

            const float cutoff = 0.001; // 0.1%

            // Generate the convolution function.
            WProjConvFunc cLM;
            cLM.generateLM(innerSize, padding, pixelSizeLM_rads, w, taperFactor);

            WProjConvFunc cUV;
            cUV.generateUV(innerSize, padding, pixelSizeLM_rads, w, taperFactor,
                    cutoff);

            // Plot the convolution function.
            _p.push_back(new PlotWidget);
            _p.back()->plotImage(cLM.size(), cLM.values(), PlotWidget::RE, "WProjConvFunc LM Real");
            _p.push_back(new PlotWidget);
            _p.back()->plotImage(cUV.size(), cUV.values(), PlotWidget::ABS, "WProjConvFunc UV Real");
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
