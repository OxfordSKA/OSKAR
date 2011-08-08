#ifndef QTEST_CONV_FUNC_H_
#define QTEST_CONV_FUNC_H_

/**
 * @file QTestConvFunc.h
 */

#include "widgets/plotting/oskar_PlotWidget.h"
#include "imaging/ConvFunc.h"
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
 * @class QTestConvFunc
 *
 * @brief
 *
 * @details
 */

class QTestConvFunc : public QObject
{
        Q_OBJECT

    public:
        QTestConvFunc() {}
        ~QTestConvFunc()
        {
            for (unsigned i = 0; i < _p.size(); ++i)
                delete _p[i];
        }

    private slots:

//        void pillbox()
//        {
//            const unsigned support = 2;     // odd or even?
//            const unsigned oversample = 99; // needs to be odd?
//            const float width = 0.8f;
//
//            // TODO: NEED TO CHECK CENTERING OF THIS FUNCTION
//            // WITH DIFFERENT SUPPORT AND OVERSAMPLE.
//
//            ConvFunc c;
//            c.pillbox(support, oversample, width);
//
//            vector<float> x(c.size());
//            for (unsigned i = 0; i < c.size(); ++i)
//                x[i] = static_cast<float>(i);
//
//            // Plot the convolution function.
//            _p.push_back(new PlotWidget);
//            _p.back()->plotCurve(c.size(), c.values(), &x[0], "pillbox 1D", false);
//
//            c.makeConvFuncImage();
//            _p.push_back(new PlotWidget);
//            _p.back()->plotImage(c.size(), c.values(), "pillbox 2D");
//        }

//        void exp()
//        {
//            const unsigned support = 3;      // odd or even?
//            const unsigned oversample = 9;   // needs to be odds?
//
//            ConvFunc c;
//            c.exp(support, oversample);
//
//            vector<float> x(c.size());
//            for (unsigned i = 0; i < c.size(); ++i)
//                x[i] = static_cast<float>(i);
//
//            // Plot the convolution function.
//            _p.push_back(new PlotWidget);
//            _p.back()->plotCurve(c.size(), c.values(), &x[0], "exp 1D", false);
//
//            c.makeConvFuncImage();
//            _p.push_back(new PlotWidget);
//            _p.back()->plotImage(c.size(), c.values(), "exp 2D");
//        }

//        void sinc()
//        {
//            const unsigned support = 5;
//            const unsigned oversample = 3;
//
//            ConvFunc c;
//            c.sinc(support, oversample);
//
//            vector<float> x(c.size());
//            for (unsigned i = 0; i < c.size(); ++i)
//                x[i] = static_cast<float>(i);
//
//            // Plot the convolution function.
//            _p.push_back(new PlotWidget);
//            _p.back()->plotCurve(c.size(), c.values(), &x[0], "sinc 1D", false);
//
//            c.makeConvFuncImage();
//            _p.push_back(new PlotWidget);
//            _p.back()->plotImage(c.size(), c.values(), "sinc 2D");
//        }

        void expSinc()
        {
            const unsigned support = 5;
            const unsigned oversample = 3;

            ConvFunc c;
            c.expSinc(support, oversample);

            vector<double> x(c.size()), y(c.size());
            for (unsigned i = 0; i < c.size(); ++i)
            {
                x[i] = static_cast<double>(i);
                y[i] = static_cast<double>(c.values()[i]);
            }


            // Plot the convolution function.
            _p.push_back(new PlotWidget);
            _p.back()->plotCurve(c.size(), &x[0], &y[0]);
            _p.back()->setTitle("exp sinc 1D");

            c.makeConvFuncImage();
            _p.push_back(new PlotWidget);
            _p.back()->plotImage(c.size(), c.values(), "exp-sinc 2D");
        }

    private:
        std::vector<PlotWidget*> _p;
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QTestConvFunc test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}


#endif // QTEST_CONV_FUNC_H_
