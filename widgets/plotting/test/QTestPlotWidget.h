#ifndef QTEST_p1_WIDGET_H_
#define QTEST_p1_WIDGET_H_

/**
* @file QTestPlotWidget.h
*/

#include "widgets/plotting/oskar_PlotWidget.h"

#include <qwt_symbol.h>

#include <QtGui/QApplication>
#include <QtGui/QWidget>
#include <QtCore/QObject>
#include <QtTest/QtTest>
#include <QtGui/QMainWindow>
#include <QtGui/QSplashScreen>
#include <QtCore/QVector>
#include <QtCore/QTime>

#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

using namespace oskar;

/**
* @class QTestPlotWidget
*
* @brief
*
* @details
*/

class QTestPlotWidget : public QObject
{
        Q_OBJECT

    public:
        QTestPlotWidget() {
            _p1 = new PlotWidget;
            _p2 = new PlotWidget;
            _p3 = new PlotWidget;
        }
        ~QTestPlotWidget() {
            delete _p1;
            delete _p2;
            delete _p3;
        }

    private slots:

        void curve()
        {
            // Create nPoints data values
            unsigned nPoints = 1000;
            double nPeriods = 2;
            std::vector<float> x(nPoints);
            std::vector<float> y(nPoints);

            for (unsigned i = 0; i < nPoints; ++i) {
                x[i] = float(i);
                double arg = 2 * M_PI * (nPeriods / float(nPoints)) * x[i];
                y[i] = float(sin(arg));
                //y[i] = float(i);
            }

            std::vector<double> xPlot(x.begin(), x.end());
            std::vector<double> yPlot(y.begin(), y.end());

            _p1->resize(500, 500);
            _p1->show();
            _p1->plotCurve(nPoints, &xPlot[0], &yPlot[0]);
            _p1->setPlotTitle("Curve plot");
            //_p1->showGrid(true);
            //_p1->showGridMinorTicks(true);
            //_p1->savePNG("test1.png");
        }


        void image()
        {

            unsigned nX = 100, nY = 100;
            std::vector<float> data(nX * nY);
            for (unsigned i = 0; i < nY; ++i) {
                for (unsigned j = 0; j < nY; ++j) {
                    data[i * nX + j] = float(i + j);
                }
            }

            _p2->resize(500, 500);  // This method is inherited from QWidget.
            _p2->show();            // This method is inherited from QWidget.
            try {
                _p2->plotImage(&data[0], nX, nY, 0, 100.0, 0, 100.0);
            }
            catch (const QString& err) {
                std::cout << err.toStdString() << std::endl;
            }

            //_p2->savePNG("test2.png");
            //_p2->savePDF("test2");
        }


//        void curve_performance()
//        {
//            int num_points = (int)100e6;
//            _x.resize(num_points);
//            _y.resize(num_points);
//
//            double *x = _x.data();
//            double *y = _y.data();
//            for (int i = 0; i < num_points; ++i)
//            {
//                x[i] = (double)rand() / ((double)RAND_MAX + 1.0);
//                y[i] = (double)rand() / ((double)RAND_MAX + 1.0);
//            }
//
//            printf("Starting plot...\n");
//            QTime timer;
//            timer.start();
//            {
//                _plots.push_back(new QwtPlot(QwtText("test plot")));
//                _curves.push_back(new QwtPlotCurve("test curve"));
////                _s.setStyle(QwtSymbol::Ellipse);
////                _s.setPen(QPen(Qt::black));
////                _s.setBrush(QBrush(Qt::black));
////                _s.setSize(QSize(1,1));
////                _curves.back()->setSymbol(_s);
////                _curves.back()->setStyle(QwtPlotCurve::NoCurve);
//                _curves.back()->setStyle(QwtPlotCurve::Dots);
//                _curves.back()->setRawData(_x.constData(), _y.constData(), _x.size());
//                _curves.back()->attach(_plots.back());
//                _plots.back()->resize(500, 500);
//                _plots.back()->show();
//                while (!_plots.back()->isVisible())
//                {}
////                sleep(1);
//            }
//            printf("time taken to plot %f\n", (double)timer.elapsed() / 1.0e3);
//        }


        void empty()
        {
            _p3->resize(500, 500);
            _p3->show();
        }

    private:
        QVector<QwtPlot*> _plots;
        QVector<QwtPlotCurve*> _curves;
        QVector<double> _x;
        QVector<double> _y;
        QwtSymbol _s;
        PlotWidget* _p1;
        PlotWidget* _p2;
        PlotWidget* _p3;
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QTestPlotWidget test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}


#endif // QTEST_CONFIG_OPTIONS_TABLE_H
