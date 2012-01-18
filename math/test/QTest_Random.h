/*
 * Copyright (c) 2011, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef QTEST_RANDOM_H_
#define QTEST_RANDOM_H_

/**
 * @file QTest_Random.h
 */


#include <QtGui/QApplication>
#include <QtCore/QObject>
#include <QtCore/QVector>
#include <QtTest/QtTest>
#include <QtCore/QTime>

#include "math/oskar_random_uniform.h"
#include "math/oskar_random_gaussian.h"
#include "math/oskar_random_power_law.h"
#include "math/oskar_random_broken_power_law.h"
#include "widgets/plotting/oskar_PlotWidget.h"

using namespace oskar;

/**
 * @brief Unit test class that uses QTest.
 *
 * @details
 * This class uses the QTest testing framework to perform unit tests
 * on the class it is named after.
 */
class QTest_Random : public QObject
{
    private:
        Q_OBJECT

    public:
        QTest_Random() {}
        ~QTest_Random() {}

    private slots:
        void uniform()
        {
            const unsigned n = 10000;
            QVector<double> rand(n);
            QVector<double> x(n, 0.0);
            srand(1);
            for (int i = 0; i < rand.size(); ++i)
            {
                rand[i] = oskar_random_uniform();
                x[i] = static_cast<double>(i);
            }

            _p.push_back(new PlotWidget);
            _p.back()->plotCurve(n, x.constData(), rand.constData());
            _p.back()->setTitle("Uniform");
        }

        void gaussian()
        {
            const unsigned n = 10000;
            QVector<double> rand(n);
            QVector<double> x(n, 0.0);
            srand(1);
            for (int i = 0; i < rand.size(); i += 2)
            {
                double r2;
                rand[i    ] = oskar_random_gaussian(&r2);
                rand[i + 1] = r2;
                x[i] = static_cast<double>(i);
                x[i + 1] = static_cast<double>(i);
            }

            _p.push_back(new PlotWidget);
            _p.back()->plotCurve(n, x.constData(), rand.constData());
            _p.back()->setTitle("Gaussian");
        }

        void power_law()
        {
            const unsigned n = 10000;
            const double min = 1.0e-2;
            const double max = 1.0e4;
            const double power = -1.2;
            QVector<double> rand(n);
            QVector<double> x(n, 0.0);
            srand(1);
            for (int i = 0; i < rand.size(); ++i)
            {
                rand[i] = oskar_random_power_law(min, max, power);
                x[i] = static_cast<double>(i);
            }

            _p.push_back(new PlotWidget);
            _p.back()->plotCurve(n, x.constData(), rand.constData());
            _p.back()->setTitle("Power law");
        }

        void broken_power_law()
        {
            const unsigned n = 1250000;
            const double min = 1.0e-2;
            const double max = 1.0e4;
            const double threshold = 0.88;
            const double power1 = -2.0;
            const double power2 = -4.0;

//            QVector<double> x(n, 0.0);

            QTime t;
            t.start();
            {
                QVector<double> rand(n);
                srand(1);
                for (int i = 0; i < rand.size(); ++i)
                {
                    rand[i] = oskar_random_broken_power_law(min, max,
                            threshold, power1, power2);
                }
            }
            printf("Time taken for %d samples = %f sec.\n",
                    n, t.elapsed() / 1.0e3);

//                x[i] = static_cast<double>(i);
//            _p.push_back(new PlotWidget);
//            _p.back()->plotCurve(n, x.constData(), rand.constData());
//            _p.back()->setTitle("Broken power law");
        }

    private:
        std::vector<PlotWidget*> _p;
};


int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QTest_Random test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}


#endif // QTEST_RANDOM_H_
