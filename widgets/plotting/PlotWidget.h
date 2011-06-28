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

#ifndef PLOT_WIDGET_H
#define PLOT_WIDGET_H

/**
 * @file PlotWidget.h
 */

#include "widgets/plotting/ImagePlot.h"
#include "widgets/plotting/ColourMap.h"

#include <QtCore/QObject>
#include <QtCore/QString>

#include <qwt_plot.h>
#include <qwt_plot_curve.h>

#include <complex>
#include <vector>

using std::vector;

class QMenu;
class QAction;
class QwtScaleWidget;
class QwtPlotGrid;

namespace oskar {

class PlotPicker;
class PlotMagnifier;
class PlotZoomer;
class PlotPanner;

/**
 * @class PlotWidget
 *
 * @brief
 *
 * @details
 */

class PlotWidget : public QwtPlot
{
    Q_OBJECT

    public:
        /// Constructs a plot handler object
        PlotWidget(QWidget * parent = 0);

        /// Destroys a plot handler object
        virtual ~PlotWidget();

    public:
        typedef std::complex<float> Complex;

        typedef enum { RE, IM, ABS, PHASE, SQRT } c_type;

    public:
        /// Clear the plot Canvas
        void clear();

        /// Plot a curve.
        void plotCurve(const unsigned nPoints, double const * x,
                double const * y, const bool reverseX = false,
                const bool reverseY = false, bool append = false);

        /// Plot an image
        void plotImage(float const * data, unsigned nX, unsigned nY,
                double xMin, double xMax, double yMin, double yMax,
                bool reverseX = false, bool reverseY = false);

        /// Plot an image
        void plotImage(const unsigned size, float const * data,
                const QString & title = "");

        /// Plot an image
        void plotImage(const unsigned size, const Complex * data,
                const c_type type = RE, const QString & title = "");

    public:
        /// Return reference to the curve plot
        const QwtPlotCurve & curve() const { return *_curve[0]; }

        /// Returns a reference to the image plot.
        const ImagePlot& image() const { return _image; }

        /// Returns a reference to the colour map.
        ColourMap & colourMap() { return _colourMap; }

        /// Returns a reference to the colour map (const overload.)
        const ColourMap & colourMap() const { return _colourMap; }

    public slots:
        /// Set the plot title
        void setPlotTitle(const QString&);

        /// Sets the x axis label
        void setXLabel(const QString&);

        /// Set the x label via a input pop-up.
        void menuSetXLabel();

        /// Set the y label via a input pop-up.
        void menuSetYLabel();

        /// Sets the y axis label
        void setYLabel(const QString&);

        /// Sets the scale label.
        void setScaleLabel(const QString&);

        /// Point selected
        void slotSelectedPoint(const QwtDoublePoint&);

        /// Rectangle selected
        void slotSelectedRect(const QwtDoubleRect&);

        /// Set the menu title
        void menuSetTitle();

        /// Sets the state of axis visibility
        void setAxesVisible(int state);

        /// Show the plot grid.
        void showGrid(bool);

        /// Show minor grid ticks.
        void showGridMinorTicks(bool);

        /// Toggle the grid on and off.
        void toggleGrid();

        /// Toggle minor grid ticks.
        void toggleGridMinorTicks();

        /// Save the plot widget as a PNG image
        void savePNG(QString fileName = "", unsigned sizeX = 500,
                unsigned sizeY = 500);

        /// Save the plot widget as a PDF.
        void savePDF(QString fileName = "", unsigned sizeX = 500,
                unsigned sizeY = 500);

        /// Toggle display of image contours
        void showContours(bool on) { _image.showContours(on); }

        /// Toggle display of the image plot.
        void showImage(bool on) { _image.setDisplayImage(on); }

        /// Setup image contours
        void setupContours(unsigned number)
        { _image.setupContours(number); }

        /// Update the zoom base.
        void updateZoomBase();

        /// Set the picker selection type
        void rectangleSelect(bool enable = true);

        /// Set the plot curve style.
        void setCurveStyle(const unsigned iCurve);

        /// Set the colour scale object
        void enableColourScale(bool on = true);

        /// Toggle plot panning.
        void togglePanner(bool);

        /// Toggle plot zooming.
        void toggleZoomer(bool);

        /// Toggle plot position tracking.
        void togglePicker(bool);

    signals:
        /// Signal emitted when plot title changes
        void titleChanged(const QString&);

        /// Signal emitted when the x axis label changes
        void xLabelChanged(const QString&);

        /// Signal emitted when the y axis label changes
        void yLabelChanged(const QString&);

        /// Signal emitted when the y axis label changes
        void scaleLabelChanged(const QString&);

        /// Emitted when grid is enabled
        void gridEnabled(bool);

        /// Signal for toggle showing minor grid ticks
        void gridMinorTicks(bool);

        /// Signal for toggle showing plot axes
        void axesVisible(bool);

        /// Signal emitted on a picker selection rectangle event
        void xLeftChanged(double);

        /// Signal emitted on a picker selection rectangle event
        void xRightChanged(double);

        /// Signal emitted on a picker selection rectangle event
        void yBottomChanged(double);

        /// Signal emitted on a picker selection rectangle event
        void yTopChanged(double);

    protected:
        /// Function to handle mouse press events.
        void mousePressEvent(QMouseEvent*);

        /// Function to handle mouse move events.
        void mouseMoveEvent(QMouseEvent*);

        /// Function to handle mouse release events.
        void mouseReleaseEvent(QMouseEvent*);

        /// Function to handle key presses.
        void keyPressEvent(QKeyEvent*);

    private:
        /// Set the plot grid object
        void _setGrid();

        /// Set the plot panning object
        void _setPanner();

        /// Setup the Zoomer.
        void _setZoomer();

        /// Setup signal and slot connections
        void _connectSignalsAndSlots();

    private:
        vector<QwtPlotCurve*> _curve;
        ImagePlot _image;

        ColourMap _colourMap;

        bool _adjustingColours;
        int _oldx;
        int _oldy;

        PlotPicker * _picker;
        PlotPanner * _panner;
        PlotZoomer * _zoomer;
        PlotMagnifier * _magnifier;

        QwtScaleWidget * _scale;
        QwtPlotGrid* _grid;
};

} // namespace oskar
#endif // PLOT_WIDGET_H
