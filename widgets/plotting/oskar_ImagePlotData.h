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

#ifndef IMAGE_PLOT_DATA_H
#define IMAGE_PLOT_DATA_H

/**
* @file oskar_imagePlotData.h
*/

#include <qwt_raster_data.h>
#include <vector>

class QwtRasterData;
class QwtDoubleInterval;

namespace oskar {

/**
* @class ImagePlotData
*
* @brief
*
* @details
* QwtRasterData (similar to QwtData) is the link between the application data
* and the QwtRasterItem (guess here a spectrogram), that is responsible for
* displaying the data.
*
* When the raster item needs to render its image it calculates the position of
* each pixel and asks the raster data object for a value at this position. This
* value is mapped into a colour using the value range, that is also provided by
* the data object.
*/

class ImagePlotData: public QwtRasterData
{
    public:
        /// Constructs an image plot data object.
        ImagePlotData() : QwtRasterData() {}

        /// Destroys the image plot data object.
        ~ImagePlotData() {}

    public:
        /// Set the data to be held in the raster.
        void setData(float const * data, const unsigned nX, const unsigned nY);

        /// Clone the data in the raster.
        QwtRasterData* copy() const;

        /// Set the x range to display.
        void setRangeX(double min, double max);

        /// Set the y range to display.
        void setRangeY(double min, double max);

        int column(double x) const;
        int row(double y) const;


        QwtDoubleInterval xRange() const { return _xRange; }
        QwtDoubleInterval yRange() const { return _yRange; }
        QwtDoubleInterval ampRange() const { return _ampRange; }

        unsigned sizeX() const { return _sizeX; }
        unsigned sizeY() const { return _sizeY; }

        double intervalX() const { return _intervalX; }
        double intervalY() const { return _intervalY; }

    public:
        /// Return the value at a raster position (override of pure virtual
        /// method).
        double value(double x, double y) const;

        /// Return the range of values in the raster.
        QwtDoubleInterval range() const;

        /// Calculates the amplitude range of the image data.
        void _findAmpRange();

    private:
        std::vector<float> _data;
        unsigned _sizeX;
        unsigned _sizeY;
        double _intervalX;
        double _intervalY;
        QwtDoubleInterval _ampRange;
        QwtDoubleInterval _xRange;
        QwtDoubleInterval _yRange;
};

} // namespace oskar
#endif // IMAGE_PLOT_DATA_H
