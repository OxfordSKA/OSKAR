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

#ifndef IMAGE_PLOT_H_
#define IMAGE_PLOT_H_

/**
 * @file ImagePlot.h
 */

#include "widgets/plotting/ImagePlotData.h"

#include <qwt_plot_spectrogram.h>

#include <vector>

namespace oskar {

/**
 * @class ImagePlot
 *
 * @brief
 * Image plot type class for use with QWT.
 *
 * @details
 */

class ImagePlot : public QwtPlotSpectrogram
{
    public:
        /// Constructs an image plot object.
        ImagePlot() {};

        /// Destroys an image plot object.
        virtual ~ImagePlot() {}

    public:
        /// Set Image plot data.
        void setImageArray(const float* data, unsigned nX, unsigned nY,
                double xmin, double xmax, double ymin, double ymax);

        /// Set the amplitude range.
        void setAmpRange(double min, double max);

        /// Returns the amplitude range.
        QwtDoubleInterval& ampRange() { return _range; }

        /// Returns the amplitude range.
        const QwtDoubleInterval& ampRange() const { return _range; }

        /// Set the amplitude range to auto.
        void setAmpRangeAuto();

        /// Show contours on the image plot.
        void showContours(bool on);

        /// Set the image display mode.
        void setDisplayImage(bool on);

        /// Setup contour levels.
        void setupContours(unsigned nLevels);

    private:
        ImagePlotData _data;        ///< Image plot data object.
        QwtDoubleInterval _range;   ///< Image plot amplitude range.
};

} // namespace oskar
#endif // IMAGE_PLOT_H_
