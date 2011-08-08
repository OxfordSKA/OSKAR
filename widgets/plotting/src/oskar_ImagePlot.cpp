#include "widgets/plotting/oskar_ImagePlot.h"

#include <QtCore/QString>
#include <QtGui/QColor>
#include <QtGui/QPen>

#include <cmath>
#include <limits>
#include <iostream>

using namespace std;

namespace oskar {


/**
 * @details
 * Set the data to plot as an image.
 *
 * @param[in]   data    Vector of image data to plot.
 * @param[in]   nx   Size of the x dimension of the image.
 * @param[in]   ny   Size of the y dimension of the image.
 */
void ImagePlot::setImageArray(const float * data, unsigned nx, unsigned ny,
        double xmin, double xmax, double ymin, double ymax)
{
    if (data == 0)
    {
        cerr << "ImagePlot::setImageArray(): Input data error." << endl;
        return;
    }

    _data.setRangeX(xmin, xmax);
    _data.setRangeY(ymin, ymax);
    _data.setData(data, nx, ny);
    setData(_data);
    setAmpRangeAuto();
}


/**
 * @details
 * Automatically set the image amplitude range.
 */
void ImagePlot::setAmpRangeAuto()
{
    _range = data().range();

    // If the min == max add a small amount the range to fix a number of errors.
    double min = _range.minValue();
    double max = _range.maxValue();
    if (fabs(max - min) < numeric_limits<double>::epsilon())
        _range = QwtDoubleInterval(min, min + 1.0e-6);
}


/**
 * @details
 * Set the display of image contours.
 *
 * @param[in]   on  Bool flag controlling display of coutours.
 */
void ImagePlot::showContours(const bool on)
{
    setDisplayMode(QwtPlotSpectrogram::ContourMode, on);
}


/**
 * @details
 * Set the display of the image plot.
 *
 * @param[in]   on  Bool flag controlling display of the image.
 */
void ImagePlot::setDisplayImage(const bool on)
{
    setDisplayMode(QwtPlotSpectrogram::ImageMode, on);
    setDefaultContourPen(on ? QPen() : QPen(Qt::NoPen));
}


/**
 * @details
 * Setup a numbeer of linearly spaced contour levels.
 *
 * @param[in]   nlevels   Number of contours levels to plot.
 */
void ImagePlot::setupContours(const unsigned nlevels)
{
    double range = _range.width();
    double step = range / double(nlevels + 1);
    QwtValueList contourLevels;
    for (double l = step; l < range; l += step) contourLevels += l;
    setContourLevels(contourLevels);
}


} // namespace oskar
