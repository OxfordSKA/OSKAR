#include "widgets/plotting/ImagePlot.h"

#include "math/core/FloatingPointCompare.h"
#include <QtCore/QString>
#include <QtGui/QColor>
#include <QtGui/QPen>

#include <iostream>
using namespace std;

namespace oskar {


/**
 * @details
 * Set the data to plot as an image.
 *
 * @param[in]   data    Vector of image data to plot.
 * @param[in]   sizex   Size of the x dimension of the image.
 * @param[in]   sizey   Size of the y dimension of the image.
 */
void ImagePlot::setImageArray(const float* data, unsigned nX, unsigned nY,
        double xmin, double xmax, double ymin, double ymax)
{
    if (!data)
        throw QString("ImagePlot::setImageArray(): Data error.");

    _data.setData(data, nX, nY);
    _data.setRangeX(xmin, xmax);
    _data.setRangeY(ymin, ymax);
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

    //cout << "amp range = " << _range.minValue() << " " << _range.maxValue() << endl;

    // If the min = max add 0.1 to the min range
    if (approxEqual(_range.minValue(), _range.maxValue()))
    {
        _range = QwtDoubleInterval(_range.minValue(), _range.minValue() + 1.0e-6);
    }
}


/**
 * @details
 * Set the display of image contours.
 *
 * @param[in]   on  Bool flag controlling display of coutours.
 */
void ImagePlot::showContours(bool on)
{
    setDisplayMode(QwtPlotSpectrogram::ContourMode, on);
}


/**
 * @details
 * Set the display of the image plot.
 *
 * @param[in]   on  Bool flag controlling display of the image.
 */
void ImagePlot::setDisplayImage(bool on)
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
void ImagePlot::setupContours(unsigned nlevels)
{
    double range = _range.width();
    double step = range / double(nlevels + 1);
    QwtValueList contourLevels;
    for (double l = step; l < range; l += step) contourLevels += l;
    setContourLevels(contourLevels);
}


} // namespace oskar
