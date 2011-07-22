#include "widgets/plotting/ImagePlotData.h"
#include "math/FloatingPointCompare.h"

#include <QtCore/QString>

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdio>

using namespace std;

namespace oskar {

/**
 * @details
 */
void ImagePlotData::setData(const float* data, const unsigned nX, const unsigned nY)
{
    if (!data) throw QString("ImagePlotData::setData(): Data not allocated");

    _sizeX = nX;
    _sizeY = nY;
    _data.resize(nX * nY);
    memcpy(&_data[0], data, nX * nY * sizeof(float));

    _findAmpRange();

    _intervalX = _xRange.width() / (double)_sizeX;
    _intervalY = _yRange.width() / (double)_sizeY;
}


/**
 * @details
 */
QwtRasterData* ImagePlotData::copy() const
{
    ImagePlotData* clone = new ImagePlotData();
    clone->setRangeX(_xRange.minValue(), _xRange.maxValue());
    clone->setRangeY(_yRange.minValue(), _yRange.maxValue());
    clone->setBoundingRect(QwtDoubleRect(_xRange.minValue(),
            _yRange.minValue(), _xRange.width(), _yRange.width()));
    clone->setData(&_data[0], _sizeX, _sizeY);
    return clone;
}


/**
 * @details
 * If you have a colour map from blue to red you also need to define which value
 * means blue and which means red. Values above or below are mapped to the
 * borders of the colour map.
 */
QwtDoubleInterval ImagePlotData::range() const
{
    double min = _ampRange.minValue();
    double max = _ampRange.maxValue();
    return QwtDoubleInterval(min, max);
}


/**
 * @details
 */
void ImagePlotData::setRangeX(double min, double max)
{
    _xRange.setMinValue(min);
    _xRange.setMaxValue(max);
}


/**
 * @details
 */
void ImagePlotData::setRangeY(double min, double max)
{
    _yRange.setMinValue(min);
    _yRange.setMaxValue(max);
}


/**
 * @details
 * Returns value at a specific position in the raster.
 */
double ImagePlotData::value(double x, double y) const
{
    if (x > _xRange.maxValue() || x < _xRange.minValue()) return 0.0;
    if (y > _yRange.maxValue() || y < _yRange.minValue()) return 0.0;

    unsigned col = unsigned((x - _xRange.minValue()) / _intervalX);
    if (col >= _sizeX) col--;

    unsigned row = unsigned((y - _yRange.minValue()) / _intervalY);
    if (row >= _sizeY) row--;

    return _data[row * _sizeX + col];
}


int ImagePlotData::column(double x) const
{
    if (x > _xRange.maxValue() || x < _xRange.minValue()) return -1;
    int col = int((x - _xRange.minValue()) / _intervalX);
    if (col >= (int)_sizeX) col--;
    return col;
}



int ImagePlotData::row(double y) const
{
    if (y > _yRange.maxValue() || y < _yRange.minValue()) return -1;
    int row = int((y - _yRange.minValue()) / _intervalY);
    if (row >= (int)_sizeY) row--;
    return row;
}




/**
 * @details
 */
void ImagePlotData::_findAmpRange()
{
    double min = numeric_limits<double>::max();
    double max = -min;

    for (unsigned i = 0; i < _sizeX * _sizeY; ++i)
    {
        min = std::min<double>(_data[i], min);
        max = std::max<double>(_data[i], max);
    }

    if (approxEqual(min, max)) max = min + 1.0e-6;

    _ampRange.setMaxValue(max);
    _ampRange.setMinValue(min);
}


} // namespace oskar
