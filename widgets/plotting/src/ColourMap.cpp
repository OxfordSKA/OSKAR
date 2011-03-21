#include "widgets/plotting/ColourMap.h"

#include <cmath>
#include <iostream>

namespace oskar {

/**
 * @details
 * Sets the image plot colour map based on selected brightness, contrast and
 * base map type.
 */
void ColourMap::update(float brightness, float contrast)
{
    _brightness = brightness;
    _contrast = contrast;
    _colour.clear();
    _pos.clear();

    switch (_type)
    {
        case RAINBOW:  _rainbow(); break;
        case HEAT:     _heat(); break;
        case GRAY:     _gray(); break;
        case GRAY_INV: _grayInv(); break;
        case JET:      _jet(); break;
        case STANDARD: _standard(); break;
        default: throw QString("ColourMap::update(): Unknown colour map.");
    }

    unsigned nSteps = 32;
    unsigned nColours = (unsigned)_colour.size();
    Q_ASSERT(_colour.size() == _pos.size());

    QColor colour;

    float minContrast = 1.0f / float(nSteps);
    if (std::fabs(contrast) < minContrast)
    {
        _contrast = (_contrast > 0.0f) ? minContrast : -minContrast;
    }

    // Convert contrast to the normalised stretch of the
    // colour table across the available colour index range.
    float span = 1.0f / std::fabs(_contrast);

    // Translate from brightness and contrast to the colour index
    // coordinates at start and end of colour table, cStart and cEnd
    float cStart = 0.0f;
    float cEnd = 0.0f;
    if (contrast >= 0.0f)
    {
        cStart = 1.0f - _brightness * (span + 1.0f);
        cEnd = cStart + span;
    }
    else
    {
        cStart = _brightness * (span + 1.0f);
        cEnd = cStart - span;
    }

    // Determine the number of colour indexes spanned by the colour table
    int nSpan = int(span * float(nSteps));

    // Determine the direction in which the colour table should be traversed
    bool forward = (cStart <= cEnd);

    // Initialise the indexes at which to start searching the colour table
    int below = nColours - 1;
    unsigned above = 0;

    // Linearly interpolate the colour table RGB values onto each colour index
    _colours.clear();
    _colours.resize(nSteps);
    _position.resize(nSteps);
    float level = 0.0f;
    float rb, ra, gb, ga, bb, ba, red, green, blue;
    for (unsigned c = 0; c < nSteps; c++)
    {

        // Turn the colour index into a fraction of the range
        float iColourFraction = float(c) / float(nSteps);

        // Determine the colour table position that corresponds to colour index
        if (nSpan > 0)
            level = (iColourFraction - cStart) / (cEnd - cStart);
        else
        {
            if (iColourFraction <= cStart) level = 0.0f;
            else level = 1.0f;
        }

        // Search for the indexes of the two colour table entries that straddle
        // 'level', assuming that values in L() always increase
        if (forward)
        {
            while (above <= (nColours - 1) && _pos[above] < level) above++;
            below = above - 1;
        }
        else
        {
            while (below >= 0 && _pos[below] > level) below--;
            above = below + 1;
        }

        // If the indexes lie outside the table, substitute the index of the
        // nearest edge of the table
        if (below < 0)
        {
            level = 0;
            below = 0;
            above = 0;
        }
        else if (above >= nColours)
        {
            level = 1;
            below = nColours - 1;
            above = nColours - 1;
        }

        // Linearly interpolate the primary colour intensities from colour table
        float posDiff = _pos[above] - _pos[below];
        float posFrac;

        if (posDiff > minContrast) posFrac = (level - _pos[below]) / posDiff;
        else posFrac = 0.0f;

        rb = (float)_colour[below].redF();
        ra = (float)_colour[above].redF();
        gb = (float)_colour[below].greenF();
        ga = (float)_colour[above].greenF();
        bb = (float)_colour[below].blueF();
        ba = (float)_colour[above].blueF();
        red = rb + (ra - rb) * posFrac;
        green = gb + (ga - gb) * posFrac;
        blue = bb + (ba - bb)  * posFrac;

        // Clamp between 0 and 1
        if (red < 0) red = 0;
        if (red > 1) red = 1;
        if (green < 0) green = 0;
        if (green > 1) green = 1;
        if (blue < 0) blue = 0;
        if (blue > 1) blue = 1;

        // Store the colour
        colour.setRgbF(red, green, blue);
        _colours[c] = colour;
        _position[c] = iColourFraction;
    }

    _colourMap = QwtLinearColorMap(_colours[0], _colours[nSteps-1]);

    for (unsigned ic = 1; ic < nSteps - 2; ++ic)
        _colourMap.addColorStop((double)_position[ic], _colours[ic]);

    emit mapUpdated();
}


/**
 * @param type
 */
void ColourMap::set(ColourMapType type)
{
    _type = type;
    update();
    emit mapTypeChanged(int(type));
}


void ColourMap::set(int type)
{
    _type = ColourMapType(type);
    update();
    emit mapTypeChanged(type);
}


/**
 * @details
 * Set the rainbow colour map defaults
 */
void ColourMap::_rainbow()
{
    _colour.resize(9);
    _colour[0] = Qt::black;
    _colour[1].setRgbF(0.0, 0.0, 0.3);
    _colour[2].setRgbF(0.0, 0.0, 0.8);
    _colour[3].setRgbF(0.0, 1.0, 1.0);
    _colour[4].setRgbF(0.6, 1.0, 0.3);
    _colour[5].setRgbF(1.0, 1.0, 0.0);
    _colour[6].setRgbF(1.0, 0.6, 0.0);
    _colour[7].setRgbF(1.0, 0.0, 0.0);
    _colour[8] = Qt::white;
    _pos.resize(9);
    _pos[0] = -0.50f;
    _pos[1] =  0.00f;
    _pos[2] =  0.17f;
    _pos[3] =  0.33f;
    _pos[4] =  0.50f;
    _pos[5] =  0.67f;
    _pos[6] =  0.83f;
    _pos[7] =  1.00f;
    _pos[8] =  1.70f;
}


/**
 * @details
 * Set the gray colour map defaults
 */
void ColourMap::_gray()
{
    _colour.resize(3);
    _colour[0] = Qt::black;
    _colour[1] = Qt::white;
    _colour[2] = Qt::white;
    _pos.resize(3);
    _pos[0] = 0.0f;
    _pos[1] = _brightness;
    _pos[2] = 1.0f;
}


/**
 * @details
 * Set the inverted gray colour map defaults
 */
void ColourMap::_grayInv()
{
    _colour.resize(3);
    _colour[0] = Qt::white;
    _colour[1] = Qt::black;
    _colour[2] = Qt::black;
    _pos.resize(3);
    _pos[0] = 0.0f;
    _pos[1] = _brightness;
    _pos[2] = 1.0f;
}


/**
 * @details
 * set the heat colour map defaults
 */
void ColourMap::_heat()
{
    _colour.resize(5);
    _colour[0].setRgbF(0.0, 0.0, 0.0);
    _colour[1].setRgbF(0.5, 0.0, 0.0);
    _colour[2].setRgbF(1.0, 0.5, 0.0);
    _colour[3].setRgbF(1.0, 1.0, 0.3);
    _colour[4].setRgbF(1.0, 1.0, 1.0);
    _pos.resize(5);
    _pos[0] = 0.0f;
    _pos[1] = 0.2f;
    _pos[2] = 0.4f;
    _pos[3] = 0.6f;
    _pos[4] = 1.0f;
}


/**
 * @details
 * Sets the jet colour map defaults
 */
void ColourMap::_jet()
{
    _colour.resize(14);
    _colour[0].setRgb(0, 0, 189);
    _colour[1].setRgb(0, 0, 255);
    _colour[2].setRgb(0, 66, 255);
    _colour[3].setRgb(0, 132, 255);
    _colour[4].setRgb(0, 189, 255);
    _colour[5].setRgb(0, 255, 255);
    _colour[6].setRgb(66, 255, 189);
    _colour[7].setRgb(132, 255, 132);
    _colour[8].setRgb(189, 255, 66);
    _colour[9].setRgb(255, 255, 0);
    _colour[10].setRgb(255, 189,0);
    _colour[11].setRgb(255, 66, 0);
    _colour[12].setRgb(189, 0, 0);
    _colour[13].setRgb(132, 0, 0);
    _pos.resize(14);
    for (int i = 0; i < 14; ++i) _pos[i] = float(i) / 13.0f;
}


/**
 * @details
 * Sets the Qwt 'standard' colour map defaults
 */
void ColourMap::_standard()
{
    _colour.resize(5);
    _colour[0] = Qt::darkCyan;
    _colour[1] = Qt::cyan;
    _colour[2] = Qt::green;
    _colour[3] = Qt::yellow;
    _colour[4] = Qt::red;
    _pos.resize(5);
    _pos[0] = 0.00f;
    _pos[1] = 0.10f;
    _pos[2] = 0.60f;
    _pos[3] = 0.95f;
    _pos[4] = 1.00f;
}


} // namespace oskar.
