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

#ifndef COLOUR_MAP_H_
#define COLOUR_MAP_H_

/**
 * @file oskar_ColourMap.h
 */

#include <qwt_color_map.h>

#include <QtCore/QObject>
#include <QtCore/QStringList>
#include <QtCore/QString>
#include <QtCore/QObject>

#include <vector>

using std::vector;

namespace oskar {

/**
 * @class ColourMap
 *
 * @brief
 *
 * @details
 */

class ColourMap : public QObject
{
    Q_OBJECT

    public:
        /// Enumeration: Image colour map.
        enum ColourMapType
        { RAINBOW, HEAT, GRAY, GRAY_INV, JET, STANDARD };

    public:
        ColourMap() {}

        virtual ~ColourMap() {};

    public:
        /// Returns the colour map brightness
        float brightness() const
        { return _brightness; }

        /// Returns the colour map contrast
        float contrast() const
        { return _contrast; }

        /// Returns the colour map.
        const QwtLinearColorMap& map() const
        { return _colourMap; }

    signals:
        /// Signal emitted when the colour map is changed.
        void mapUpdated();

        /// Signal emitted when the colour map type is changed.
        void mapTypeChanged(int);

    public slots:
        /// Update the colour map setting the brightness and contrast.
        void update(float brightness = 0.5f, float constrast = 1.0f);

        /// Set the colour map.
        void set(ColourMapType type = RAINBOW);

        /// Sets the colour map type.
        void set(int type);

    public:
        /// Return a string list of the colour maps supported.
        static QStringList mapTypes()
        {
            QStringList types;
            types << "Rainbow" << "Heat" << "Gray" << "Gray [inverted]";
            types << "Jet" << "Standard";
            return types;
        }

    private:
        /// Set the rainbow colour map.
        void _rainbow();

        /// Set the gray colour map.
        void _gray();

        /// Set the inverted gray colour map.
        void _grayInv();

        /// Set the heat colour map.
        void _heat();

        /// Set Jet colour map.
        void _jet();

        /// Set the standard Qwt image style colour map.
        void _standard();

    private:
        vector<QColor> _colours;
        vector<float> _position;
        ColourMapType _type;
        QwtLinearColorMap _colourMap;

        vector<QColor> _colour;
        vector<float> _pos;

        float _brightness;
        float _contrast;
};


} // namespace oskar
#endif // COLOUR_MAP_H_
