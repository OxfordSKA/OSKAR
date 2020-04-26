/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#include "gui/oskar_DoubleSpinBox.h"
#include "settings/oskar_settings_utility_string.h"
#include <QDoubleValidator>
#include <QFocusEvent>
#include <QKeyEvent>
#include <QLineEdit>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>

using namespace std;

namespace oskar {

DoubleSpinBox::DoubleSpinBox(QWidget* parent)
: QAbstractSpinBox(parent)
{
    value_ = 0.0;
    max_ = DBL_MAX;
    min_ = -DBL_MAX;
    decimals_ = 16;
    singleStep_ = 1.0;
    v_ = new QDoubleValidator(this);
    connect(this, SIGNAL(editingFinished()), this, SLOT(setValue()));
}

QString DoubleSpinBox::cleanText() const
{
    return textFromValue(value_);
}

int DoubleSpinBox::decimals() const
{
    return decimals_;
}

void DoubleSpinBox::setDecimals(int prec)
{
    decimals_ = prec;
}

void DoubleSpinBox::setRange(double minimum, double maximum)
{
    min_ = minimum;
    max_ = maximum;
    v_->setRange(minimum, maximum, v_->decimals());
}

double DoubleSpinBox::rangeMin() const
{
    return min_;
}

void DoubleSpinBox::setSingleStep(double val)
{
    singleStep_ = val;
}

double DoubleSpinBox::singleStep() const
{
    return singleStep_;
}

void DoubleSpinBox::stepBy(int steps)
{
    // Get cursor position and locate "e" character.
    int p = lineEdit()->cursorPosition();
    int e = text().indexOf('e', 0, Qt::CaseInsensitive);
    double val = valueFromText(text());

    // Set appropriate value.
    if (e < 0)
    {
        // Exponent not present: change normal decimal number.
        setValue(val + steps * singleStep_);

        // Set selected region.
        lineEdit()->selectAll();
    }
    else if (p < e + 1)
    {
        // Change mantissa.
        int exponent = (int) floor(log10(fabs(val)));
        double mantissa = val / pow(10.0, exponent);

        if (mantissa < 0.0)
        {
            if (steps < 0)
            {
                if (mantissa > -9.0)
                    mantissa--;
                else
                {
                    exponent++;
                    mantissa += 8.0;
                }
            }
            else
            {
                if (mantissa > -2.0)
                {
                    exponent--;
                    mantissa -= 8.0;
                }
                else
                    mantissa++;
            }
        }
        else
        {
            if (steps < 0)
            {
                if (mantissa < 2.0)
                {
                    exponent--;
                    mantissa += 8.0;
                }
                else
                    mantissa--;
            }
            else
            {
                if (mantissa < 9.0)
                    mantissa++;
                else
                {
                    exponent++;
                    mantissa -= 8.0;
                }
            }
        }
        setValue(mantissa * pow(10.0, exponent));

        // Set selected region.
        int e = text().indexOf('e', 0, Qt::CaseInsensitive);
        lineEdit()->setSelection(0, e);
    }
    else
    {
        // Change exponent only.
        setValue((steps < 0) ? val / 10.0 : val * 10.0);

        // Set selected region.
        int e = text().indexOf('e', 0, Qt::CaseInsensitive);
        lineEdit()->setSelection(e + 1, text().length() - (e + 1));
    }
}

QString DoubleSpinBox::textFromValue(double value) const
{
    QString t;
    if (value <= min_ && !minText_.isEmpty())
    {
        t = minText_;
    }
    else if (text().contains('e', Qt::CaseInsensitive))
    {
        std::string s = oskar_settings_utility_double_to_string_2(value, 'e');
        t = QString::fromStdString(s);
    }
    else
    {
        std::string s = oskar_settings_utility_double_to_string_2(value, 'g');
        t = QString::fromStdString(s);
    }

    return t;
}

QValidator::State DoubleSpinBox::validate(QString& text, int& pos) const
{

    QValidator::State state = QValidator::Invalid;

    // If the minimum text is set allow typing it.
    if (!minText_.isEmpty() && minText_.startsWith(text, Qt::CaseInsensitive))
    {
        if (text.compare(minText_, Qt::CaseInsensitive) == 0)
        {
            state = QValidator::Acceptable;
        }
        else
        {
            state = QValidator::Intermediate;
        }
    }
    else
    {
        state = v_->validate(text, pos);
    }

    return state;
}

double DoubleSpinBox::value() const
{
    return value_;
}

double DoubleSpinBox::valueFromText(const QString& text) const
{
    if (!minText_.isEmpty() && text.compare(minText_, Qt::CaseInsensitive) == 0)
        return min_;
    else {
        double value =  text.toDouble();
        return value;
    }
}

void DoubleSpinBox::setMinText(const QString& text)
{
    minText_ = text;
}

QString DoubleSpinBox::minText() const
{
    return minText_;
}

void DoubleSpinBox::setValue(double val)
{
    value_ = val;
    QString t = textFromValue(val);
    lineEdit()->setText(t);
    emit valueChanged(val);
    emit valueChanged(t);
}

void DoubleSpinBox::setValue(const QString& text)
{
    QVariant var(text);
    if (!var.canConvert(QVariant::Double) && text != minText_)
        return;
    lineEdit()->setText(text);
    value_ = var.toDouble();
    emit valueChanged(value_);
    emit valueChanged(text);
}

void DoubleSpinBox::setValue()
{
    setValue(valueFromText(text()));
}

// Protected functions.

void DoubleSpinBox::focusInEvent(QFocusEvent* event)
{
    lineEdit()->selectAll();
    QAbstractSpinBox::focusInEvent(event);
}

void DoubleSpinBox::keyPressEvent(QKeyEvent* event)
{
    QAbstractSpinBox::keyPressEvent(event);
    int pos = 0;
    QString txt = text();
    if (validate(txt, pos) == QValidator::Acceptable)
        value_ = valueFromText(txt);
}

QAbstractSpinBox::StepEnabled DoubleSpinBox::stepEnabled() const
{
    return QAbstractSpinBox::StepUpEnabled | QAbstractSpinBox::StepDownEnabled;
}

} /* namespace oskar */
