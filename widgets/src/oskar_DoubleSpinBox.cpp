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

#include "widgets/oskar_DoubleSpinBox.h"
#include <QtGui/QDoubleValidator>
#include <QtGui/QLineEdit>
#include <cfloat>
#include <cmath>
#include <cstdio>

#define round(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

oskar_DoubleSpinBox::oskar_DoubleSpinBox(QWidget* parent)
: QAbstractSpinBox(parent)
{
    value_ = 0.0;
    max_ = DBL_MAX;
    min_ = -DBL_MAX;
    decimals_ = 3;
    singleStep_ = 1.0;
    v_ = new QDoubleValidator(this);
    connect(this, SIGNAL(editingFinished()), this, SLOT(setValue()));
}

QString oskar_DoubleSpinBox::cleanText() const
{
    return textFromValue(value_);
}

int oskar_DoubleSpinBox::decimals() const
{
    return decimals_;
}

void oskar_DoubleSpinBox::setDecimals(int prec)
{
    decimals_ = prec;
}

void oskar_DoubleSpinBox::setRange(double minimum, double maximum)
{
    min_ = minimum;
    max_ = maximum;
    v_->setRange(minimum, maximum, v_->decimals());
}

void oskar_DoubleSpinBox::setSingleStep(double val)
{
    singleStep_ = val;
}

double oskar_DoubleSpinBox::singleStep() const
{
    return singleStep_;
}

void oskar_DoubleSpinBox::stepBy(int steps)
{
    // Get cursor position and locate "e" character.
    int p = lineEdit()->cursorPosition();
    int e = 1 + text().indexOf('e', 0, Qt::CaseInsensitive);

    // Set appropriate value.
    if (e < 0)
        setValue(value_ + steps * singleStep_);
    else if (p < e)
    {
        double a = log10(fabs(value_));
        int power = (int)a;
        if (a < 0.0) --power;
        double f = pow(10.0, power);
        //printf("a = %f, power = %d, f = %f\n", a, power, f);
        if (fabs(f) > fabs(value_ + steps * singleStep_)) f /= 10.0;
        setValue(value_ + steps * singleStep_ * f);
    }
    else
        setValue((steps < 0) ? value_ / 10.0 : value_ * 10.0);

    // Restore the cursor position.
    lineEdit()->setCursorPosition(p);
}

QString oskar_DoubleSpinBox::textFromValue(double value) const
{
    if (text().contains('e', Qt::CaseInsensitive))
    {
        return QString::number(value, 'e', decimals());
    }
    else if (value != 0.0 && (fabs(value) < pow(10.0, -decimals()) ||
            fabs(value) > pow(10.0, decimals())))
    {
        return QString::number(value, 'e', decimals());
    }
    return QString::number(value, 'f', decimals());
}

QValidator::State oskar_DoubleSpinBox::validate(QString& text, int& pos) const
{
    return v_->validate(text, pos);
}

double oskar_DoubleSpinBox::value() const
{
    return value_;
}

double oskar_DoubleSpinBox::valueFromText(const QString& text) const
{
    return text.toDouble();
}

void oskar_DoubleSpinBox::setValue(double val)
{
    value_ = val;
    QString t = textFromValue(val);
    lineEdit()->setText(t);
    emit valueChanged(val);
    emit valueChanged(t);
}

QAbstractSpinBox::StepEnabled oskar_DoubleSpinBox::stepEnabled() const
{
    return QAbstractSpinBox::StepUpEnabled | QAbstractSpinBox::StepDownEnabled;
}

void oskar_DoubleSpinBox::setValue()
{
    setValue(valueFromText(text()));
}
