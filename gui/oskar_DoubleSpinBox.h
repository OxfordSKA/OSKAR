/*
 * Copyright (c) 2012-2020, The University of Oxford
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

#ifndef OSKAR_DOUBLE_SPIN_BOX_H_
#define OSKAR_DOUBLE_SPIN_BOX_H_

#include <QAbstractSpinBox>

class QDoubleValidator;

namespace oskar {

class DoubleSpinBox : public QAbstractSpinBox
{
    Q_OBJECT

public:
    DoubleSpinBox(QWidget* parent = 0);
    QString cleanText() const;
    int decimals() const;
    void setDecimals(int prec);
    void setRange(double minimum, double maximum);
    double rangeMin() const;
    void setSingleStep(double val);
    double singleStep() const;
    virtual void stepBy(int steps);
    virtual QString textFromValue(double value) const;
    virtual QValidator::State validate(QString& text, int& pos) const;
    double value() const;
    virtual double valueFromText(const QString& text) const;
    void setMinText(const QString& text);
    QString minText() const;

public Q_SLOTS:
    void setValue(double val);
    void setValue(const QString& text);

Q_SIGNALS:
    void valueChanged(double d);
    void valueChanged(const QString& text);

protected:
    virtual void focusInEvent(QFocusEvent* event);
    virtual void keyPressEvent(QKeyEvent* event);
    virtual StepEnabled stepEnabled() const;

private Q_SLOTS:
    void setValue();

private:
    double value_;
    double max_;
    double min_;
    QString minText_;
    double singleStep_;
    int decimals_;
    QDoubleValidator* v_;
};

} /* namespace oskar */

#endif /* OSKAR_DOUBLE_SPIN_BOX_H_ */
