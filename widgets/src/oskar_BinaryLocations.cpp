/*
 * Copyright (c) 2012, The University of Oxford
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

#include "oskar_global.h"
#include "widgets/oskar_BinaryLocations.h"

#include <QtGui/QApplication>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QFileDialog>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QGroupBox>
#include <QtGui/QGridLayout>
#include <QtGui/QVBoxLayout>

oskar_BinaryLocations::oskar_BinaryLocations(QWidget *parent) : QDialog(parent)
{
    // Set up the GUI.
    setWindowTitle("Binary Locations");
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);
    if (parent)
        resize(int(0.85 * parent->width()), 0);

    // Create binary location group.
    QGroupBox* grp = new QGroupBox("OSKAR Binary Locations", this);
    QGridLayout* gridLayout = new QGridLayout(grp);

    // Create widgets and add to layout.
    QLabel* labelInterferometer = new QLabel("Interferometer binary:", this);
    editInterferometer_ = new QLineEdit(this);
    QPushButton* browseInterferometer = new QPushButton("Browse...", this);
    connect(browseInterferometer, SIGNAL(clicked()),
            this, SLOT(setInterferometer()));
    gridLayout->addWidget(labelInterferometer, 0, 0);
    gridLayout->addWidget(editInterferometer_, 0, 1);
    gridLayout->addWidget(browseInterferometer, 0, 2);

    QLabel* labelBeamPattern = new QLabel("Beam pattern binary:", this);
    editBeamPattern_ = new QLineEdit(this);
    QPushButton* browseBeamPattern = new QPushButton("Browse...", this);
    connect(browseBeamPattern, SIGNAL(clicked()), this, SLOT(setBeamPattern()));
    gridLayout->addWidget(labelBeamPattern, 1, 0);
    gridLayout->addWidget(editBeamPattern_, 1, 1);
    gridLayout->addWidget(browseBeamPattern, 1, 2);

    QLabel* labelImager = new QLabel("Imager binary:", this);
    editImager_ = new QLineEdit(this);
    QPushButton* browseImager = new QPushButton("Browse...", this);
    connect(browseImager, SIGNAL(clicked()), this, SLOT(setImager()));
    gridLayout->addWidget(labelImager, 2, 0);
    gridLayout->addWidget(editImager_, 2, 1);
    gridLayout->addWidget(browseImager, 2, 2);

    // Add binary location group.
    vLayoutMain->addWidget(grp);
    vLayoutMain->addStretch();

    // Create close button.
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok |
            QDialogButtonBox::Cancel, Qt::Horizontal, this);
    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));
    vLayoutMain->addWidget(buttons);
}

QString oskar_BinaryLocations::beamPattern() const
{
    return editBeamPattern_->text();
}

QString oskar_BinaryLocations::imager() const
{
    return editImager_->text();
}

QString oskar_BinaryLocations::interferometer() const
{
    return editInterferometer_->text();
}

void oskar_BinaryLocations::setBeamPattern(const QString& value)
{
    editBeamPattern_->setText(value);
}

void oskar_BinaryLocations::setImager(const QString& value)
{
    editImager_->setText(value);
}

void oskar_BinaryLocations::setInterferometer(const QString& value)
{
    editInterferometer_->setText(value);
}

// Private slots.

void oskar_BinaryLocations::setBeamPattern()
{
    QString name = QFileDialog::getOpenFileName(this, "Beam Pattern Binary");
    if (!name.isEmpty())
        setBeamPattern(name);
}

void oskar_BinaryLocations::setImager()
{
    QString name = QFileDialog::getOpenFileName(this, "Imager Binary");
    if (!name.isEmpty())
        setImager(name);
}

void oskar_BinaryLocations::setInterferometer()
{
    QString name = QFileDialog::getOpenFileName(this, "Interferometer Binary");
    if (!name.isEmpty())
        setInterferometer(name);
}