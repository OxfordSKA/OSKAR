/*
 * Copyright (c) 2017, The University of Oxford
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

#include "gui/oskar_AppLocation.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QGridLayout>
#include <QLineEdit>
#include <QPushButton>

namespace oskar {

AppLocation::AppLocation(QWidget *parent) : QDialog(parent)
{
    setWindowTitle("Location of OSKAR Applications");
    setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint |
            Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
    if (parent) resize(int(0.75 * parent->width()), 0);
    QGridLayout* gridLayout = new QGridLayout(this);
    editDir_ = new QLineEdit(this);
    QPushButton* browse = new QPushButton("Browse...", this);
    connect(browse, SIGNAL(clicked()), this, SLOT(browseClicked()));
    gridLayout->addWidget(editDir_, 0, 0, 1, 1);
    gridLayout->addWidget(browse, 0, 1, 1, 1);
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok |
            QDialogButtonBox::Cancel, Qt::Horizontal, this);
    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));
    gridLayout->addWidget(buttons, 1, 0, 1, 2);
}

QString AppLocation::dir() const
{
    return editDir_->text();
}

void AppLocation::setDir(const QString& value)
{
    editDir_->setText(value);
}

void AppLocation::browseClicked()
{
    QString name = QFileDialog::getExistingDirectory(this, "OSKAR Applications");
    if (!name.isEmpty())
        setDir(QDir::toNativeSeparators(name));
}

} /* namespace oskar */
