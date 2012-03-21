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

#include "oskar_global.h"
#include "widgets/oskar_About.h"

#include <QtGui/QApplication>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QLabel>
#include <QtGui/QTextEdit>
#include <QtGui/QTextDocument>
#include <QtGui/QSizePolicy>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

oskar_About::oskar_About(QWidget *parent) : QDialog(parent)
{
    // Set up the GUI.
    vLayoutMain_ = new QVBoxLayout(this);
    vLayout1_ = new QVBoxLayout;
    hLayout1_ = new QHBoxLayout;
    hLayout2_ = new QHBoxLayout;

    // Create icon.
    icon_ = new QLabel(this);
    QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
    sizePolicy.setHeightForWidth(icon_->sizePolicy().hasHeightForWidth());
    icon_->setSizePolicy(sizePolicy);
    icon_->setPixmap(QPixmap(QString::fromUtf8(":/icons/oskar-32x32.png")));
    icon_->setAlignment(Qt::AlignCenter);
    icon_->setMargin(10);
    hLayout1_->addWidget(icon_);

    // Create title.
    setWindowTitle("About OSKAR");
    title_ = new QLabel("OSKAR-2", this);
    title_->setFont(QFont("Arial", 28));
    hLayout1_->addWidget(title_);

    // Add title block to vertical layout.
    hLayout1_->setContentsMargins(0, 0, 80, 0);
    vLayout1_->addLayout(hLayout1_);

    // Create version label.
    version_ = new QLabel(QString("OSKAR Version %1")
            .arg(OSKAR_VERSION_STR), this);
    vLayout1_->addWidget(version_);

    // Create compilation date label.
    date_ = new QLabel(QString("Build Date: %1, %2").
            arg(__DATE__).arg(__TIME__), this);
    vLayout1_->addWidget(date_);

    // Add vertical spacer.
    verticalSpacer_ = new QSpacerItem(20, 40, QSizePolicy::Minimum,
            QSizePolicy::Expanding);
    vLayout1_->addItem(verticalSpacer_);

    // Create logos.
    oerc_ = new QLabel(this);
    oerc_->setSizePolicy(sizePolicy);
    oerc_->setPixmap(QPixmap(QString(":/icons/oerc-128x128.png")));
    oerc_->setAlignment(Qt::AlignCenter);
    oerc_->setMargin(4);
    oxford_ = new QLabel(this);
    oxford_->setSizePolicy(sizePolicy);
    oxford_->setPixmap(QPixmap(QString(":/icons/oxford-128x128.png")));
    oxford_->setAlignment(Qt::AlignCenter);
    oxford_->setMargin(4);
    hLayout2_->addLayout(vLayout1_);
    hLayout2_->addWidget(oerc_);
    hLayout2_->addWidget(oxford_);

    // Add top banner to main vertical layout.
    vLayoutMain_->addLayout(hLayout2_);

    // Create licence text.
    licenceText_ = new QTextDocument(this);
    QTextBlockFormat paragraph;
    paragraph.setBottomMargin(10);
    QTextCursor cursor(licenceText_);
    cursor.setBlockFormat(paragraph);
    cursor.insertText("Copyright (c) 2011-2012, The University of Oxford. \n"
            "All rights reserved.");
    cursor.insertList(QTextListFormat::ListDecimal);
    cursor.insertText("Redistributions of source code must retain the above "
            "copyright notice, this list of conditions and the following "
            "disclaimer.");
    cursor.insertBlock();
    cursor.insertText("Redistributions in binary form must reproduce the "
            "above copyright notice, this list of conditions and the "
            "following disclaimer in the documentation and/or other "
            "materials provided with the distribution.");
    cursor.insertBlock();
    cursor.insertText("Neither the name of the University of Oxford nor "
            "the names of its contributors may be used to endorse or promote "
            "products derived from this software without specific prior "
            "written permission.");
    cursor.insertBlock();
    cursor.setBlockFormat(paragraph);
    cursor.insertText("THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "
            "AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED "
            "WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED "
            "WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR "
            "PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT "
            "HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, "
            "INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES "
            "(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS "
            "OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS "
            "INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, "
            "WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING "
            "NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF "
            "THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH "
            "DAMAGE.");

    // Create license display.
    licence_ = new QTextEdit(this);
    licence_->setDocument(licenceText_);
    sizePolicy = licence_->sizePolicy();
    sizePolicy.setVerticalStretch(10);
    licence_->setSizePolicy(sizePolicy);
    licence_->setReadOnly(true);
    vLayoutMain_->addWidget(licence_);

    // Create attribution labels.
    attribution1_ = new QLabel("If you use OSKAR-2 in your research, please "
            "reference the following publication:", this);
    vLayoutMain_->addWidget(attribution1_);
    attribution2_ = new QLabel("(Insert name of publication here) "
            "MNRAS 2012, in prep.", this);
    vLayoutMain_->addWidget(attribution2_);

    // Create close button.
    buttons_ = new QDialogButtonBox(QDialogButtonBox::Ok,
            Qt::Horizontal, this);
    connect(buttons_, SIGNAL(accepted()), this, SLOT(accept()));
    vLayoutMain_->addWidget(buttons_);
}
