/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_global.h>
#include <oskar_version.h>
#include <apps/gui/oskar_About.h>

#include <QtGui/QApplication>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QFont>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QLabel>
#include <QtGui/QTextBrowser>
#include <QtGui/QTextDocument>
#include <QtGui/QSizePolicy>
#include <QtGui/QVBoxLayout>

oskar_About::oskar_About(QWidget *parent) : QDialog(parent)
{
    // Set up the GUI.
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);
    QVBoxLayout* vLayout1 = new QVBoxLayout;
    QHBoxLayout* hLayout1 = new QHBoxLayout;
    QHBoxLayout* hLayout2 = new QHBoxLayout;

    // Create icon.
    QLabel* icon = new QLabel(this);
    QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
    sizePolicy.setHeightForWidth(icon->sizePolicy().hasHeightForWidth());
    icon->setSizePolicy(sizePolicy);
    icon->setPixmap(QPixmap(QString::fromUtf8(":/icons/oskar-32x32.png")));
    icon->setAlignment(Qt::AlignCenter);
    icon->setMargin(10);
    hLayout1->addWidget(icon);

    // Create title.
    setWindowTitle("About OSKAR");
    QLabel* title = new QLabel("OSKAR 2", this);
    title->setFont(QFont("Arial", 28));
    hLayout1->addWidget(title);

    // Add title block to vertical layout.
    hLayout1->setContentsMargins(0, 0, 80, 0);
    vLayout1->addLayout(hLayout1);

    // Create version label.
    QLabel* version = new QLabel(QString("OSKAR Version %1")
            .arg(OSKAR_VERSION_STR), this);
    vLayout1->addWidget(version);

    // Create compilation date label.
    QLabel* date = new QLabel(QString("Build Date: %1, %2").
            arg(__DATE__).arg(__TIME__), this);
    vLayout1->addWidget(date);

    // Add vertical spacer.
    vLayout1->addStretch();

    // Create logos.
    QLabel* oerc = new QLabel(this);
    oerc->setSizePolicy(sizePolicy);
    oerc->setPixmap(QPixmap(QString(":/icons/oerc-128x128.png")));
    oerc->setAlignment(Qt::AlignCenter);
    oerc->setMargin(4);
    QLabel* oxford = new QLabel(this);
    oxford->setSizePolicy(sizePolicy);
    oxford->setPixmap(QPixmap(QString(":/icons/oxford-128x128.png")));
    oxford->setAlignment(Qt::AlignCenter);
    oxford->setMargin(4);
    hLayout2->addLayout(vLayout1);
    hLayout2->addWidget(oerc);
    hLayout2->addWidget(oxford);

    // Add top banner to main vertical layout.
    vLayoutMain->addLayout(hLayout2);

    // Create license group.
    QGroupBox* grpLic = new QGroupBox("License", this);
    sizePolicy = grpLic->sizePolicy();
    sizePolicy.setVerticalStretch(10);
    grpLic->setSizePolicy(sizePolicy);
    QVBoxLayout* vLayoutLic = new QVBoxLayout(grpLic);

    // Create license text.
    QTextDocument* licenseText = new QTextDocument(this);
    {
        QTextBlockFormat paragraph;
        paragraph.setBottomMargin(10);
        QTextCursor cursor(licenseText);
        cursor.setBlockFormat(paragraph);
        cursor.insertText("Copyright (c) 2011-2015, The University of Oxford. "
                "\nAll rights reserved.");
        cursor.insertList(QTextListFormat::ListDecimal);
        cursor.insertText("Redistributions of source code must retain the "
                "above copyright notice, this list of conditions and the "
                "following disclaimer.");
        cursor.insertBlock();
        cursor.insertText("Redistributions in binary form must reproduce the "
                "above copyright notice, this list of conditions and the "
                "following disclaimer in the documentation and/or other "
                "materials provided with the distribution.");
        cursor.insertBlock();
        cursor.insertText("Neither the name of the University of Oxford nor "
                "the names of its contributors may be used to endorse or "
                "promote products derived from this software without specific "
                "prior written permission.");
        cursor.insertBlock();
        cursor.setBlockFormat(paragraph);
        cursor.insertText("THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "
                "AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED "
                "WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED "
                "WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR "
                "PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT "
                "HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, "
                "INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES "
                "(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE "
                "GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS "
                "INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, "
                "WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING "
                "NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF "
                "THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH "
                "DAMAGE.");
    }

    // Create license display.
    QTextBrowser* license = new QTextBrowser(this);
    license->setDocument(licenseText);
    license->setReadOnly(true);
    vLayoutLic->addWidget(license);

    // Add license group.
    vLayoutMain->addWidget(grpLic);

    // Create attribution group.
    QGroupBox* grpAtt = new QGroupBox("Attribution && Acknowledgements", this);
    sizePolicy = grpAtt->sizePolicy();
    sizePolicy.setVerticalStretch(10);
    grpAtt->setSizePolicy(sizePolicy);
    QVBoxLayout* vLayoutAtt = new QVBoxLayout(grpAtt);

    // Create attribution document.
    QString html;
    html.append("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" "
            "\"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
            "<html><head></head><body>\n");
    html.append("<p>OSKAR 2 has been developed using hardware "
            "donated by NVIDIA UK.</p>");
    html.append("<p>OSKAR 2 directly links against or uses the following "
            "software libraries:</p>");
    html.append("<ul>");
    html.append("<li>NVIDIA CUDA "
                "(<a href=\"http://www.nvidia.com/object/cuda_home.html\">"
                "http://www.nvidia.com/object/cuda_home.html</a>)</li>");
#ifndef OSKAR_NO_LAPACK
    html.append("<li>LAPACK "
                "(<a href=\"http://www.netlib.org/lapack/\">"
                "http://www.netlib.org/lapack/</a>)</li>");
#endif
    html.append("<li>DIERCKX for surface fitting using splines "
                "(<a href=\"http://netlib.org/dierckx/\">"
                "http://netlib.org/dierckx/</a>)</li>");
    html.append("<li>The Qt cross-platform application framework "
                "(<a href=\"http://qt.io/\">"
                "http://qt.io/</a>)</li>");
#ifndef OSKAR_NO_MS
    html.append("<li>casacore for Measurement Set export "
                "(<a href=\"http://code.google.com/p/casacore/\">"
                "http://code.google.com/p/casacore/</a>)</li>");
#endif
    html.append("<li>CFITSIO for FITS file export "
                "(<a href=\"http://heasarc.gsfc.nasa.gov/fitsio/\">"
                "http://heasarc.gsfc.nasa.gov/fitsio/</a>)</li>");
    html.append("<li>Random123 for parallel random number generation "
                "(<a href=\"http://www.deshawresearch.com/resources_random123.html\">"
                "http://www.deshawresearch.com/resources_random123.html</a>)</li>");
    html.append("<li>ezOptionParser "
                "(<a href=\"http://sourceforge.net/projects/ezoptionparser/\">"
                "http://sourceforge.net/projects/ezoptionparser/</a>)</li>");
    html.append("</ul>");
    html.append("<p>The following tools have been used during the development "
            "of OSKAR 2:</p>");
    html.append("<ul>");
    html.append("<li>The CMake cross-platform build system "
                "(<a href=\"http://www.cmake.org/\">"
                "http://www.cmake.org/</a>)</li>");
    html.append("<li>The Google Test unit-testing framework "
                "(<a href=\"http://code.google.com/p/googletest/\">"
                "http://code.google.com/p/googletest/</a>)</li>");
    html.append("<li>The Eclipse source-code IDE "
                "(<a href=\"http://www.eclipse.org/\">"
                "http://www.eclipse.org/</a>)</li>");
    html.append("<li>The Valgrind memory checker "
                "(<a href=\"http://valgrind.org/\">"
                "http://valgrind.org/</a>)</li>");
    html.append("<li>The GCC toolchain "
                "(<a href=\"http://gcc.gnu.org/\">"
                "http://gcc.gnu.org/</a>)</li>");
    html.append("<li>MATLAB "
                "(<a href=\"http://www.mathworks.co.uk/products/matlab/\">"
                "http://www.mathworks.co.uk/products/matlab/</a>)</li>");
    html.append("</ul>");
    html.append("<p>This research has made use of SAOImage DS9, developed "
            "by Smithsonian Astrophysical Observatory.</p>");
    html.append("</body></html>");

    // Create attribution document display.
    QTextBrowser* libs = new QTextBrowser(this);
    libs->setOpenExternalLinks(true);
    libs->setHtml(html);
    libs->setReadOnly(true);
    vLayoutAtt->addWidget(libs);

    // Create acknowledgement labels.
    QLabel* ack1 = new QLabel("If OSKAR has been helpful in your research, "
            "please give the following acknowledgement:", this);
    vLayoutAtt->addWidget(ack1);
    QLabel* ack2 = new QLabel("<blockquote><i>\"This research has made use of OSKAR, "
            "developed at the University of Oxford.\"</i></blockquote>", this);
    ack2->setTextFormat(Qt::RichText);
    vLayoutAtt->addWidget(ack2);
//    QLabel* ack3 = new QLabel("and/or reference the following publication:",
//            this);
//    vLayoutAtt->addWidget(ack3);
//    QLabel* ack4 = new QLabel("<blockquote>Dulwich, F., Mort, B. J., Salvini, S., "
//            "\"<i>OSKAR: A software package to simulate data from radio "
//            "interferometers\"</i>,<br>"
//            "MNRAS 2015, in preparation.</blockquote>", this);
//    ack4->setTextFormat(Qt::RichText);
//    vLayoutAtt->addWidget(ack4);

    // Add attribution group.
    vLayoutMain->addWidget(grpAtt);

    // Create close button.
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok,
            Qt::Horizontal, this);
    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    vLayoutMain->addWidget(buttons);
}
