/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "gui/oskar_About.h"

#include <QApplication>
#include <QDialogButtonBox>
#include <QFont>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSizePolicy>
#include <QTextBrowser>
#include <QTextDocument>
#include <QVBoxLayout>

namespace oskar {

About::About(QString app_name, QString app_version, QWidget *parent)
: QDialog(parent)
{
    // Set up the GUI.
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);
    QVBoxLayout* vLayout1 = new QVBoxLayout;
    QHBoxLayout* hLayout1 = new QHBoxLayout;

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
    QLabel* title = new QLabel("OSKAR", this);
    title->setFont(QFont("Arial", 28));
    hLayout1->addWidget(title);

    // Add title block to vertical layout.
    hLayout1->setContentsMargins(0, 0, 80, 0);
    vLayout1->addLayout(hLayout1);

    // Create application name and version label.
    if (app_name.isEmpty())
        app_name = QString("unknown");
    if (app_version.isEmpty())
        app_version = QString("unknown");
    QLabel* app = new QLabel(QString("Application: %1").arg(app_name), this);
    QLabel* ver = new QLabel(QString("Version: %1").arg(app_version), this);
    vLayout1->addWidget(app);
    vLayout1->addWidget(ver);

    // Create compilation date label.
    QLabel* date = new QLabel(QString("GUI Build Date: %1, %2").
            arg(__DATE__).arg(__TIME__), this);
    vLayout1->addWidget(date);

    // Add vertical spacer.
    vLayout1->addStretch();

    // Add top banner to main vertical layout.
    vLayoutMain->addLayout(vLayout1);

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
        cursor.insertText(
                "Copyright (c) 2011-2022, The OSKAR Developers.\n"
                "All rights reserved.");
        cursor.insertBlock();
        cursor.insertText("Redistribution and use in source and binary forms, "
                "with or without modification, are permitted provided that "
                "the following conditions are met:");
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
        cursor.insertText("Neither the name of the copyright holder nor "
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
    license->setMinimumWidth(500);
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
    html.append("<p>OSKAR has been developed using hardware "
            "donated by NVIDIA UK.</p>");
    html.append("<p>OSKAR uses the following tools and libraries:</p>");
    html.append("<ul>");
    html.append("<li>The NVIDIA CUDA toolkit: "
                "<a href=\"https://developer.nvidia.com/cuda-zone\">"
                "https://developer.nvidia.com/cuda-zone</a></li>");
    html.append("<li>The HDF 5 library: "
                "<a href=\"http://www.hdfgroup.org/\">"
                "http://www.hdfgroup.org/</a></li>");
    html.append("<li>The FFTPACK 5 FFT library: "
                "<a href=\"https://www2.cisl.ucar.edu/resources/legacy/fft5\">"
                "https://www2.cisl.ucar.edu/resources/legacy/fft5</a></li>");
    html.append("<li>The LAPACK linear algebra library: "
                "<a href=\"http://www.netlib.org/lapack/\">"
                "http://www.netlib.org/lapack/</a></li>");
    html.append("<li>The DIERCKX spline fitting library: "
                "<a href=\"http://netlib.org/dierckx/\">"
                "http://netlib.org/dierckx/</a></li>");
    html.append("<li>The Qt GUI framework: "
                "<a href=\"https://www.qt.io/\">https://www.qt.io/</a></li>");
    html.append("<li>The casacore Measurement Set library: "
                "<a href=\"https://github.com/casacore/casacore/\">"
                "https://github.com/casacore/casacore/</a></li>");
    html.append("<li>The CFITSIO FITS file library: "
                "<a href=\"https://heasarc.gsfc.nasa.gov/fitsio/\">"
                "https://heasarc.gsfc.nasa.gov/fitsio/</a></li>");
    html.append("<li>The Random123 random number generator: "
                "<a href=\"https://www.deshawresearch.com/resources_random123.html\">"
                "https://www.deshawresearch.com/resources_random123.html</a></li>");
    html.append("<li>The ezOptionParser command line parser: "
                "<a href=\"http://ezoptionparser.sourceforge.net/\">"
                "http://ezoptionparser.sourceforge.net/</a></li>");
    html.append("<li>The Tiny Template Library: "
                "<a href=\"http://tinytl.sourceforge.net/\">"
                "http://tinytl.sourceforge.net/</a></li>");
    html.append("<li>The RapidXML XML parser: "
                "<a href=\"http://rapidxml.sourceforge.net/\">"
                "http://rapidxml.sourceforge.net/</a></li>");
    html.append("<li>The HARP beam library</li>");
    html.append("<li>The CMake build system: "
                "<a href=\"https://cmake.org/\">https://cmake.org/</a></li>");
    html.append("<li>The Google Test framework: "
                "<a href=\"https://github.com/google/googletest/\">"
                "https://github.com/google/googletest/</a></li>");
    html.append("<li>Python: "
                "<a href=\"https://www.python.org/\">"
                "https://www.python.org/</a></li>");
    html.append("</ul>");
    html.append("<p>This research has made use of SAOImage DS9, developed "
            "by Smithsonian Astrophysical Observatory. "
            "SAOImage DS9 can be obtained from "
            "<a href=\"http://ds9.si.edu\">http://ds9.si.edu</a></p>");
    html.append("</body></html>");

    // Create attribution document display.
    QTextBrowser* libs = new QTextBrowser(this);
    libs->setOpenExternalLinks(true);
    libs->setHtml(html);
    libs->setReadOnly(true);
    vLayoutAtt->addWidget(libs);

    // Add attribution group.
    vLayoutMain->addWidget(grpAtt);

    // Create close button.
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok,
            Qt::Horizontal, this);
    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    vLayoutMain->addWidget(buttons);
}

} /* namespace oskar */
