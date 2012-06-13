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
#include "widgets/oskar_DocumentationDisplay.h"

#include <QtGui/QApplication>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGroupBox>
#include <QtGui/QTextBrowser>
#include <QtGui/QVBoxLayout>

oskar_DocumentationDisplay::oskar_DocumentationDisplay(QWidget *parent)
: QDialog(parent)
{
    // Set up the GUI.
    setWindowTitle("Documentation");
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);

    // Create help group.
    QGroupBox* grp = new QGroupBox("OSKAR Documentation", this);
    grp->setMinimumSize(600, 300);
    QVBoxLayout* vLayoutAtt = new QVBoxLayout(grp);

    // Create help document.
    QString html;
    html.append("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" "
            "\"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
            "<html><head></head><body>\n");
    html.append("<p>");
    html.append("For the current release of OSKAR and all related "
            "documentation, please point your web browser at:");
    html.append("<ul><li><a href=\"http://www.oerc.ox.ac.uk/~ska/oskar2\">"
            "http://www.oerc.ox.ac.uk/~ska/oskar2</a></li></ul>");
    html.append("</p>");
    html.append("<p>");
    html.append("Please send all bug reports, feature requests, and general "
            "OSKAR-2 correspondence to the following email address:");
    html.append("<ul><li><a href=\"mailto:oskar@oerc.ox.ac.uk\">"
            "oskar@oerc.ox.ac.uk</a></li></ul>");
    html.append("</p>");
    html.append("<p>The following PDF documents are available:</p>");
    html.append("<ol>");
    html.append("<p><li>Release Notes<ul>");
    html.append("<li>Describes the features in the current release of "
            "OSKAR-2, and addresses common questions about the release.</li>");
    html.append("</ul></li></p>");
    html.append("<p><li>Installation Guide<ul>");
    html.append("<li>Describes how to build and install OSKAR-2.</li>");
    html.append("</ul></li></p>");
    html.append("<p><li>Example<ul>");
    html.append("<li>Describes how to run an example simulation and test that "
            "your version of OSKAR-2 is working as intended.</li>");
    html.append("</ul></li></p>");
    html.append("<p><li>Theory of Operation<ul>");
    html.append("<li>Describes the theory of operation of OSKAR-2, its "
            "implementation of the measurement equation and its treatment of "
            "polarisation. Please read this document to verify that OSKAR-2 "
            "works as you expect.</li>");
    html.append("</ul></li></p>");
    html.append("<p><li>Apps<ul>");
    html.append("<li>Describes the available OSKAR-2 applications and the "
            "MATLAB interface.</li>");
    html.append("</ul></li></p>");
    html.append("<p><li>Sky Model<ul>");
    html.append("<li>Describes the format of the OSKAR-2 sky model files.</li>");
    html.append("</ul></li></p>");
    html.append("<p><li>Telescope Model<ul>");
    html.append("<li>Describes the format of the OSKAR-2 telescope model "
            "files and directories.</li>");
    html.append("</ul></li></p>");
    html.append("<p><li>Settings Files<ul>");
    html.append("<li>Describes the format of the OSKAR-2 settings files.</li>");
    html.append("</ul></li></p>");
    html.append("</ol>");
    html.append("<p></p>");
    html.append("</body></html>");

    // Create help document display.
    QTextBrowser* display = new QTextBrowser(this);
    display->setHtml(html);
    display->setOpenExternalLinks(true);
    display->setReadOnly(true);
    vLayoutAtt->addWidget(display);

    // Add help group.
    vLayoutMain->addWidget(grp);

    // Create close button.
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok,
            Qt::Horizontal, this);
    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    vLayoutMain->addWidget(buttons);
}
