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

static void add_doc(QString& html, const char* link, const char* title,
        const char* desc);


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

    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Introduction.pdf",
            "Introduction",
            "An introduction to the OSKAR-2 documentation");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Release-Notes-and-FAQ.pdf",
            "Release Notes & FAQ",
            "Describes the features of the current release of OSKAR-2 and "
            "addresses common questions about the release");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Install.pdf",
            "Installation Guide",
            "Describes how to build and install OSKAR-2");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Example.pdf",
            "Example",
            "Describes how to run an example simulation and test that "
            "your version of OSKAR-2 is working as intended");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Theory.pdf",
            "Theory of Operation",
            "Describes the theory of operation of OSKAR-2, its "
            "implementation of the measurement equation and its treatment of "
            "polarisation. Please read this document to verify that OSKAR-2 "
            "works as you expect");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Apps.pdf",
            "Applications",
            "Describes the available OSKAR-2 applications");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Sky-Model.pdf",
            "Sky Model",
            "Describes the format of the OSKAR-2 sky model files");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Telescope-Model.pdf",
            "Telescope Model",
            "Describes the format of the OSKAR-2 telescope model files and "
            "directories");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Pointing-File.pdf",
            "Pointing File",
            "Describes the format of OSKAR-2 pointing files");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Settings-Files.pdf",
            "Settings Files",
            "Describes the format of the OSKAR-2 settings files");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-MATLAB-Interface.pdf",
            "MATLAB Interface",
            "Describes an experimental interface for accessing OSKAR data types "
            "and making images of visibility data in MATLAB");
    add_doc(html,
            "http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Binary-File-Format.pdf",
            "Binary File Format",
            "Describes the format of binary files written by OSKAR applications");


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

static void add_doc(QString& html, const char* link, const char* title,
        const char* desc)
{
    html.append("<p>");
    html.append("<li>&nbsp;");
    html.append(QString("<a href=\"%1\">%2</a>").arg(link).arg(title));
    html.append("<ul>");
    html.append(QString("<li>%1.</li>").arg(desc));
    html.append("</ul>");
    html.append("</li>");
    html.append("</p>");
}

