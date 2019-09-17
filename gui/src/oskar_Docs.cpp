/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#include "gui/oskar_Docs.h"

#include <QtWidgets/QApplication>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QVBoxLayout>

static const char* root_url = "https://github.com/OxfordSKA/OSKAR/releases";

namespace oskar {

Docs::Docs(QWidget *parent) : QDialog(parent)
{
    // Set up the GUI.
    setWindowTitle("Documentation");
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);

    // Create help document.
    QString html;
    html.append("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" "
            "\"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
            "<html><head></head><body>\n");
    html.append("<p>");
    html.append("For the current release of OSKAR and all related "
            "documentation, please visit:");
    html.append(QString("<ul><li><a href=\"%1\">%2</a></li></ul>").
            arg(root_url).arg(root_url));
    html.append("</p>");
    html.append("<p>The following PDF documents are available:</p>");
    html.append("<ol>");

    add_doc(html,
            "Introduction & FAQ",
            "An introduction to the OSKAR package");
    add_doc(html,
            "Release Notes",
            "Describes the changes in the current release of OSKAR");
    add_doc(html,
            "Installation Guide",
            "Describes how to build and install OSKAR");
    add_doc(html,
            "Example",
            "Describes how to run an example simulation and test that "
            "your version of OSKAR is working as intended");
    add_doc(html,
            "Theory of Operation",
            "Describes the theory of operation of OSKAR, its "
            "implementation of the measurement equation and its treatment of "
            "polarisation. Please read this document to verify that OSKAR "
            "works as you expect");
    add_doc(html,
            "Applications",
            "Describes the available OSKAR applications");
    add_doc(html,
            "Sky Model",
            "Describes the format of the OSKAR sky model files");
    add_doc(html,
            "Telescope Model",
            "Describes the format of the OSKAR telescope model files and "
            "directories");
    add_doc(html,
            "Pointing File",
            "Describes the format of OSKAR pointing files");
    add_doc(html,
            "Settings",
            "Describes the format of the OSKAR settings files");
    add_doc(html,
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
    display->setMinimumSize(600, 300);

    // Add help group.
    vLayoutMain->addWidget(display);

    // Create close button.
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok,
            Qt::Horizontal, this);
    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    vLayoutMain->addWidget(buttons);
}

void Docs::add_doc(QString& html, const char* title, const char* desc)
{
    html.append("<p>");
    html.append("<li>&nbsp;");
    html.append(QString("%1").arg(title));
    html.append("<ul>");
    html.append(QString("<li>%1.</li>").arg(desc));
    html.append("</ul>");
    html.append("</li>");
    html.append("</p>");
}

} /* namespace oskar */
