/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "gui/oskar_Docs.h"

#include <QDialogButtonBox>
#include <QTextBrowser>
#include <QVBoxLayout>

namespace oskar {

Docs::Docs(QWidget *parent) : QDialog(parent)
{
    // Set up the GUI.
    setWindowTitle("Documentation");
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);

    // Create help document.
    QString gitlab_repo = "https://gitlab.com/ska-telescope/sim/oskar";
    QString doc_url = "https://ska-telescope.gitlab.io/sim/oskar";
    QString html;
    html.append("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" "
            "\"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
            "<html><head></head><body>\n");
    html.append("<p>");
    html.append("The OSKAR repository is now on GitLab at:");
    html.append(QString("<ul><li><a href=\"%1\">%2</a></li></ul>").
            arg(gitlab_repo).arg(gitlab_repo));
    html.append("</p>");
    html.append("<p>");
    html.append("All documentation is available online at:");
    html.append(QString("<ul><li><a href=\"%1\">%2</a></li></ul>").
            arg(doc_url).arg(doc_url));
    html.append("</p>");
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

} /* namespace oskar */
