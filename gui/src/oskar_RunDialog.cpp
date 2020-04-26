/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "gui/oskar_RunDialog.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollBar>
#include <QSettings>
#include <QTextEdit>
#include <QVBoxLayout>

namespace oskar {

RunDialog::RunDialog(const QString& app, QWidget *parent) : QDialog(parent)
{
    // Initialise members.
    aborted_ = false;
    failed_ = false;
    finished_ = false;
    app_ = app;

    // Set up the GUI.
    QFileInfo fi(app);
    QString app_name = fi.fileName();
    setWindowTitle(QString("Output of %1").arg(app_name));
    setWindowModality(Qt::ApplicationModal);
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);
    QSettings settings;

    // Create text labels.
    labelText_ = new QLabel("OSKAR is running; please wait.", this);
    labelCommand_ = new QLabel(this);

    // Create terminal output display.
    display_ = new QTextEdit(this);
    display_->setReadOnly(true);
    QFont terminalFont;
#if defined(Q_OS_WIN32)
    terminalFont.setFamily("Lucida Console");
    terminalFont.setPointSize(10);
#elif defined(Q_OS_MAC)
    terminalFont.setFamily("Menlo");
    terminalFont.setPointSize(11);
#else
    terminalFont.setFamily("DejaVu Sans Mono");
    terminalFont.setPointSize(10);
#endif
    terminalFont.setStyleHint(QFont::TypeWriter);
    display_->setFont(terminalFont);

    // Set the display size based on the font.
    QFontMetrics metric(terminalFont);
    int displayWidth = 75 * metric.width('A');
    int displayHeight = 25 * metric.lineSpacing() + metric.ascent() - 2;
    display_->setMinimumSize(displayWidth, displayHeight);

    // Create check box.
    autoClose_ = new QCheckBox(this);
    autoClose_->setText("Automatically close when the run has completed "
            "successfully.");
    autoClose_->setChecked(settings.value("run_dialog/close_when_finished",
            false).toBool());

    // Create buttons.
    buttons_ = new QDialogButtonBox(Qt::Horizontal, this);
    closeButton_ = buttons_->addButton(QDialogButtonBox::Close);
    cancelButton_ = buttons_->addButton(QDialogButtonBox::Cancel);
    closeButton_->setDisabled(true);
    connect(buttons_, SIGNAL(clicked(QAbstractButton*)),
            this, SLOT(buttonClicked(QAbstractButton*)));

    // Add widgets to layout.
    vLayoutMain->addWidget(labelText_);
    vLayoutMain->addWidget(labelCommand_);
    vLayoutMain->addWidget(display_);
    vLayoutMain->addWidget(autoClose_);
    vLayoutMain->addWidget(buttons_);

    // Create process handle.
    process_ = new QProcess(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, SIGNAL(readyRead()), this, SLOT(readProcess()));
    connect(process_, SIGNAL(finished(int, QProcess::ExitStatus)),
            this, SLOT(runFinished(int, QProcess::ExitStatus)));
}

RunDialog::~RunDialog()
{
    QSettings settings;
    settings.setValue("run_dialog/close_when_finished", autoClose_->isChecked());
    buttons_->clear();
}

void RunDialog::setAllowAutoClose(bool value)
{
    autoClose_->setEnabled(value);
}

void RunDialog::start(const QStringList& args)
{
    // Set text labels.
    QString commandString = app_ + " ";
    Q_FOREACH(QString a, args)
    {
        if (a.contains(' '))
            commandString += QString("\"%1\"").arg(a);
        else
            commandString += a;
    }
    labelCommand_->setText(QString("Command: %1").arg(commandString));
    show();

    // Start the run.
    process_->start(app_, args);
}

// Protected methods.

void RunDialog::closeEvent(QCloseEvent* event)
{
    finished_ ? event->accept() : event->ignore();
}

// Private slots.

void RunDialog::readProcess()
{
    QString output = QString(process_->readAll());
    if (output.size() > 0)
    {
        // Determine whether scrolling needs to happen.
        QScrollBar* scrollBar = display_->verticalScrollBar();
        const int old_value = scrollBar->value();
        const bool scroll = (old_value >= scrollBar->maximum() - 4);

        // Append the output text and set scroll bar position.
        display_->moveCursor(QTextCursor::End);
        display_->insertPlainText(output);
        if (scroll)
            display_->ensureCursorVisible();
        else
            scrollBar->setValue(old_value);
    }
}

void RunDialog::buttonClicked(QAbstractButton* button)
{
    if (button == closeButton_)
        accept();
    else if (button == cancelButton_)
    {
        aborted_ = true;
        process_->kill();
    }
}

void RunDialog::runAborted()
{
    labelText_->setText("Run aborted.");
}

void RunDialog::runCompleted()
{
    labelText_->setText("Run completed.");
}

void RunDialog::runCrashed()
{
    failed_ = true;
    labelText_->setText("Run crashed.");
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("OSKAR");
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.setText("Oops! Sorry, OSKAR seems to have crashed.");
    msgBox.setInformativeText("Please report this problem to the OSKAR "
            "developers, and describe the steps required to reproduce "
            "the issue so we can investigate it.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}

void RunDialog::runFailed()
{
    failed_ = true;
    labelText_->setText("Run failed.");
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("OSKAR");
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.setText("Run failed: please see output log for details.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}

void RunDialog::runFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    finished_ = true;
    readProcess();

    // Check exit conditions.
    if (aborted_)
        runAborted();
    else if (exitStatus == QProcess::CrashExit)
        runCrashed();
    else if (exitCode != 0)
        runFailed();
    else
        runCompleted();

    // Set the button status.
    cancelButton_->setEnabled(false);
    closeButton_->setEnabled(true);
    closeButton_->setFocus();
    if (autoClose_->isChecked() && autoClose_->isEnabled()
            && !(failed_ || aborted_))
        accept();
}

} /* namespace oskar */
