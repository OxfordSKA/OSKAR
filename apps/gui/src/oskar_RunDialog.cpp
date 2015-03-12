/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <apps/gui/oskar_RunDialog.h>
#include <apps/gui/oskar_RunThread.h>

#include <oskar_version.h>
#include <oskar_get_error_string.h>

#include <QtCore/QProcess>
#include <QtCore/QSettings>
#include <QtGui/QApplication>
#include <QtGui/QCheckBox>
#include <QtGui/QCloseEvent>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QLabel>
#include <QtGui/QMessageBox>
#include <QtGui/QPushButton>
#include <QtGui/QScrollBar>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>

oskar_RunDialog::oskar_RunDialog(QWidget *parent)
: QDialog(parent), labelSettingsFile_(0)
{
    // Initialise members.
    failed_ = false;
    finished_ = false;

    // Set up the thread.
    thread_ = new oskar_RunThread(this);
    connect(thread_, SIGNAL(aborted()), this, SLOT(runAborted()));
    connect(thread_, SIGNAL(completed()), this, SLOT(runCompleted()));
    connect(thread_, SIGNAL(crashed()), this, SLOT(runCrashed()));
    connect(thread_, SIGNAL(failed()), this, SLOT(runFailed()));
    connect(thread_, SIGNAL(finished()), this, SLOT(runFinished()));
    connect(thread_, SIGNAL(outputData(QString)),
            this, SLOT(appendOutput(QString)));

    // Set up the GUI.
    setWindowTitle("OSKAR Run Progress");
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
#ifdef Q_OS_WIN32
    terminalFont.setFamily("Lucida Console");
#else
    terminalFont.setFamily("DejaVu Sans Mono");
#endif
    terminalFont.setPointSize(10);
    terminalFont.setStyleHint(QFont::TypeWriter);
    display_->setFont(terminalFont);
    display_->setMinimumSize(600, 300);

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
}

oskar_RunDialog::~oskar_RunDialog()
{
    QSettings settings;
    settings.setValue("run_dialog/close_when_finished", autoClose_->isChecked());

    buttons_->clear();
}

void oskar_RunDialog::start(const QString& binary_name,
        const QString& settings_file)
{
    // Store variables.
    binaryName_ = binary_name;
    settingsFile_ = settings_file;

    // Set text labels.
    QString commandString = binary_name + " ";
    if (settings_file.contains(' '))
        commandString += QString("\"%1\"").arg(settings_file);
    else
        commandString += settings_file;
    labelCommand_->setText(QString("Command: %1").arg(commandString));
    show();

    // Start the run in another thread.
    thread_->start(binary_name, settings_file);
}

// Protected methods.

void oskar_RunDialog::closeEvent(QCloseEvent* event)
{
    if (finished_)
        event->accept();
    else
        event->ignore();
}

// Private slots.

void oskar_RunDialog::appendOutput(QString output)
{
    if (output.size() > 0)
    {
        // Determine whether scrolling needs to happen.
        bool scroll = false;
        QScrollBar* scrollBar = display_->verticalScrollBar();
        if (scrollBar->value() == scrollBar->maximum())
            scroll = true;

        // Append the output text.
        QTextCursor cursor = display_->textCursor();
        cursor.movePosition(QTextCursor::End);
        display_->setTextCursor(cursor);
        display_->insertPlainText(output);

        // Scroll to bottom if necessary.
        if (scroll)
            scrollBar->setValue(scrollBar->maximum());
    }
}

void oskar_RunDialog::buttonClicked(QAbstractButton* button)
{
    if (button == cancelButton_)
    {
        // Stop the running thread (kills the process).
        thread_->stop();
        thread_->wait();
    }
    else if (button == closeButton_)
    {
        // Close the dialog.
        accept();
    }
}

void oskar_RunDialog::runAborted()
{
    labelText_->setText("Run aborted.");
}

void oskar_RunDialog::runCompleted()
{
    labelText_->setText("Run completed.");
}

void oskar_RunDialog::runCrashed()
{
    failed_ = true;
    labelText_->setText("Run crashed.");
    QMessageBox msgBox(this);
    msgBox.setWindowTitle(QString("OSKAR (%1)").arg(OSKAR_VERSION_STR));
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.setText("Oops! Sorry, OSKAR seems to have crashed.");
    msgBox.setInformativeText("Please report this problem by sending an email "
            "to the oskar@oerc.ox.ac.uk support address, and provide details "
            "of your version of OSKAR, your hardware configuration, your run "
            "log, and what you were trying to do at the time.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}

void oskar_RunDialog::runFailed()
{
    failed_ = true;
    labelText_->setText("Run failed.");
    QMessageBox msgBox(this);
    msgBox.setWindowTitle(QString("OSKAR (%1)").arg(OSKAR_VERSION_STR));
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.setText("Run failed: please see output log for details.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}

void oskar_RunDialog::runFinished()
{
    finished_ = true;

    // Set the button status.
    cancelButton_->setEnabled(false);
    closeButton_->setEnabled(true);
    closeButton_->setFocus();
    if (autoClose_->isChecked() && !failed_)
        accept();
}
