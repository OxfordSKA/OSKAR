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
#include "widgets/oskar_RunDialog.h"
#include "widgets/oskar_RunThread.h"

#include <QtCore/QProcess>
#include <QtCore/QSettings>
#include <QtGui/QApplication>
#include <QtGui/QCheckBox>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QScrollBar>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>

oskar_RunDialog::oskar_RunDialog(oskar_SettingsModel* model, QWidget *parent)
: QDialog(parent)
{
    aborted_ = false;

    // Set up the thread.
    thread_ = new oskar_RunThread(model, this);
    connect(thread_, SIGNAL(finished()), this, SLOT(runFinished()));
    connect(thread_, SIGNAL(outputData(QString)),
            this, SLOT(appendOutput(QString)));

    // Set up the GUI.
    setWindowTitle("OSKAR Run Progress");
    setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    setWindowModality(Qt::ApplicationModal);
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);
    QSettings settings;

    // Create text labels.
    labelText_ = new QLabel("OSKAR is running; please wait.", this);
    labelCommand_ = new QLabel(this);

    // Create terminal output display.
    display_ = new QTextEdit(this);
    display_->setReadOnly(true);
    QFont terminalFont("Dejavu Sans Mono");
    terminalFont.setStyleHint(QFont::TypeWriter);
    display_->setFont(terminalFont);
    display_->setMinimumSize(600, 300);

    // Create check box.
    autoClose_ = new QCheckBox(this);
    autoClose_->setText("Automatically close when run is over.");
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
        const QString& settings_file, QStringList outputs)
{
    // Store variables.
    binaryName_ = binary_name;
    settingsFile_ = settings_file;
    outputFiles_ = outputs;

    // Set text labels.
    QString commandString = binary_name + " ";
    if (settings_file.contains(' '))
        commandString += QString("\"%1\"").arg(settings_file);
    else
        commandString += settings_file;
    labelCommand_->setText(QString("Command: %1").arg(commandString));
    show();

    // Start the run in another thread.
    thread_->start(binary_name, settings_file, outputs);
}

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
        aborted_ = true;
        thread_->stop();
        thread_->wait();
    }
    else if (button == closeButton_)
    {
        // Close the dialog.
        accept();
    }
}

void oskar_RunDialog::runFinished()
{
    // Set the text label and button status.
    if (aborted_)
        labelText_->setText("Run cancelled.");
    else
        labelText_->setText("Run finished.");
    cancelButton_->setEnabled(false);
    closeButton_->setEnabled(true);
    if (autoClose_->isChecked())
        accept();
}
