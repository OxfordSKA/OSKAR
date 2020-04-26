/*
 * Copyright (c) 2012-2020, The University of Oxford
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

#ifndef OSKAR_RUN_DIALOG_H_
#define OSKAR_RUN_DIALOG_H_

#include <QDialog>
#include <QProcess>

class QAbstractButton;
class QCheckBox;
class QCloseEvent;
class QDialogButtonBox;
class QLabel;
class QPushButton;
class QTextEdit;

namespace oskar {

class RunDialog : public QDialog
{
    Q_OBJECT

public:
    RunDialog(const QString& app, QWidget *parent = 0);
    ~RunDialog();

    void setAllowAutoClose(bool value);
    void start(const QStringList& args = QStringList());

protected:
    void closeEvent(QCloseEvent*);

private slots:
    void buttonClicked(QAbstractButton* button);
    void readProcess();
    void runAborted();
    void runCompleted();
    void runCrashed();
    void runFailed();
    void runFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    bool aborted_;
    bool failed_;
    bool finished_;
    QCheckBox* autoClose_;
    QTextEdit* display_;
    QLabel* labelText_;
    QLabel* labelCommand_;
    QDialogButtonBox* buttons_;
    QPushButton* closeButton_;
    QPushButton* cancelButton_;
    QString app_;
    QProcess* process_;
};

} /* namespace oskar */

#endif /* OSKAR_RUN_DIALOG_H_ */
