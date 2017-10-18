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

#ifndef OSKAR_MAIN_WINDOW_H_
#define OSKAR_MAIN_WINDOW_H_

#include <QtWidgets/QMainWindow>
#include <QtCore/QString>
#include <QtCore/QHash>

class QComboBox;
class QLineEdit;
class QWidget;
class QNetworkAccessManager;
class QNetworkReply;

namespace oskar {

class SettingsModel;
class SettingsView;
class SettingsTree;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = 0);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event);

public slots:
    void clear();
    void open(QString filename = QString());
    void save(QString filename = QString());

private slots:
    void about();
    void appChanged(QString text = QString());
    void setAppDir(bool hint = false);
    void changeDir();
    void checkForUpdate();
    void cudaInfo();
    void helpDoc();
    void processNetworkReply(QNetworkReply*);
    void runButton();

private:
    void notFound(const QString& app_name);
    void run(QString app, const QStringList& args = QStringList(),
            bool allow_auto_close = true);
    void setApp(const QString& app, bool refresh = true);

private:
    bool checked_update_;
    QString title_;
    QLineEdit* current_dir_label_;
    QLineEdit* filter_;
    QComboBox* selector_;
    SettingsTree* settings_;
    SettingsView* view_;
    SettingsModel* model_;
    QHash<QString, QString> files_;
    QString current_dir_;
    QString current_app_;
    QString current_app_version_;
    QString app_dir_;

    QNetworkAccessManager* net_;
    QString version_url_;
};

} /* namespace oskar */

#endif /* OSKAR_MAIN_WINDOW_H_ */
