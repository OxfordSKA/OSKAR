/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <QtGui/QMainWindow>
#include <QtCore/QString>

class oskar_SettingsModel;
class oskar_SettingsModelFilter;
class oskar_SettingsView;
class QAction;
class QModelIndex;
class QWidget;
class QNetworkAccessManager;
class QNetworkReply;

class oskar_MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    oskar_MainWindow(QWidget* parent = 0);

protected:
    void closeEvent(QCloseEvent* event);

public slots:
    void openSettings(QString filename = QString());
    void saveSettingsAs(QString filename = QString());

private slots:
    void about();
    void binLocations();
    void checkForUpdate();
    void cudaInfo();
    void helpDoc();
    void openRecentFile();
    void processNetworkReply(QNetworkReply*);
    void runBeamPattern();
    void runInterferometer();
    void runImager();
    void runFitElementData();
    void setHideUnsetItems(bool value);

private:
    void runButton();

    void createRecentFileActions();
    void updateRecentFileActions();
    void updateRecentFileList();

private:
    QString mainTitle_;
    QWidget* widget_;
    oskar_SettingsModel* model_;
    oskar_SettingsModelFilter* modelProxy_;
    oskar_SettingsView* view_;
    QAction* actHideUnset_;
    QString settingsFile_;
    QString run_binary_;

    QMenuBar* menubar_;
    QMenu* menuFile_;

    enum { MaxRecentFiles = 3 };
    QMenu* recentFileMenu_;
    QAction* separator_;
    QAction* recentFiles_[MaxRecentFiles];
    QNetworkAccessManager* networkManager_;
    QString version_url_;

    // Binary path names.
    QString binary_interferometer_;
    QString binary_beam_pattern_;
    QString binary_imager_;
    QString binary_fit_element_data_;
    QString binary_cuda_info_;

    bool isModified_;
};

#endif // OSKAR_MAIN_WINDOW_H_
