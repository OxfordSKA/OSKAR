/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "apps/gui/oskar_MainWindow.h"

#include "oskar_version.h"
#include "apps/gui/oskar_About.h"
#include "apps/gui/oskar_BinaryLocations.h"
#include "apps/gui/oskar_CudaInfoDisplay.h"
#include "apps/gui/oskar_DocumentationDisplay.h"
#include "apps/gui/oskar_RunDialog.h"
#include "oskar_SettingsDelegate_new.hpp"
#include "oskar_SettingsDeclareXml.hpp"
#include "oskar_SettingsModel_new.hpp"
#include "oskar_SettingsFileHandlerIni.hpp"
#include "oskar_SettingsView.h"
#include "oskar_SettingsTree.hpp"

// FIXME(FD) Replace these by querying the binary itself for its own XML.
#include "apps/xml/oskar_sim_beam_pattern_xml_all.h"
#include "apps/xml/oskar_sim_interferometer_xml_all.h"
#include "apps/xml/oskar_imager_xml_all.h"
#include "apps/xml/oskar_fit_element_data_xml_all.h"

#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QCloseEvent>
#include <QtGui/QComboBox>
#include <QtGui/QFileDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QMessageBox>
#include <QtGui/QPushButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QKeySequence>
#include <QtCore/QModelIndex>
#include <QtCore/QProcess>
#include <QtCore/QTimer>
#include <QtCore/QSettings>
#include <QtNetwork/QNetworkAccessManager>
#include <QtNetwork/QNetworkReply>

namespace oskar {

MainWindow::MainWindow(QWidget* parent)
: QMainWindow(parent)
{
    // Set the window title.
    mainTitle_ = QString("OSKAR (%1)").arg(OSKAR_VERSION_STR);

    // Create the central widget and main layout.
    setWindowTitle(mainTitle_);
    setWindowIcon(QIcon(":/icons/oskar.ico"));
    widget_ = new QWidget(this);
    setCentralWidget(widget_);
    QVBoxLayout* v_layout = new QVBoxLayout(widget_);
    QGridLayout* gridLayout = new QGridLayout;

    // Create the settings model.
    settings_ = new oskar::SettingsTree;
    handler_ = new oskar::SettingsFileHandlerIni(OSKAR_VERSION_STR);
    settings_->set_file_handler(handler_);
    model_ = new oskar::SettingsModel(settings_, this);
    QSortFilterProxyModel* modelProxy_ = new oskar::SettingsModelFilter(this);
    modelProxy_->setSourceModel(model_);

    // Create the binary selector and search box.
    QLabel* label1 = new QLabel("Application", widget_);
    selector_ = new QComboBox(widget_);
    selector_->addItem("oskar_sim_interferometer");
    selector_->addItem("oskar_sim_beam_pattern");
    selector_->addItem("oskar_imager");
    selector_->addItem("oskar_fit_element_data");
    selector_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    QPushButton* runner = new QPushButton(widget_);
    runner->setText("Run");
    QLabel* label2 = new QLabel("Settings View Filter", widget_);
    filterBox_ = new QLineEdit(widget_);
    gridLayout->addWidget(label1, 0, 0);
    gridLayout->addWidget(selector_, 0, 1);
    gridLayout->addWidget(runner, 0, 2);
    gridLayout->addWidget(label2, 1, 0);
    gridLayout->addWidget(filterBox_, 1, 1);
    connect(runner, SIGNAL(clicked()), SLOT(runButton()));
    connect(selector_, SIGNAL(currentIndexChanged(const QString&)),
            SLOT(binaryChanged(const QString&)));
    connect(filterBox_, SIGNAL(textChanged(QString)),
            modelProxy_, SLOT(setFilterRegExp(QString)));

    // Create and set up the settings view.
    view_ = new oskar::SettingsView(widget_);
    oskar::SettingsDelegate* delegate =
            new oskar::SettingsDelegate(view_, widget_);
    view_->setModel(modelProxy_);
    view_->setItemDelegate(delegate);
    view_->resizeColumnToContents(0);
    v_layout->addLayout(gridLayout);
    v_layout->addWidget(view_);
    connect(model_, SIGNAL(fileReloaded()), view_, SLOT(fileReloaded()));

    // Create and set up the actions.
    QAction* actOpen = new QAction("&Open...", this);
    QAction* actSaveAs = new QAction("&Save As...", this);
    QAction* actExit = new QAction("E&xit", this);
    QAction* actBinLocations = new QAction("&Binary Locations...", this);
    QAction* actShowFirstLevel = new QAction("Show &First Level", this);
    QAction* actExpandAll = new QAction("&Expand All", this);
    QAction* actCollapseAll = new QAction("&Collapse All", this);
    QAction* actDisplayKeys = new QAction("Display &Keys", this);
    QAction* actDisplayLabels = new QAction("Display &Labels", this);
    QAction* actHelpDoc = new QAction("Documentation...", this);
    QAction* actCudaInfo = new QAction("CUDA System Info...", this);
    QAction* actAbout = new QAction("&About OSKAR...", this);
    connect(actOpen, SIGNAL(triggered()), this, SLOT(openSettings()));
    connect(actSaveAs, SIGNAL(triggered()), this, SLOT(saveSettingsAs()));
    connect(actExit, SIGNAL(triggered()), this, SLOT(close()));
    connect(actBinLocations, SIGNAL(triggered()), this, SLOT(binLocations()));
    connect(actShowFirstLevel, SIGNAL(triggered()),
            view_, SLOT(showFirstLevel()));
    connect(actDisplayKeys, SIGNAL(triggered()), view_, SLOT(displayKeys()));
    connect(actDisplayLabels, SIGNAL(triggered()),
            view_, SLOT(displayLabels()));
    connect(actExpandAll, SIGNAL(triggered()), view_, SLOT(expandSettingsTree()));
    connect(actCollapseAll, SIGNAL(triggered()), view_, SLOT(collapseAll()));
    connect(actAbout, SIGNAL(triggered()), this, SLOT(about()));
    connect(actCudaInfo, SIGNAL(triggered()), this, SLOT(cudaInfo()));
    connect(actHelpDoc, SIGNAL(triggered()), this, SLOT(helpDoc()));

    // Set up keyboard shortcuts.
    actOpen->setShortcut(QKeySequence::Open);
    actExit->setShortcut(QKeySequence::Quit);
    actShowFirstLevel->setShortcut(QKeySequence(Qt::ALT + Qt::Key_1));
    actExpandAll->setShortcut(QKeySequence(Qt::ALT + Qt::Key_2));
    actCollapseAll->setShortcut(QKeySequence(Qt::ALT + Qt::Key_3));

    // Create the menus.
    menubar_ = new QMenuBar(this);
    setMenuBar(menubar_);
    menuFile_ = new QMenu("&File", menubar_);
    QMenu* menuEdit = new QMenu("&Edit", menubar_);
    QMenu* menuView = new QMenu("&View", menubar_);
    QMenu* menuHelp = new QMenu("&Help", menubar_);
    menubar_->addAction(menuFile_->menuAction());
    menubar_->addAction(menuEdit->menuAction());
    menubar_->addAction(menuView->menuAction());
    menubar_->addAction(menuHelp->menuAction());
    menuFile_->addAction(actOpen);
    menuFile_->addAction(actSaveAs);
    createRecentFileActions();
    updateRecentFileActions();
    menuFile_->addSeparator();
    menuFile_->addAction(actExit);
    menuEdit->addAction(actBinLocations);
    menuView->addAction(actShowFirstLevel);
    menuView->addAction(actExpandAll);
    menuView->addAction(actCollapseAll);
    menuView->addSeparator();
    menuView->addAction(actDisplayKeys);
    menuView->addAction(actDisplayLabels);
    menuHelp->addAction(actHelpDoc);
    menuHelp->addSeparator();
    menuHelp->addAction(actCudaInfo);
    menuHelp->addAction(actAbout);

    // Create the network access manager to check the current version.
    version_url_ = "http://www.oerc.ox.ac.uk/~ska/oskar/current_version.txt";
    networkManager_ = new QNetworkAccessManager(this);
    connect(networkManager_, SIGNAL(finished(QNetworkReply*)),
                this, SLOT(processNetworkReply(QNetworkReply*)));

    // Load the settings.
    QSettings settings;
    restoreGeometry(settings.value("main_window/geometry").toByteArray());
    restoreState(settings.value("main_window/state").toByteArray());

    // Set up binary path names.
    binary_interferometer_ = settings.value("binaries/interferometer",
            "oskar_sim_interferometer").toString();
    binary_beam_pattern_ = settings.value("binaries/beam_pattern",
            "oskar_sim_beam_pattern").toString();
    binary_imager_ = settings.value("binaries/imager",
            "oskar_imager").toString();
    binary_fit_element_data_ = settings.value("binaries/fit_element_data",
            "oskar_fit_element_data").toString();
    binary_cuda_info_ = settings.value("binaries/cuda_info",
            "oskar_cuda_system_info").toString();

    // Set the selected binary.
    selectedBinary_ = selector_->currentText();
    binaryChanged(selectedBinary_);

    // Set the focus policy.
    setFocusPolicy(Qt::StrongFocus);
    setFocus(Qt::ActiveWindowFocusReason);

    // Restore the scroll bar position.
    // A single-shot timer is used to do this after the main event loop starts.
    QTimer::singleShot(0, view_, SLOT(restorePosition()));

    // Check for updates.
    // A single-shot timer is used to do this after the main event loop starts.
    QTimer::singleShot(500, this, SLOT(checkForUpdate()));
}

MainWindow::~MainWindow()
{
    delete handler_;
    delete settings_;
}

void MainWindow::openSettings(QString filename)
{
    if (settingsFile_.isEmpty() && settings_->is_modified())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(mainTitle_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Opening a new file will discard unsaved modifications.");
        msgBox.setInformativeText("Do you want to proceed?");
        msgBox.setStandardButtons(QMessageBox::Open | QMessageBox::Cancel);
        if (msgBox.exec() == QMessageBox::Cancel)
            return;
    }

    // Check if the supplied filename is empty, and prompt to open file if so.
    if (filename.isEmpty())
    {
        filename = QFileDialog::getOpenFileName(this, "Open Settings",
                settingsFile_);
    }

    // Open the file if one was selected.
    if (!filename.isEmpty())
    {
        view_->saveExpanded(selector_->currentText());
        settingsFile_ = filename;
        model_->load_settings_file(filename);
        setWindowTitle(mainTitle_ + " [" + filename + "]");
        updateRecentFileList();
        view_->restoreExpanded(selector_->currentText());

        // Check the version of OSKAR that created the file, if it exists.
        if (QFile::exists(filename))
        {
            QString ver = QString::fromStdString(settings_->file_version());
            QStringList vers = ver.split('.');
            int major = (vers.size() > 0) ? vers[0].toInt() : 0;
            int minor = (vers.size() > 1) ? vers[1].toInt() : 0;

            if (ver.isEmpty())
            {
                QMessageBox msgBox(this);
                msgBox.setWindowTitle(mainTitle_);
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.setText("This file was created by an unknown version "
                        "of OSKAR.");
                msgBox.setInformativeText("Please check all settings "
                        "carefully, as the keys may have changed.");
                msgBox.exec();
            }
            else if (major != ((OSKAR_VERSION & 0xFF0000) >> 16) ||
                    minor != (OSKAR_VERSION & 0x00FF00) >> 8)
            {
                QMessageBox msgBox(this);
                msgBox.setWindowTitle(mainTitle_);
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.setText(QString("This file was created by OSKAR %1.")
                        .arg(ver));
                msgBox.setInformativeText("Please check all settings "
                        "carefully, as the keys may have changed since that "
                        "version. You should open the settings file in a text "
                        "editor to verify the keys and update the version "
                        "number in the file, which will suppress future "
                        "warnings of this type.");
                msgBox.exec();
            }
        }
    }
}

void MainWindow::saveSettingsAs(QString filename)
{
    // Check if the supplied filename is empty, and prompt to save file if so.
    if (filename.isEmpty())
    {
        filename = QFileDialog::getSaveFileName(this, "Save Settings",
                settingsFile_);
    }

    // Save the file if one was selected.
    if (!filename.isEmpty())
    {
        // Remove any existing file with this name.
        if (QFile::exists(filename))
        {
            if (!QFile::remove(filename))
            {
                QMessageBox::critical(this, mainTitle_,
                        QString("Could not overwrite file at %1").arg(filename),
                        QMessageBox::Ok);
                return;
            }
        }

        // Try to open the file for writing.
        QFile file(filename);
        bool result = file.open(QFile::ReadWrite);
        file.close();
        if (!result) return;

        view_->saveExpanded(selector_->currentText());
        settingsFile_ = filename;
        model_->save_settings_file(filename);
        setWindowTitle(mainTitle_ + " [" + filename + "]");
        updateRecentFileList();
        view_->restoreExpanded(selector_->currentText());
    }
}

// =========================================================  Protected methods.

void MainWindow::closeEvent(QCloseEvent* event)
{
    // Save the state of the tree view.
    view_->saveExpanded(selector_->currentText());
    view_->savePosition();

    QSettings settings;
    settings.setValue("main_window/geometry", saveGeometry());
    settings.setValue("main_window/state", saveState());
    settings.setValue("binaries/interferometer", binary_interferometer_);
    settings.setValue("binaries/beam_pattern", binary_beam_pattern_);
    settings.setValue("binaries/imager", binary_imager_);
    settings.setValue("binaries/fit_element_data", binary_fit_element_data_);
    settings.setValue("binaries/cuda_info", binary_cuda_info_);

    if (settingsFile_.isEmpty() && settings_->is_modified())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(mainTitle_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("This action will discard unsaved modifications.");
        msgBox.setInformativeText("Do you want to proceed?");
        msgBox.setStandardButtons(QMessageBox::Discard | QMessageBox::Cancel |
                QMessageBox::Save);
        msgBox.setDefaultButton(QMessageBox::Discard);
        int ret = msgBox.exec();
        if (ret == QMessageBox::Discard)
        {
            event->accept();
        }
        else if (ret == QMessageBox::Save)
        {
            saveSettingsAs();

            // Check if the save failed for any reason.
            if (settingsFile_.isEmpty())
                event->ignore();
            else
                event->accept();
        }
        else
        {
            event->ignore();
        }
    }
}

// =========================================================  Private slots.

void MainWindow::about()
{
    oskar_About aboutDialog(this);
    aboutDialog.exec();
}

void MainWindow::binaryChanged(const QString& value)
{
    // Clear the filter text.
    filterBox_->clear();

    // FIXME(FD) Better way to do this is to query the binaries themselves
    // for their settings.
    if (value == "oskar_sim_interferometer")
        swapSettings(oskar_sim_interferometer_XML_STR);
    else if (value == "oskar_sim_beam_pattern")
        swapSettings(oskar_sim_beam_pattern_XML_STR);
    else if (value == "oskar_imager")
        swapSettings(oskar_imager_XML_STR);
    else if (value == "oskar_fit_element_data")
        swapSettings(oskar_fit_element_data_XML_STR);
}

void MainWindow::binLocations()
{
    oskar_BinaryLocations binaryLocations(this);
    binaryLocations.setBeamPattern(binary_beam_pattern_);
    binaryLocations.setCudaSystemInfo(binary_cuda_info_);
    binaryLocations.setFitElementData(binary_fit_element_data_);
    binaryLocations.setImager(binary_imager_);
    binaryLocations.setInterferometer(binary_interferometer_);
    if (binaryLocations.exec() == QDialog::Accepted)
    {
        binary_beam_pattern_ = binaryLocations.beamPattern();
        binary_cuda_info_ = binaryLocations.cudaSystemInfo();
        binary_fit_element_data_ = binaryLocations.fitElementData();
        binary_imager_ = binaryLocations.imager();
        binary_interferometer_ = binaryLocations.interferometer();
    }
}

void MainWindow::checkForUpdate()
{
    QNetworkRequest request;
    request.setUrl(QUrl(version_url_));
    request.setRawHeader("User-Agent", "OSKAR " OSKAR_VERSION_STR);
    networkManager_->get(request);
}

void MainWindow::cudaInfo()
{
    // Check that the binary actually exists.
    QProcess process;
    process.start(binary_cuda_info_);
    if (!process.waitForStarted())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(mainTitle_);
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.setText("The CUDA system info binary could not be found.");
        msgBox.setInformativeText("Please edit the binary location and try again.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        binLocations();
        return;
    }

    oskar_CudaInfoDisplay infoDisplay(binary_cuda_info_, this);
    infoDisplay.exec();
}

void MainWindow::helpDoc()
{
    oskar_DocumentationDisplay helpDisplay(this);
    helpDisplay.exec();
}

void MainWindow::openRecentFile()
{
    QAction* act = qobject_cast<QAction*>(sender());
    if (act)
    {
        QString filename = act->data().toString();

        if (QFile::exists(filename))
        {
            // If the file exists, then open it.
            openSettings(filename);
        }
        else
        {
            // If the file doesn't exist, display a warning message.
            QMessageBox::critical(this, mainTitle_,
                    QString("File %1 not found").arg(filename),
                    QMessageBox::Ok);

            // Remove it from the list.
            QSettings settings;
            QStringList files = settings.value("recent_files/files").toStringList();
            files.removeAll(filename);
            settings.setValue("recent_files/files", files);
            updateRecentFileActions();
        }
    }
}

void MainWindow::processNetworkReply(QNetworkReply* reply)
{
    QString currentVerString;
    // Obtain the current version of OSKAR from the server.
    if (reply->request().url().toString() == version_url_)
    {
        QByteArray data = reply->readLine();
        if (data.endsWith('\n')) data.chop(1);
        if (data.isEmpty() || reply->error() != QNetworkReply::NoError)
            return;
        currentVerString = QString(data);
    }

    // Split version strings into components.
    QRegExp rx("\\.|-");
    QStringList currentVer = currentVerString.split(rx);
    QStringList thisVer = QString(OSKAR_VERSION_STR).split(rx);

    // Version strings should be of the format major.minor.patch-flag
    // where flag is optional. Therefore they will be of length 3 or 4 components.
    bool validVer = currentVer.size() == 3 || currentVer.size() == 4;
    validVer = validVer && (thisVer.size() == 3 || thisVer.size() == 4);
    if (!validVer) return;

    // If the server (current) version is flagged (i.e. has a hyphen tag after
    // its version number) - *NEVER* prompt for an update.
    // This avoids asking people to download release candidates, for example.
    if (currentVer.size() == 4)
        return;

    // If the current version is newer than the version of this code
    // notify of the update.
    bool isNewMajorVer   = currentVer[0].toInt() >  thisVer[0].toInt();
    bool isEqualMajorVer = currentVer[0].toInt() == thisVer[0].toInt();
    bool isNewMinorVer   = currentVer[1].toInt() >  thisVer[1].toInt();
    bool isEqualMinorVer = currentVer[1].toInt() == thisVer[1].toInt();
    bool isNewPatchVer   = currentVer[2].toInt() >  thisVer[2].toInt();

    // Conditions to prompt for update:
    // 1) Major version is out of date.
    // 2) Major version not changed but minor version out of date.
    // 3) Major and minor versions not changed but patch out of date.
    if (isNewMajorVer || (isEqualMajorVer && isNewMinorVer) ||
            (isEqualMajorVer && isEqualMinorVer && isNewPatchVer))
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(mainTitle_);
        msgBox.setIcon(QMessageBox::Information);
        msgBox.setTextFormat(Qt::RichText);
        msgBox.setText(QString("A newer version of OSKAR (%1) is "
                "<a href=\"http://www.oerc.ox.ac.uk/~ska/oskar/\">"
                "available for download</a>.").arg(currentVerString));
        msgBox.setInformativeText("Please update your installed version "
                "as soon as possible.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
    }

    // Mark the reply for deletion.
    reply->deleteLater();
}

void MainWindow::runButton()
{
    QString run_binary;
    if (selector_->currentText() == "oskar_sim_interferometer")
        run_binary = binary_interferometer_;
    else if (selector_->currentText() == "oskar_sim_beam_pattern")
        run_binary = binary_beam_pattern_;
    else if (selector_->currentText() == "oskar_imager")
        run_binary = binary_imager_;
    else if (selector_->currentText() == "oskar_fit_element_data")
        run_binary = binary_fit_element_data_;

    // Block signals from the model to prevent an erroneous warning saying
    // the settings file was updated by another program.
    // (oskar_SettingsView::fileReloaded)
    // FIXME(FD) Block signals from the model. May not be needed now.
//    bool oldState = model_->blockSignals(true);

    // Save settings if they are not already saved.
    if (settingsFile_.isEmpty())
    {
        // Get the name of the new settings file (return if empty).
        saveSettingsAs(QString());
        if (settingsFile_.isEmpty())
        {
            // FIXME(FD) Restore signals from the model. May not be needed now.
//            model_->blockSignals(oldState);
            return;
        }
    }

    // Check that the selected binary actually exists.
    QProcess process;
    process.start(run_binary);
    if (!process.waitForStarted())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(mainTitle_);
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.setText("The selected binary could not be found.");
        msgBox.setInformativeText("Please edit the binary location and try again.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        binLocations();
        // FIXME(FD) Restore signals from the model. May not be needed now.
//        model_->blockSignals(oldState);
        return;
    }

    // Run simulation.
    oskar_RunDialog dialog(this);
    dialog.start(run_binary, settingsFile_);
    dialog.exec();

    // FIXME(FD) Restore signals from the model. May not be needed now.
//    model_->blockSignals(oldState);
}


// =========================================================  Private methods.

void MainWindow::swapSettings(const char* xml)
{
    // Save the current settings view.
    if (view_->isVisible())
        view_->saveExpanded(selectedBinary_);

    // Declare new settings, reload settings file, and reset the model
    model_->beginReset();
    oskar::settings_declare_xml(settings_, xml);
    model_->load_settings_file(settingsFile_);
    model_->endReset();

    // Restore the view for this binary.
    view_->restoreExpanded(selector_->currentText());
    view_->resizeColumnToContents(0);
    selectedBinary_ = selector_->currentText();
}


void MainWindow::createRecentFileActions()
{
    for (int i = 0; i < MaxRecentFiles; ++i)
    {
        recentFiles_[i] = new QAction(this);
        recentFiles_[i]->setVisible(false);
        connect(recentFiles_[i], SIGNAL(triggered()),
                this, SLOT(openRecentFile()));
    }
    separator_ = menuFile_->addSeparator();
    for (int i = 0; i < MaxRecentFiles; ++i)
    {
        menuFile_->addAction(recentFiles_[i]);
    }
}


void MainWindow::updateRecentFileActions()
{
    QSettings settings;
    QStringList files = settings.value("recent_files/files").toStringList();

    int num_files = qMin(files.size(), (int)MaxRecentFiles);

    for (int i = 0; i < num_files; ++i)
    {
        QFileInfo info(files[i]);
        QString txt = QString("&%1 %2  [%3]").arg(i+1).arg(info.fileName()).
                arg(info.absolutePath());
        recentFiles_[i]->setText(txt);
        recentFiles_[i]->setData(files[i]);
        recentFiles_[i]->setVisible(true);
    }
    for (int i = num_files; i < MaxRecentFiles; ++i)
    {
        recentFiles_[i]->setVisible(false);
    }
    separator_->setVisible(num_files > 0);
}


void MainWindow::updateRecentFileList()
{
    QSettings settings;
    QStringList files = settings.value("recent_files/files").toStringList();
    QFileInfo file(settingsFile_);
    files.removeAll(file.absoluteFilePath());
    files.prepend(file.absoluteFilePath());
    while (files.size() > MaxRecentFiles)
        files.removeLast();

    settings.setValue("recent_files/files", files);

    updateRecentFileActions();
}

}
