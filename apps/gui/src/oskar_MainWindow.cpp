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

#include <apps/gui/oskar_MainWindow.h>

#include <apps/gui/oskar_About.h>
#include <apps/gui/oskar_BinaryLocations.h>
#include <apps/gui/oskar_CudaInfoDisplay.h>
#include <apps/gui/oskar_DocumentationDisplay.h>
#include <apps/gui/oskar_RunDialog.h>
#include <oskar_SettingsDelegate.h>
#include <oskar_SettingsItem.h>
#include <oskar_SettingsModelXML.h>
#include <oskar_SettingsView.h>

#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QCloseEvent>
#include <QtGui/QFileDialog>
#include <QtGui/QFormLayout>
#include <QtGui/QLineEdit>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QMessageBox>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QKeySequence>
#include <QtCore/QModelIndex>
#include <QtCore/QProcess>
#include <QtCore/QTimer>
#include <QtCore/QSettings>
#include <QtNetwork/QNetworkAccessManager>
#include <QtNetwork/QNetworkReply>

oskar_MainWindow::oskar_MainWindow(QWidget* parent)
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
    QFormLayout* formLayout = new QFormLayout;

    // Create and set up the settings model.
    //model_ = new oskar_SettingsModelApps(widget_);
    model_ = new oskar::SettingsModelXML(widget_);
    modelProxy_ = new oskar_SettingsModelFilter(widget_);
    modelProxy_->setSourceModel(model_);

    // Create the search box.
    QLineEdit* filterBox = new QLineEdit(widget_);
    formLayout->addRow("Settings View Filter", filterBox);
    formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    connect(filterBox, SIGNAL(textChanged(QString)),
            modelProxy_, SLOT(setFilterText(QString)));
    v_layout->addLayout(formLayout);

    // Create and set up the settings view.
    view_ = new oskar_SettingsView(widget_);
    oskar_SettingsDelegate* delegate = new oskar_SettingsDelegate(view_,
            widget_);
    view_->setModel(modelProxy_);
    view_->setItemDelegate(delegate);
    view_->resizeColumnToContents(0);
    connect(model_, SIGNAL(fileReloaded()), view_, SLOT(fileReloaded()));
    v_layout->addWidget(view_);

    // Create and set up the actions.
    QAction* actOpen = new QAction("&Open...", this);
    QAction* actSaveAs = new QAction("&Save As...", this);
    QAction* actExit = new QAction("E&xit", this);
    QAction* actBinLocations = new QAction("&Binary Locations...", this);
    actHideUnset_ = new QAction("&Hide Unset Items", this);
    actHideUnset_->setCheckable(true);
    QAction* actShowFirstLevel = new QAction("Show &First Level", this);
    QAction* actExpandAll = new QAction("&Expand All", this);
    QAction* actCollapseAll = new QAction("&Collapse All", this);
    QAction* actRunInterferometer = new QAction("Run &Interferometer", this);
    QAction* actRunBeamPattern = new QAction("Run &Beam Pattern", this);
    QAction* actRunImager = new QAction("Run I&mager", this);
    QAction* actRunFitElementData = new QAction("Run Element Data Fit", this);
    QAction* actHelpDoc = new QAction("Documentation...", this);
    QAction* actCudaInfo = new QAction("CUDA System Info...", this);
    QAction* actAbout = new QAction("&About OSKAR...", this);
    connect(actOpen, SIGNAL(triggered()), this, SLOT(openSettings()));
    connect(actSaveAs, SIGNAL(triggered()), this, SLOT(saveSettingsAs()));
    connect(actExit, SIGNAL(triggered()), this, SLOT(close()));
    connect(actBinLocations, SIGNAL(triggered()), this, SLOT(binLocations()));
    connect(actHideUnset_, SIGNAL(toggled(bool)),
            this, SLOT(setHideUnsetItems(bool)));
    connect(actShowFirstLevel, SIGNAL(triggered()),
            view_, SLOT(showFirstLevel()));
    connect(actExpandAll, SIGNAL(triggered()), view_, SLOT(expandSettingsTree()));
    connect(actCollapseAll, SIGNAL(triggered()), view_, SLOT(collapseAll()));
    connect(actAbout, SIGNAL(triggered()), this, SLOT(about()));
    connect(actCudaInfo, SIGNAL(triggered()), this, SLOT(cudaInfo()));
    connect(actRunInterferometer, SIGNAL(triggered()),
            this, SLOT(runInterferometer()));
    connect(actRunBeamPattern, SIGNAL(triggered()),
            this, SLOT(runBeamPattern()));
    connect(actRunImager, SIGNAL(triggered()), this, SLOT(runImager()));
    connect(actRunFitElementData, SIGNAL(triggered()), this,
            SLOT(runFitElementData()));
    connect(actHelpDoc, SIGNAL(triggered()), this, SLOT(helpDoc()));

    // Set up keyboard shortcuts.
    actOpen->setShortcut(QKeySequence::Open);
    actExit->setShortcut(QKeySequence::Quit);
    actHideUnset_->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_H));
    actShowFirstLevel->setShortcut(QKeySequence(Qt::ALT + Qt::Key_1));
    actExpandAll->setShortcut(QKeySequence(Qt::ALT + Qt::Key_2));
    actCollapseAll->setShortcut(QKeySequence(Qt::ALT + Qt::Key_3));

    // Create the menus.
    menubar_ = new QMenuBar(this);
    setMenuBar(menubar_);
    menuFile_ = new QMenu("&File", menubar_);
    QMenu* menuEdit = new QMenu("&Edit", menubar_);
    QMenu* menuView = new QMenu("&View", menubar_);
    QMenu* menuRun = new QMenu("&Run", menubar_);
    QMenu* menuHelp = new QMenu("&Help", menubar_);
    menubar_->addAction(menuFile_->menuAction());
    menubar_->addAction(menuEdit->menuAction());
    menubar_->addAction(menuView->menuAction());
    menubar_->addAction(menuRun->menuAction());
    menubar_->addAction(menuHelp->menuAction());
    menuFile_->addAction(actOpen);
    menuFile_->addAction(actSaveAs);
    createRecentFileActions();
    updateRecentFileActions();
    menuFile_->addSeparator();
    menuFile_->addAction(actExit);
    menuEdit->addAction(actBinLocations);
    menuView->addAction(actHideUnset_);
    menuView->addSeparator();
    menuView->addAction(actShowFirstLevel);
    menuView->addAction(actExpandAll);
    menuView->addAction(actCollapseAll);
    menuRun->addAction(actRunInterferometer);
    menuRun->addAction(actRunBeamPattern);
    menuRun->addAction(actRunImager);
    menuRun->addAction(actRunFitElementData);
    menuHelp->addAction(actHelpDoc);
    menuHelp->addSeparator();
    menuHelp->addAction(actCudaInfo);
    menuHelp->addAction(actAbout);

    // Create the toolbar.
    QToolBar* toolbar = new QToolBar(this);
    toolbar->setObjectName("Run");
    toolbar->addAction(actRunInterferometer);
    toolbar->addAction(actRunBeamPattern);
    toolbar->addAction(actRunImager);
    toolbar->addAction(actRunFitElementData);
    addToolBar(Qt::TopToolBarArea, toolbar);

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

    // First, restore the expanded items.
    view_->restoreExpanded();

    // Optionally hide unset items.
    actHideUnset_->setChecked(settings.value("main_window/hide_unset_items").toBool());

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

void oskar_MainWindow::openSettings(QString filename)
{
    if (settingsFile_.isEmpty() && model_->isModified())
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
        view_->saveExpanded();
        settingsFile_ = filename;
        model_->loadSettingsFile(filename);
        setWindowTitle(mainTitle_ + " [" + filename + "]");
        updateRecentFileList();
        view_->restoreExpanded();

        // Check the version of OSKAR that created the file, if it exists.
        if (QFile::exists(filename))
        {
            QString ver = model_->version();
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

void oskar_MainWindow::saveSettingsAs(QString filename)
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

        view_->saveExpanded();
        settingsFile_ = filename;
        model_->saveSettingsFile(filename);
        setWindowTitle(mainTitle_ + " [" + filename + "]");
        updateRecentFileList();
        view_->restoreExpanded();
    }
}

// =========================================================  Protected methods.

void oskar_MainWindow::closeEvent(QCloseEvent* event)
{
    // Save the state of the tree view.
    view_->saveExpanded();
    view_->savePosition();

    QSettings settings;
    settings.setValue("main_window/geometry", saveGeometry());
    settings.setValue("main_window/state", saveState());
    settings.setValue("main_window/hide_unset_items", actHideUnset_->isChecked());
    settings.setValue("binaries/interferometer", binary_interferometer_);
    settings.setValue("binaries/beam_pattern", binary_beam_pattern_);
    settings.setValue("binaries/imager", binary_imager_);
    settings.setValue("binaries/fit_element_data", binary_fit_element_data_);
    settings.setValue("binaries/cuda_info", binary_cuda_info_);

    if (settingsFile_.isEmpty() && model_->isModified())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(mainTitle_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Exiting OSKAR will discard unsaved modifications.");
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

void oskar_MainWindow::about()
{
    oskar_About aboutDialog(this);
    aboutDialog.exec();
}

void oskar_MainWindow::binLocations()
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

void oskar_MainWindow::checkForUpdate()
{
    QNetworkRequest request;
    request.setUrl(QUrl(version_url_));
    request.setRawHeader("User-Agent", "OSKAR " OSKAR_VERSION_STR);
    networkManager_->get(request);
}

void oskar_MainWindow::cudaInfo()
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

void oskar_MainWindow::helpDoc()
{
    oskar_DocumentationDisplay helpDisplay(this);
    helpDisplay.exec();
}

void oskar_MainWindow::openRecentFile()
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

void oskar_MainWindow::processNetworkReply(QNetworkReply* reply)
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

void oskar_MainWindow::runBeamPattern()
{
    run_binary_ = binary_beam_pattern_;
    runButton();
}

void oskar_MainWindow::runInterferometer()
{
    run_binary_ = binary_interferometer_;
    runButton();
}

void oskar_MainWindow::runImager()
{
    run_binary_ = binary_imager_;
    runButton();
}

void oskar_MainWindow::runFitElementData()
{
    run_binary_ = binary_fit_element_data_;
    runButton();
}

void oskar_MainWindow::setHideUnsetItems(bool value)
{
    view_->saveExpanded();
    modelProxy_->setHideUnsetItems(value);
    view_->restoreExpanded();
    view_->update();
}


// =========================================================  Private methods.

void oskar_MainWindow::runButton()
{
    // Block signals from the model to prevent an erroneous warning saying
    // the settings file was updated by another program.
    // (oskar_SettingsView::fileReloaded)
    // FIXME this is a hack! find a better fix...
    bool oldState = model_->blockSignals(true);

    // Save settings if they are not already saved.
    if (settingsFile_.isEmpty())
    {
        // Get the name of the new settings file (return if empty).
        saveSettingsAs(QString());
        if (settingsFile_.isEmpty())
        {
            // FIXME Restore signals from the model.
            model_->blockSignals(oldState);
            return;
        }
    }

    // Check that the selected binary actually exists.
    QProcess process;
    process.start(run_binary_);
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
        // FIXME Restore signals from the model.
        model_->blockSignals(oldState);
        return;
    }

    // Run simulation.
    oskar_RunDialog dialog(this);
    dialog.start(run_binary_, settingsFile_);
    dialog.exec();

    // FIXME Restore signals from the model.
    model_->blockSignals(oldState);
}

void oskar_MainWindow::createRecentFileActions()
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


void oskar_MainWindow::updateRecentFileActions()
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


void oskar_MainWindow::updateRecentFileList()
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

