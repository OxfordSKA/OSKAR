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

#include "apps/oskar_MainWindow.h"
#include "apps/lib/oskar_sim_beam_pattern.h"
#include "apps/lib/oskar_sim_interferometer.h"
#include "apps/lib/oskar_imager.h"
#include "apps/lib/oskar_SettingsModelApps.h"
#include "widgets/oskar_About.h"
#include "widgets/oskar_CudaInfoDisplay.h"
#include "widgets/oskar_SettingsDelegate.h"
#include "widgets/oskar_SettingsItem.h"
#include "widgets/oskar_SettingsView.h"
#include "utility/oskar_get_error_string.h"

#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QFileDialog>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QMessageBox>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QKeySequence>
#include <QtCore/QModelIndex>
#include <QtCore/QTimer>

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
    layout_ = new QVBoxLayout(widget_);

    // Create and set up the settings model.
    model_ = new oskar_SettingsModelApps(widget_);
    modelProxy_ = new oskar_SettingsModelFilter(widget_);
    modelProxy_->setSourceModel(model_);

    // Create and set up the settings view.
    view_ = new oskar_SettingsView(widget_);
    oskar_SettingsDelegate* delegate = new oskar_SettingsDelegate(view_,
            widget_);
    view_->setModel(modelProxy_);
    view_->setItemDelegate(delegate);
    view_->resizeColumnToContents(0);
    layout_->addWidget(view_);

    // Create and set up the actions.
    QAction* actOpen = new QAction("&Open...", this);
    QAction* actSaveAs = new QAction("&Save As...", this);
    QAction* actExit = new QAction("E&xit", this);
    actHideUnset_ = new QAction("&Hide Unset Items", this);
    actHideUnset_->setCheckable(true);
    QAction* actShowFirstLevel = new QAction("Show &First Level", this);
    QAction* actExpandAll = new QAction("&Expand All", this);
    QAction* actCollapseAll = new QAction("&Collapse All", this);
    QAction* actAbout = new QAction("&About OSKAR...", this);
    QAction* actCudaInfo = new QAction("CUDA System Info...", this);
    QAction* actRunInterferometer = new QAction("Run &Interferometer", this);
    QAction* actRunBeamPattern = new QAction("Run &Beam Pattern", this);
    QAction* actRunImager = new QAction("Run I&mager", this);
    connect(actOpen, SIGNAL(triggered()), this, SLOT(openSettings()));
    connect(actSaveAs, SIGNAL(triggered()), this, SLOT(saveSettingsAs()));
    connect(actExit, SIGNAL(triggered()), this, SLOT(close()));
    connect(actHideUnset_, SIGNAL(toggled(bool)),
            this, SLOT(setHideIfUnset(bool)));
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

    // Set up keyboard shortcuts.
    actOpen->setShortcut(QKeySequence::Open);
    actExit->setShortcut(QKeySequence::Quit);
    actHideUnset_->setShortcut(QKeySequence(Qt::ALT + Qt::Key_H));
    actShowFirstLevel->setShortcut(QKeySequence(Qt::ALT + Qt::Key_1));
    actExpandAll->setShortcut(QKeySequence(Qt::ALT + Qt::Key_2));
    actCollapseAll->setShortcut(QKeySequence(Qt::ALT + Qt::Key_3));

    // Create the menus.
    menubar_ = new QMenuBar(this);
    setMenuBar(menubar_);
    menuFile_ = new QMenu("&File", menubar_);
    QMenu* menuView = new QMenu("&View", menubar_);
    QMenu* menuRun = new QMenu("&Run", menubar_);
    menubar_->addAction(menuFile_->menuAction());
    menubar_->addAction(menuView->menuAction());
    menubar_->addAction(menuRun->menuAction());
    menuFile_->addAction(actOpen);
    menuFile_->addAction(actSaveAs);
    createRecentFileActions();
    updateRecentFileActions();
    menuFile_->addSeparator();
    menuFile_->addAction(actExit);
    menuView->addAction(actHideUnset_);
    menuView->addSeparator();
    menuView->addAction(actShowFirstLevel);
    menuView->addAction(actExpandAll);
    menuView->addAction(actCollapseAll);
    menuView->addSeparator();
    menuView->addAction(actAbout);
    menuView->addAction(actCudaInfo);
    menuRun->addAction(actRunInterferometer);
    menuRun->addAction(actRunBeamPattern);
    menuRun->addAction(actRunImager);

    // Create the toolbar.
    QToolBar* toolbar = new QToolBar(this);
    toolbar->setObjectName("Run");
    toolbar->addAction(actRunInterferometer);
    toolbar->addAction(actRunBeamPattern);
    toolbar->addAction(actRunImager);
    addToolBar(Qt::TopToolBarArea, toolbar);

    // Load the settings.
    QSettings settings;
    restoreGeometry(settings.value("main_window/geometry").toByteArray());
    restoreState(settings.value("main_window/state").toByteArray());

    // First, restore the expanded items.
    view_->restoreExpanded();

    // Optionally hide unset items.
    actHideUnset_->setChecked(settings.value("main_window/hide_unset_items").toBool());

    // Restore the scroll bar position.
    // A single-shot timer is used to do this after the main event loop starts.
    QTimer::singleShot(0, view_, SLOT(restorePosition()));
}

void oskar_MainWindow::openSettings(QString filename, bool check)
{
    if (settingsFile_.isEmpty() && check)
    {
        int ret = QMessageBox::warning(this, "OSKAR",
                "Opening a new file will discard any current unsaved modifications.\n"
                "Do you want to proceed?",
                QMessageBox::Ok | QMessageBox::Cancel);
        if (ret == QMessageBox::Cancel)
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
    QMainWindow::closeEvent(event);
}

// =========================================================  Private slots.

void oskar_MainWindow::about()
{
    oskar_About aboutDialog(this);
    aboutDialog.exec();
}

void oskar_MainWindow::cudaInfo()
{
    oskar_CudaInfoDisplay infoDisplay(this);
    infoDisplay.exec();
}

void oskar_MainWindow::runBeamPattern()
{
    run_function_ = &oskar_sim_beam_pattern;
    runButton();
}

void oskar_MainWindow::runInterferometer()
{
    run_function_ = &oskar_sim_interferometer;
    runButton();
}

void oskar_MainWindow::runImager()
{
    run_function_ = &oskar_imager;
    runButton();
}

void oskar_MainWindow::setHideIfUnset(bool value)
{
    view_->saveExpanded();
    modelProxy_->setHideIfUnset(value);
    view_->restoreExpanded();
    view_->update();
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
            openSettings(filename, true);
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


// =========================================================  Private methods.

void oskar_MainWindow::runButton()
{
    // Save settings if they are not already saved.
    if (settingsFile_.isEmpty())
    {
        // Get the name of the new settings file (return if empty).
        saveSettingsAs(QString());
        if (settingsFile_.isEmpty())
            return;
    }

    // Get the (list of) output file names.
    QStringList outputFiles;
    QStringList keys = model_->data(QModelIndex(),
            oskar_SettingsModel::OutputKeysRole).toStringList();
    for (int i = 0; i < keys.size(); ++i)
    {
        const oskar_SettingsItem* it = model_->getItem(keys[i]);
        QString file = it->value().toString();
        outputFiles.append(file);
    }

    // Run simulation recursively.
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    run(0, outputFiles);
    QApplication::restoreOverrideCursor();

    // Restore the output files.
    for (int i = 0; i < keys.size(); ++i)
    {
        model_->setValue(keys[i], outputFiles[i]);
    }
}

void oskar_MainWindow::run(int depth, QStringList outputFiles)
{
    QByteArray settings = settingsFile_.toAscii();
    QStringList iterationKeys = model_->data(QModelIndex(),
            oskar_SettingsModel::IterationKeysRole).toStringList();
    if (iterationKeys.size() == 0)
    {
        int error = (*run_function_)(settings);
        if (error)
        {
            fprintf(stderr, "\n>>> Run failed (code %d): %s.\n", error,
                    oskar_get_error_string(error));
        }
    }
    else
    {
        QStringList outputKeys = model_->data(QModelIndex(),
                oskar_SettingsModel::OutputKeysRole).toStringList();
        QString key = iterationKeys[depth];
        const oskar_SettingsItem* item = model_->getItem(key);
        QVariant start = item->value();
        QVariant inc = item->iterationInc();

        // Modify all the output file names with the subkey name.
        for (int i = 0; i < outputFiles.size(); ++i)
        {
            if (!outputFiles[i].isEmpty())
            {
                QString separator = (depth == 0) ? "__" : "_";
                outputFiles[i].append(separator + item->subkey());
            }
        }
        QStringList outputFilesStart = outputFiles;

        for (int i = 0; i < item->iterationNum(); ++i)
        {
            // Set the settings file parameter.
            QVariant val;
            if (item->type() == oskar_SettingsItem::INT)
                val = QVariant(start.toInt() + i * inc.toInt());
            else if (item->type() == oskar_SettingsItem::DOUBLE)
                val = QVariant(start.toDouble() + i * inc.toDouble());
            model_->setValue(key, val);

            // Modify all the output file names with the parameter value.
            for (int i = 0; i < outputFiles.size(); ++i)
            {
                if (!outputFiles[i].isEmpty())
                {
                    outputFiles[i].append("_" + val.toString());
                    model_->setValue(outputKeys[i], outputFiles[i]);
                }
            }

            // Check if recursion depth has been reached.
            if (depth < iterationKeys.size() - 1)
            {
                // If not, then call this function again.
                run(depth + 1, outputFiles);
            }
            else
            {
                // Run the simulation with these settings.
                int error = (*run_function_)(settings);
                if (error)
                {
                    fprintf(stderr, "\n>>> Run failed (code %d): %s.\n", error,
                            oskar_get_error_string(error));
                }
            }

            // Restore the list of output file names.
            outputFiles = outputFilesStart;
        }

        // Restore initial value.
        model_->setValue(key, start);
    }
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

