/*
 * Copyright (c) 2011, The University of Oxford
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
#include "widgets/oskar_SettingsDelegate.h"
#include "widgets/oskar_SettingsItem.h"
#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsView.h"
#include "utility/oskar_get_error_string.h"

#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QFileDialog>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtCore/QModelIndex>
#include <QtCore/QTimer>

oskar_MainWindow::oskar_MainWindow(QWidget* parent)
: QMainWindow(parent)
{
    // Create the central widget and main layout.
    setWindowTitle("OSKAR GUI");
    widget_ = new QWidget(this);
    setCentralWidget(widget_);
    layout_ = new QVBoxLayout(widget_);

    // Create and set up the settings model.
    model_ = new oskar_SettingsModel(widget_);
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
    QAction* actOpen = new QAction("Open...", this);
    QAction* actSaveAs = new QAction("Save Copy As...", this);
    QAction* actExit = new QAction("Exit", this);
    actHideUnset_ = new QAction("Hide Unset Items", this);
    actHideUnset_->setCheckable(true);
    QAction* actShowFirstLevel = new QAction("Show First Level", this);
    QAction* actExpandAll = new QAction("Expand All", this);
    QAction* actCollapseAll = new QAction("Collapse All", this);
    QAction* actRunInterferometer = new QAction("Run Interferometer", this);
    QAction* actRunBeamPattern = new QAction("Run Beam Pattern", this);
    connect(actOpen, SIGNAL(triggered()), this, SLOT(openSettings()));
    connect(actSaveAs, SIGNAL(triggered()), this, SLOT(saveAs()));
    connect(actExit, SIGNAL(triggered()), this, SLOT(close()));
    connect(actHideUnset_, SIGNAL(toggled(bool)),
            this, SLOT(setHideIfUnset(bool)));
    connect(actShowFirstLevel, SIGNAL(triggered()),
            view_, SLOT(showFirstLevel()));
    connect(actExpandAll, SIGNAL(triggered()), view_, SLOT(expandAll()));
    connect(actCollapseAll, SIGNAL(triggered()), view_, SLOT(collapseAll()));
    connect(actRunInterferometer, SIGNAL(triggered()),
            this, SLOT(runInterferometer()));
    connect(actRunBeamPattern, SIGNAL(triggered()),
            this, SLOT(runBeamPattern()));

    // Set up keyboard shortcuts.
    actOpen->setShortcut(QKeySequence::Open);
    actExit->setShortcut(QKeySequence::Quit);

    // Create the menus.
    QMenuBar* menubar = new QMenuBar(this);
    setMenuBar(menubar);
    QMenu* menuFile = new QMenu("File", menubar);
    QMenu* menuView = new QMenu("View", menubar);
    QMenu* menuRun = new QMenu("Run", menubar);
    menubar->addAction(menuFile->menuAction());
    menubar->addAction(menuView->menuAction());
    menubar->addAction(menuRun->menuAction());
    menuFile->addAction(actOpen);
    menuFile->addAction(actSaveAs);
    menuFile->addSeparator();
    menuFile->addAction(actExit);
    menuView->addAction(actHideUnset_);
    menuView->addSeparator();
    menuView->addAction(actShowFirstLevel);
    menuView->addAction(actExpandAll);
    menuView->addAction(actCollapseAll);
    menuRun->addAction(actRunInterferometer);
    menuRun->addAction(actRunBeamPattern);

    // Create the toolbar.
    QToolBar* toolbar = new QToolBar(this);
    toolbar->setObjectName("Run");
    toolbar->addAction(actRunInterferometer);
    toolbar->addAction(actRunBeamPattern);
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

void oskar_MainWindow::openSettings(QString filename)
{
    // Check if the supplied filename is empty, and prompt to open file if so.
    if (filename.isEmpty())
    {
        view_->saveExpanded();
        filename = QFileDialog::getOpenFileName(this, "Open Settings",
                settingsFile_);
    }

    // Set the file if one was selected.
    if (!filename.isEmpty())
    {
        settingsFile_ = filename;
        model_->loadSettingsFile(filename);
        setWindowTitle("OSKAR GUI [" + filename + "]");

        // Restore the expanded items.
        view_->restoreExpanded();
    }
}

void oskar_MainWindow::saveAs(QString filename)
{
    if (filename.isEmpty())
    {
        view_->saveExpanded();
        filename = QFileDialog::getSaveFileName(this, "Save Settings File As ...");
    }

    settingsFile_ = filename;
    if (settingsFile_.isEmpty()) return;

    // Save the settings file.
    view_->saveExpanded();
    model_->saveSettingsFile(settingsFile_);
    setWindowTitle("OSKAR GUI [" + settingsFile_ + "]");
    view_->restoreExpanded();
}


// Protected methods.

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

// Private slots.

void oskar_MainWindow::runBeamPattern()
{
    sim_function_ = &oskar_sim_beam_pattern;
    runButton();
}

void oskar_MainWindow::runInterferometer()
{
    sim_function_ = &oskar_sim_interferometer;
    runButton();
}

void oskar_MainWindow::setHideIfUnset(bool value)
{
    view_->saveExpanded();
    modelProxy_->setHideIfUnset(value);
    view_->restoreExpanded();
}

// Private members.

void oskar_MainWindow::runButton()
{
    // Save settings if they are not already saved.
    if (settingsFile_.isEmpty())
    {
        // Get the name of the new settings file (return if empty).
        settingsFile_ = QFileDialog::getSaveFileName(this, "Save Settings");
        if (settingsFile_.isEmpty())
            return;

        // Remove any existing file with this name.
        if (QFile::exists(settingsFile_))
            QFile::remove(settingsFile_);

        // Save the settings file.
        view_->saveExpanded();
        model_->saveSettingsFile(settingsFile_);
        setWindowTitle("OSKAR GUI [" + settingsFile_ + "]");
        view_->restoreExpanded();
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
    runSim(0, outputFiles);

    // Restore the output files.
    for (int i = 0; i < keys.size(); ++i)
    {
        model_->setValue(keys[i], outputFiles[i]);
    }
}

void oskar_MainWindow::runSim(int depth, QStringList outputFiles)
{
    QByteArray settings = settingsFile_.toAscii();
    QStringList iterationKeys = model_->data(QModelIndex(),
            oskar_SettingsModel::IterationKeysRole).toStringList();
    if (iterationKeys.size() == 0)
    {
        int error = (*sim_function_)(settings);
        if (error)
        {
            fprintf(stderr, ">>> Run failed (code %d): %s.\n", error,
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
                runSim(depth + 1, outputFiles);
            }
            else
            {
                // Run the simulation with these settings.
                int error = (*sim_function_)(settings);
                if (error)
                {
                    fprintf(stderr, ">>> Run failed (code %d): %s.\n", error,
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
