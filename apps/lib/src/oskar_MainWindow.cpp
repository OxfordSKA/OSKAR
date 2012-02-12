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

#include "apps/lib/oskar_MainWindow.h"
#include "apps/lib/oskar_sim.h"
#include "widgets/oskar_SettingsDelegate.h"
#include "widgets/oskar_SettingsItem.h"
#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsView.h"
#include "utility/oskar_get_error_string.h"

#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QFileDialog>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QMessageBox>
#include <QtGui/QVBoxLayout>

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
    model_->registerSetting("global/double_precision", "Use double precision", oskar_SettingsItem::BOOL);
    model_->registerSetting("global/max_sources_per_chunk", "Max. number of sources per chunk", oskar_SettingsItem::INT);
    model_->registerSetting("global/cuda_device_ids", "CUDA device IDs to use", oskar_SettingsItem::INT_CSV_LIST);
    model_->setCaption("global", "Global settings");
    model_->setCaption("sky", "Sky model settings");
    model_->registerSetting("sky/oskar_source_file", "Input OSKAR source file", oskar_SettingsItem::INPUT_FILE_NAME);
    model_->registerSetting("sky/gsm_file", "Input Global Sky Model file", oskar_SettingsItem::INPUT_FILE_NAME);
    model_->registerSetting("sky/output_sky_file", "Output OSKAR source file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    model_->registerSetting("sky/generator", "Generator", oskar_SettingsItem::STRING);
    model_->setCaption("sky/generator/healpix", "HEALPix (all sky) grid");
    model_->registerSetting("sky/generator/healpix/nside", "Nside", oskar_SettingsItem::INT);
    model_->setCaption("sky/generator/random", "Random (broken) power-law in flux");
    model_->registerSetting("sky/generator/random/num_sources", "Number of sources", oskar_SettingsItem::INT);
    model_->registerSetting("sky/generator/random/flux_density_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/generator/random/flux_density_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/generator/random/threshold", "Threshold (Jy)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/generator/random/power", "Power law index", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/generator/random/power1", "Power law index 1", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/generator/random/power2", "Power law index 2", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/generator/random/seed", "Random seed", oskar_SettingsItem::RANDOM_SEED);
    model_->setCaption("sky/filter", "Source filter settings");
    model_->registerSetting("sky/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/filter/flux_min", "Flux min (Jy)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("sky/filter/flux_max", "Flux max (Jy)", oskar_SettingsItem::DOUBLE);
    model_->setCaption("telescope", "Telescope model settings");
    model_->registerSetting("telescope/layout_file", "Array layout file", oskar_SettingsItem::INPUT_FILE_NAME);
    model_->registerSetting("telescope/latitude_deg", "Latitude (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("telescope/longitude_deg", "Longitude (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("telescope/altitude_m", "Altitude (m)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("station/station_directory", "Station directory", oskar_SettingsItem::INPUT_DIR_NAME);
    model_->registerSetting("station/disable_station_beam", "Disable station beam", oskar_SettingsItem::BOOL);
    model_->setCaption("station", "Station model settings");
    model_->setCaption("observation", "Observation settings");
    model_->registerSetting("observation/start_frequency", "Start frequency (Hz)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/num_channels", "Number of channels", oskar_SettingsItem::INT);
    model_->registerSetting("observation/frequency_inc", "Frequency increment (Hz)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/channel_bandwidth", "Channel bandwidth (Hz)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/phase_centre_ra_deg", "Phase centre RA (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/phase_centre_dec_deg", "Phase centre Dec (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/num_vis_dumps", "Number of visibility dumps", oskar_SettingsItem::INT);
    model_->registerSetting("observation/num_vis_ave", "Number of visibility averages", oskar_SettingsItem::INT);
    model_->registerSetting("observation/num_fringe_ave", "Number of fringe averages", oskar_SettingsItem::INT);
    model_->registerSetting("observation/start_time_utc", "Start time (UTC)", oskar_SettingsItem::DATE_TIME);
    model_->registerSetting("observation/length", "Observation length (h:m:s.z)", oskar_SettingsItem::TIME);
    model_->registerSetting("observation/oskar_vis_filename", "Output OSKAR visibility file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    model_->registerSetting("observation/ms_filename", "Output Measurement Set name", oskar_SettingsItem::OUTPUT_FILE_NAME);

    // Create and set up the settings view.
    view_ = new oskar_SettingsView(widget_);
    oskar_SettingsDelegate* delegate = new oskar_SettingsDelegate(view_, widget_);
    view_->setModel(model_);
    view_->setItemDelegate(delegate);
    view_->resizeColumnToContents(0);
    layout_->addWidget(view_);

    // Create the button box.
    buttons_ = new QDialogButtonBox(QDialogButtonBox::Close,
            Qt::Horizontal, widget_);
    buttons_->addButton("Run", QDialogButtonBox::AcceptRole);
    layout_->addWidget(buttons_);
    connect(buttons_, SIGNAL(accepted()), this, SLOT(runButton()));
    connect(buttons_, SIGNAL(rejected()), this, SLOT(close()));

    // Create the menus.
    menubar_ = new QMenuBar(this);
    menubar_->setGeometry(QRect(0, 0, 576, 25));
    setMenuBar(menubar_);
    menuFile_ = new QMenu("File", menubar_);
    menubar_->addAction(menuFile_->menuAction());
    actionOpen_ = new QAction("Open...", this);
    menuFile_->addAction(actionOpen_);
    connect(actionOpen_, SIGNAL(triggered()), this, SLOT(openSettings()));

    // Load the settings.
    QSettings settings;
    restoreGeometry(settings.value("main_window/geometry").toByteArray());
    restoreState(settings.value("main_window/state").toByteArray());
}

void oskar_MainWindow::openSettings(QString filename)
{
    // Check if the supplied filename is empty, and prompt to open file if so.
    if (filename.isEmpty())
        filename = QFileDialog::getOpenFileName(this, "Open Settings",
                settingsFile_);

    // Set the file if one was selected.
    if (!filename.isEmpty())
    {
        settingsFile_ = filename;
        model_->setFile(filename);
        setWindowTitle("OSKAR GUI [" + filename + "]");
    }
}

// Protected methods.

void oskar_MainWindow::closeEvent(QCloseEvent* event)
{
    QSettings settings;
    settings.setValue("main_window/geometry", saveGeometry());
    settings.setValue("main_window/state", saveState());
    QMainWindow::closeEvent(event);
}

// Private slots.

void oskar_MainWindow::runButton()
{
    if (settingsFile_.isEmpty())
    {
        QMessageBox::critical(this, "Error",
                "Must specify an OSKAR settings file.");
        return;
    }

    // Get the (list of) output file names.
    QStringList outputFiles;
    const QList<QString>& keys =  model_->outputKeys();
    for (int i = 0; i < keys.size(); ++i)
    {
        oskar_SettingsItem* it = model_->getItem(keys[i]);
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

// Private members.

void oskar_MainWindow::runSim(int depth, QStringList outputFiles)
{
    QByteArray settings = settingsFile_.toAscii();
    if (model_->iterationKeys().size() == 0)
    {
        int error = oskar_sim(settings);
        if (error)
        {
            fprintf(stderr, ">>> Run failed (code %d): %s.\n", error,
                    oskar_get_error_string(error));
        }
    }
    else
    {
        QString key = model_->iterationKeys()[depth];
        oskar_SettingsItem* item = model_->getItem(key);
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
                    model_->setValue(model_->outputKeys()[i], outputFiles[i]);
                }
            }

            // Check if recursion depth has been reached.
            if (depth < model_->iterationKeys().size() - 1)
            {
                // If not, then call this function again.
                runSim(depth + 1, outputFiles);
            }
            else
            {
                // Run the simulation with these settings.
                int error = oskar_sim(settings);
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
