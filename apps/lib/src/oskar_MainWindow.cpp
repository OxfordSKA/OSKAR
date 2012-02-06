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
    model_->registerSetting("observation/start_frequency", "Start frequency", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/num_channels", "Number of channels", oskar_SettingsItem::INT);
    model_->registerSetting("observation/frequency_inc", "Frequency increment (Hz)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/channel_bandwidth", "Channel bandwidth (Hz)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/phase_centre_ra_deg", "Phase centre RA (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/phase_centre_dec_deg", "Phase centre Dec (deg)", oskar_SettingsItem::DOUBLE);
    model_->registerSetting("observation/num_vis_dumps", "Number of visibility dumps", oskar_SettingsItem::INT);
    model_->registerSetting("observation/num_vis_ave", "Number of visibility averages", oskar_SettingsItem::INT);
    model_->registerSetting("observation/num_fringe_ave", "Number of fringe averages", oskar_SettingsItem::INT);
    model_->registerSetting("observation/start_time_utc", "Start time (UTC)", oskar_SettingsItem::DATE_TIME);
    model_->registerSetting("observation/length_seconds", "Observation length (s)", oskar_SettingsItem::DOUBLE);
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
    buttons_ = new QDialogButtonBox(
            (QDialogButtonBox::Ok | QDialogButtonBox::Cancel),
            Qt::Horizontal, widget_);
    layout_->addWidget(buttons_);
    connect(buttons_, SIGNAL(accepted()), this, SLOT(runButton()));
    connect(buttons_, SIGNAL(rejected()), qApp, SLOT(quit()));

    // Create the menus.
    menubar_ = new QMenuBar(this);
    menubar_->setGeometry(QRect(0, 0, 576, 25));
    setMenuBar(menubar_);
    menuFile_ = new QMenu("File", menubar_);
    menubar_->addAction(menuFile_->menuAction());
    actionOpen_ = new QAction("Open...", this);
    menuFile_->addAction(actionOpen_);
    connect(actionOpen_, SIGNAL(triggered()), this, SLOT(openSettings()));
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
    runSim(0);
}

void oskar_MainWindow::openSettings()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Settings");
    if (!filename.isEmpty())
    {
        settingsFile_ = filename;
        model_->setFile(filename);
        setWindowTitle("OSKAR GUI [" + filename + "]");
    }
}

// Private members.

void oskar_MainWindow::runSim(int depth)
{
    QByteArray settings = settingsFile_.toAscii();
    if (model_->iterationKeys().size() == 0)
    {
        oskar_sim(settings);
    }
    else
    {
        QString key = model_->iterationKeys()[depth];
        oskar_SettingsItem* item = model_->getItem(key);
        int iter = item->iterationNum();
        int startInt = item->value().toInt();
        double startDbl = item->value().toDouble();
        for (int i = 0; i < iter; ++i)
        {
            // Set the settings file here.
            printf("setting %s = ", key.toAscii().constData());
            QVariant var;
            if (item->type() == oskar_SettingsItem::INT)
            {
                var = startInt + i * item->iterationInc().toInt();
                printf("%d\n", var.toInt());
            }
            else if (item->type() == oskar_SettingsItem::DOUBLE)
            {
                var = startDbl + i * item->iterationInc().toDouble();
                printf("%.3f\n", var.toDouble());
            }

            // Run the simulation with these settings.
            if (depth < model_->iterationKeys().size() - 1)
                runSim(depth + 1);
            else
            {
                printf("Starting run %d with depth %d\n", i, depth);
//                oskar_sim(settings);
            }
        }
    }
}
