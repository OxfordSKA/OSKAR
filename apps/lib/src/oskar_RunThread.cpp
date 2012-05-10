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

#include "apps/lib/oskar_RunThread.h"
#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsItem.h"
#include "utility/oskar_get_error_string.h"

#include <QtCore/QVariant>
#include <cstdio>

oskar_RunThread::oskar_RunThread(oskar_SettingsModel* model,
        QObject* parent)
: QThread(parent)
{
    model_ = model;
}

void oskar_RunThread::go(int (*run_function)(const char*),
        QString settings_file, QStringList outputFiles)
{
    run_function_ = run_function;
    settingsFile_ = settings_file;
    outputFiles_ = outputFiles;
    start();
}

void oskar_RunThread::run()
{
    QByteArray settings = settingsFile_.toAscii();
    run(0, outputFiles_);
}

int oskar_RunThread::status() const
{
    return error_;
}

void oskar_RunThread::run(int depth, QStringList outputFiles)
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
