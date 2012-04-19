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

#include "oskar_global.h"
#include "utility/oskar_cuda_info_create.h"
#include "utility/oskar_cuda_info_free.h"
#include "utility/oskar_CudaInfo.h"
#include "utility/oskar_get_error_string.h"
#include "widgets/oskar_CudaInfoDisplay.h"

#include <cuda_runtime_api.h>
#include <QtGui/QApplication>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QMessageBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QTextBrowser>
#include <QtGui/QVBoxLayout>

oskar_CudaInfoDisplay::oskar_CudaInfoDisplay(QWidget *parent) : QDialog(parent)
{
    // Create the CUDA info structure.
    oskar_CudaInfo* info = NULL;
    int err = oskar_cuda_info_create(&info);

    // Set up the GUI.
    setWindowTitle("CUDA System Info");
    QVBoxLayout* vLayoutMain = new QVBoxLayout(this);

    // Create system info group.
    QGroupBox* grpAtt = new QGroupBox("CUDA System Info", this);
    grpAtt->setMinimumSize(600, 300);
    QVBoxLayout* vLayoutAtt = new QVBoxLayout(grpAtt);

    // Create system info document.
    QString html;
    html.append("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" "
            "\"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
            "<html><head></head><body>\n");
    if (!err)
    {
        html.append("<p>");
        html.append(QString("CUDA driver version: %1.%2<br />").
                arg(info->driver_version / 1000).
                arg((info->driver_version % 100) / 10));
        html.append(QString("CUDA runtime version: %1.%2<br />").
                arg(info->runtime_version / 1000).
                arg((info->runtime_version % 100) / 10));
        html.append(QString("Number of CUDA devices detected: %1<br />").
                arg(info->num_devices));
        html.append("</p>");
        for (int i = 0; i < info->num_devices; ++i)
        {
            html.append(QString("<p>Device [%1] name: %2</p>").
                    arg(i).arg(info->device[i].name));
            html.append("<ul>");
            html.append(QString("<li>Compute capability: %1.%2</li>").
                    arg(info->device[i].compute.capability.major).
                    arg(info->device[i].compute.capability.minor));
            html.append(QString("<li>Supports double precision: %1</li>").
                    arg(info->device[i].supports_double ? "true" : "false"));
            html.append(QString("<li>Global memory (MiB): %1</li>").
                    arg(info->device[i].global_memory_size / 1024.0));
            html.append(QString("<li>Free global memory (MiB): %1</li>").
                    arg(info->device[i].free_memory / 1024.0));
            html.append(QString("<li>Number of multiprocessors: %1</li>").
                    arg(info->device[i].num_multiprocessors));
            html.append(QString("<li>Number of CUDA cores: %1</li>").
                    arg(info->device[i].num_cores));
            html.append(QString("<li>GPU clock speed (MHz): %1</li>").
                    arg(info->device[i].gpu_clock / 1000.0));
            html.append(QString("<li>Memory clock speed (MHz): %1</li>").
                    arg(info->device[i].memory_clock / 1000.0));
            html.append(QString("<li>Memory bus width: %1-bit</li>").
                    arg(info->device[i].memory_bus_width));
            html.append(QString("<li>Level-2 cache size (kiB): %1</li>").
                    arg(info->device[i].level_2_cache_size / 1024));
            html.append(QString("<li>Shared memory size (kiB): %1</li>").
                    arg(info->device[i].shared_memory_size / 1024));
            html.append(QString("<li>Registers per block: %1</li>").
                    arg(info->device[i].num_registers));
            html.append(QString("<li>Warp size: %1</li>").
                    arg(info->device[i].warp_size));
            html.append(QString("<li>Max threads per block: %1</li>").
                    arg(info->device[i].max_threads_per_block));
            html.append(QString("<li>Max size of each dimension of a block: "
                    "(%1 x %2 x %3)</li>").
                    arg(info->device[i].max_threads_dim[0]).
                    arg(info->device[i].max_threads_dim[1]).
                    arg(info->device[i].max_threads_dim[2]));
            html.append(QString("<li>Max size of each dimension of a grid: "
                    "(%1 x %2 x %3)</li>").
                    arg(info->device[i].max_grid_size[0]).
                    arg(info->device[i].max_grid_size[1]).
                    arg(info->device[i].max_grid_size[2]));
            html.append("</ul>");
        }
    }
    else
    {
        html.append("<p>");
        html.append(QString("Could not obtain CUDA system info: %1").
                arg(oskar_get_error_string(err)));
        html.append("</p>");
    }
    html.append("</body></html>");

    // Create system info document display.
    QTextBrowser* display = new QTextBrowser(this);
    display->setHtml(html);
    display->setReadOnly(true);
    vLayoutAtt->addWidget(display);

    // Add system info group.
    vLayoutMain->addWidget(grpAtt);

    // Create close button.
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok,
            Qt::Horizontal, this);
    connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    vLayoutMain->addWidget(buttons);

    // Release the CUDA context(s).
    if (!err)
    {
        for (int i = 0; i < info->num_devices; ++i)
        {
            cudaSetDevice(i);
            cudaDeviceReset();
        }
        cudaSetDevice(0);
    }

    // Free the CUDA info structure.
    oskar_cuda_info_free(&info);
}
