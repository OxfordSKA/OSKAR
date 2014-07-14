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
#include <apps/lib/oskar_OptionParser.h>

#include <QtGui/QApplication>
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv)
{
    oskar_OptionParser opt("oskar");
    opt.setDescription("GUI application providing an interface for editing "
            "OSKAR settings files and launching OSKAR application binaries.");
    opt.addOptional("settings file");
    opt.addExample("oskar");
    opt.addExample("oskar settings.ini");
    if (!opt.check_options(argc, argv))
        return OSKAR_FAIL;

    // Create the QApplication and initialise settings fields.
    QApplication app(argc, argv);
    app.setApplicationName("OSKAR2");
    app.setOrganizationName("OeRC");

    // Create the main window.
    oskar_MainWindow mainWindow;

    // Show the main window.
    mainWindow.show();

    // Load settings file if one is provided on the command line.
    if (argc > 1)
        mainWindow.openSettings(QString(argv[1]));

    // Enter the event loop.
    return app.exec();
}
