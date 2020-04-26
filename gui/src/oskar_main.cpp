/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#include "settings/oskar_option_parser.h"
#include "gui/oskar_MainWindow.h"

#include <QApplication>

int main(int argc, char** argv)
{
    oskar::OptionParser opt("oskar", "");
    opt.set_description("GUI application providing an interface for editing "
            "OSKAR settings files and launching OSKAR application binaries.");
    opt.add_optional("settings file");
    opt.add_example("oskar");
    opt.add_example("oskar settings.ini");
    if (!opt.check_options(argc, argv))
        return EXIT_FAILURE;

    // Create the QApplication and initialise settings fields.
#if QT_VERSION >= 0x050600
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
    QApplication app(argc, argv);
    app.setApplicationName("OSKAR2");
    app.setOrganizationName("OeRC");

#ifdef Q_OS_MACOS
    // Add installed plugin search path.
    QString app_path = QCoreApplication::applicationDirPath();
    QString plugin_path = app_path + "/../../PlugIns/";
    QCoreApplication::addLibraryPath(plugin_path);
#endif

    // Create and show the main window.
    oskar::MainWindow main_window;
    main_window.show();

    // Load settings file if one is provided on the command line.
    if (argc > 1)
        main_window.open(QString(argv[1]));

    // Enter the event loop.
    return app.exec();
}
