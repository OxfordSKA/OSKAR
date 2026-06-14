/*
 * Copyright (c) 2012-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
#if QT_VERSION >= 0x050600 && QT_VERSION < 0x060000
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
