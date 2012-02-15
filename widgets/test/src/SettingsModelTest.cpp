#include <QtGui/QApplication>
#include <QtCore/QString>
#include <cstdio>
#include <cstdlib>

#include "widgets/oskar_SettingsDelegate.h"
#include "widgets/oskar_SettingsItem.h"
#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsView.h"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Specify OSKAR settings file on command line.\n");
        return EXIT_FAILURE;
    }

    QApplication app(argc, argv);

    oskar_SettingsModel mod;
    oskar_SettingsView view;
    oskar_SettingsDelegate delegate(&view);
    view.setModel(&mod);
    view.setItemDelegate(&delegate);
    view.setWindowTitle("OSKAR Settings");
    view.show();
    view.resizeColumnToContents(0);
    mod.setSettingsFile(QString(argv[1]));

    int status = app.exec();
    return status;
}
