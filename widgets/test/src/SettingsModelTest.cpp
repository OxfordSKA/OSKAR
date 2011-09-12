#include <QtGui/QApplication>
#include <QtGui/QTreeView>
#include <QtCore/QString>
#include <cstdio>
#include <cstdlib>

#include "widgets/oskar_SettingsModel.h"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);


    QString a = "hello";
    oskar_SettingsModel settings_model(a, 0);

    QTreeView tree_view;
    tree_view.setModel(&settings_model);
    tree_view.setWindowTitle("test tree view");
    tree_view.show();

    int status = app.exec();
    return status;
}



