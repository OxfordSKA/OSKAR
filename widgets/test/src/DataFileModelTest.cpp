#include <QtGui/QApplication>
#include <QtGui/QTableView>
#include <QtCore/QString>
#include <cstdio>
#include <cstdlib>

#include "widgets/oskar_DataFileModel.h"


int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    // Create a test config file.
    const char* filename = "temp_test_config_file.txt";
    FILE* file = fopen(filename, "w");
    fprintf(file, "# RA, Dec, I, Q, U, V, freq0, spix, maj, min, pa\n");
    fprintf(file, "#\n");
    int num_sources = 10;
    for (int i = 0; i < num_sources; ++i)
    {
        fprintf(file, "%f ", (double)i/10.0);  // RA
        fprintf(file, "%f ", (double)i/100.0); // Dec
        fprintf(file, "%f ", 1.0);             // I
        if (i == 2)
        {
            fprintf(file, "%f ", 2.0);             // Q
            fprintf(file, "%f ", 3.0);             // U
            fprintf(file, "%f ", 4.0);             // V
            fprintf(file, "%f ", 1.0e6);           // freq0
            fprintf(file, "%f ", -0.7);            // spix
            fprintf(file, "%f ", 100.0);           // maj
            fprintf(file, "%f ", 200.0);           // min
            fprintf(file, "%f ", 15.0*(double)i);  // pa
        }
        fprintf(file, "\n");
    }
    fclose(file);


    oskar_ConfigFileModel model;
    QTableView view;

    view.setModel(&model);
    view.show();


    QStringList fields;
    fields << "RA" << "Dec" << "I" << "Q" << "U" << "V"
            << "freq0" << "spix" << "Maj" << "Min" << "pa";
    model.registerFields(fields);
    model.loadConfigFile(QString(filename));
    view.resizeColumnsToContents();

    int status = app.exec();

    remove(filename);

    return status;
}
