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

#include "widgets/oskar_DataFileModel.h"
#include <QtCore/QVector>

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <cstdio>
#include <cstdlib>


oskar_ConfigFileModel::oskar_ConfigFileModel(QObject* parent)
: QAbstractTableModel(parent)
{
}

oskar_ConfigFileModel::~oskar_ConfigFileModel()
{
    saveConfigFile();
}

void oskar_ConfigFileModel::registerFields(const QStringList& fields)
{
    field_name_ = fields;
}

void oskar_ConfigFileModel::loadConfigFile(const QString& filename)
{
    configFile_ = filename;

    FILE* file = fopen(filename.toLatin1().data(), "r");
    char* line = NULL;
    size_t bufsize = 0;
    int num_fields = field_name_.length();

    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        if (line[0] == '#') continue;

        QVector<double> par(num_fields, 0.0);
        oskar_string_to_array_d(line, num_fields, par.data());

        QList<QVariant> row_;
        for (int j = 0; j < num_fields; ++j)
        {
            if (j < 3 || par[j] != 0.0)
                row_.append(par[j]);
            else
                row_.append(QVariant());
        }
        data_.append(row_);
    }
    fclose(file);

    reset();
}


void oskar_ConfigFileModel::saveConfigFile()
{
    // Note: This method will currently overwrite the current
    // configuration file removing all existing comments. This is
    // not ideal...
    // -- maybe put the comments in the loaded table? (probably best solution)
    // -- Alternatively save just the first block of comments... (less ideal)


    QString filename = configFile_ + ".edit";
    FILE* file = fopen(filename.toLatin1().data(), "w");

    // Write header.
    fprintf(file, "#");
    fprintf(file, "\n");
    fprintf(file, "# ");
    int num_fields = field_name_.length();
    for (int i = 0; i < num_fields; ++i)
    {
        if (i > 0) fprintf(file, ", ");
        fprintf(file, "%s", field_name_[i].toLatin1().data());
    }
    fprintf(file, "\n");
    fprintf(file, "#");
    fprintf(file, "\n");

    // Write data.
    for (int row = 0; row < data_.length(); ++row)
    {
        for (int col = 0; col < data_[row].length(); ++col)
        {
            fprintf(file, "%f ", data_[row][col].toDouble());
        }
        fprintf(file, "\n");
    }

    fclose(file);
}


int oskar_ConfigFileModel::rowCount(const QModelIndex& /*parent*/) const
{
    return data_.length();
}

int oskar_ConfigFileModel::columnCount(const QModelIndex& /*parent*/) const
{
    if (data_.length() > 0)
        return data_[0].length();
    else
        return 0;
}


QVariant oskar_ConfigFileModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || role != Qt::DisplayRole)
        return QVariant();

    int row = index.row();
    int column = index.column();

    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        return data_[row][column];
    }

    return QVariant();
}

QVariant oskar_ConfigFileModel::headerData(int section, Qt::Orientation orientation,
        int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        if (section < field_name_.length())
            return QVariant(field_name_[section]);
        else
            return QVariant("Field " + QString::number(section + 1));
    }
    else if (orientation == Qt::Vertical && role == Qt::DisplayRole)
    {
        return QVariant(section);
    }

    return QVariant();
}


bool oskar_ConfigFileModel::setData(const QModelIndex& index, const QVariant& value,
        int role)
{
    printf("setData() role = %i\n", role);

    if (!index.isValid())
        return false;

    if (role == Qt::EditRole)
    {
        int row = index.row();
        int column = index.column();

        data_[row][column] = value;
        return true;
    }
    return false;
}


Qt::ItemFlags oskar_ConfigFileModel::flags(const QModelIndex& /*index*/) const
{
    return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}
