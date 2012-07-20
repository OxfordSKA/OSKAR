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

#ifndef OSKAR_CONFIG_FILE_MODEL_H_
#define OSKAR_CONFIG_FILE_MODEL_H_

/**
 * @file oskar_ConfigFileModel.h
 */

#include "oskar_global.h"
#include <QtCore/QAbstractTableModel>
#include <QtCore/QObject>
#include <QtCore/QVariant>
#include <QtCore/QModelIndex>
#include <QtCore/QStringList>
#include <QtCore/QList>

#include "sky/oskar_SkyModel.h"

class OSKAR_EXPORT oskar_ConfigFileModel : public QAbstractTableModel
{
    public:
        Q_OBJECT

    public:
        oskar_ConfigFileModel(QObject* parent = 0);
        virtual ~oskar_ConfigFileModel();

    public:
        int rowCount(const QModelIndex& parent = QModelIndex()) const;
        int columnCount(const QModelIndex& parent = QModelIndex()) const;
        QVariant data(const QModelIndex& index, int role) const;
        QVariant headerData(int section, Qt::Orientation orientation,
                int role = Qt::DisplayRole) const;
        bool setData(const QModelIndex& index, const QVariant& value,
                int role = Qt::EditRole);
        Qt::ItemFlags flags(const QModelIndex& index) const;

    public:
        // Register a field in the configuration table.
        void registerFields(const QStringList& fields);

        // Specify the configuration file this model is associated with.
        void loadConfigFile(const QString& filename);

        // Save the model to a configuration file.
        void saveConfigFile();

    private:

    private:
        QString configFile_;
        QStringList field_name_;
        QList<int> field_type_;
        typedef QList<QVariant> config_row_;
        QList<config_row_> data_;
};

#endif // OSKAR_CONFIG_FILE_MODEL_H_
