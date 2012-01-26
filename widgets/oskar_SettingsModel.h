/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_SETTINGS_MODEL_H_
#define OSKAR_SETTINGS_MODEL_H_

/**
 * @file oskar_SettingsModel.h
 */

#include <QtCore/QAbstractItemModel>
#include <QtCore/QModelIndex>
#include <QtCore/QVariant>
#include <QtCore/QStringList>
#include <QtGui/QTreeView>

class oskar_SettingsItem;

class oskar_SettingsModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    oskar_SettingsModel(QObject* parent);
    virtual ~oskar_SettingsModel();

    void append(const QString& key, const QString& keyShort,
            int type, const QVector<QVariant>& data,
            const QModelIndex& parent = QModelIndex());
    int columnCount(const QModelIndex& parent = QModelIndex()) const;
    QVariant data(const QModelIndex& index, int role) const;
    Qt::ItemFlags flags(const QModelIndex& index) const;
    QVariant headerData(int section, Qt::Orientation orientation,
            int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column,
            const QModelIndex& parent = QModelIndex()) const;
//    bool insertColumns(int position, int columns,
//            const QModelIndex& parent = QModelIndex());
    QModelIndex parent(const QModelIndex& index) const;
    void registerSetting(const QString& key, const QString& caption,
            int type, const QStringList& options = QStringList());
    int rowCount(const QModelIndex& parent = QModelIndex()) const;
    void setCaption(const QString& key,
            const QString& caption);
    bool setData(const QModelIndex& index, const QVariant& value,
            int role = Qt::EditRole);
//    bool setHeaderData(int section, Qt::Orientation orientation,
//            const QVariant& value, int role = Qt::EditRole);

private:
    oskar_SettingsItem* getItem(const QModelIndex& index) const;
    QModelIndex getChild(const QString& keyShort,
            const QModelIndex& parent = QModelIndex()) const;

    oskar_SettingsItem* rootItem_;
};

#endif /* OSKAR_SETTINGS_MODEL_H_ */
