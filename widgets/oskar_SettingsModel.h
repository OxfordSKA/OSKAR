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

#ifndef OSKAR_SETTINGS_MODEL_H_
#define OSKAR_SETTINGS_MODEL_H_

/**
 * @file oskar_SettingsModel.h
 */

#include "oskar_global.h"
#include <QtCore/QAbstractItemModel>
#include <QtCore/QHash>
#include <QtCore/QList>
#include <QtCore/QModelIndex>
#include <QtCore/QSettings>
#include <QtCore/QStringList>
#include <QtCore/QVariant>
#include <QtGui/QSortFilterProxyModel>

class oskar_SettingsItem;

class OSKAR_WIDGETS_EXPORT oskar_SettingsModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    enum {
        KeyRole = Qt::UserRole,
        ValueRole,
        TypeRole,
        RequiredRole,
        VisibleRole,
        EnabledRole,
        IterationNumRole,
        IterationIncRole,
        IterationKeysRole,
        SetIterationRole,
        ClearIterationRole,
        OutputKeysRole,
        LoadRole,
        OptionsRole,
        DefaultRole,
        DependencyKeyRole,
        DependencyValueRole,
        DependentKeyRole,
        HiddenRole
    };

public:
    oskar_SettingsModel(QObject* parent = 0);
    virtual ~oskar_SettingsModel();

    int columnCount(const QModelIndex& parent = QModelIndex()) const;
    QVariant data(const QModelIndex& index, int role) const;
    Qt::ItemFlags flags(const QModelIndex& index) const;
    const oskar_SettingsItem* getItem(const QString& key) const;
    QVariant headerData(int section, Qt::Orientation orientation,
            int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column,
            const QModelIndex& parent = QModelIndex()) const;
    QModelIndex index(const QString& key);
    bool isModified() const;
    void loadSettingsFile(const QString& filename);
    QModelIndex parent(const QModelIndex& index) const;
    void registerSetting(const QString& key, const QString& label,
            int type, bool required = false,
            const QVariant& defaultValue = QVariant());
    void registerSetting(const QString& key, const QString& label,
            int type, const QStringList& options, bool required = false,
            const QVariant& defaultValue = QVariant());
    int rowCount(const QModelIndex& parent = QModelIndex()) const;
    void saveSettingsFile(const QString& filename);
    bool setData(const QModelIndex& index, const QVariant& value,
            int role = Qt::EditRole);
    void setDefault(const QString& key, const QVariant& value);
    void setDependencies(const QString& key, const QString& dependency_key,
            const QVariant& dependency_value);
    void setLabel(const QString& key, const QString& label);
    void setTooltip(const QString& key, const QString& tooltip);
    void setValue(const QString& key, const QVariant& value);
    QHash<QString, QVariant> settings() const;

private:
    void append(const QString& key, const QString& subkey, int type,
            const QString& label, bool required, const QVariant& defaultValue,
            const QStringList& options, const QModelIndex& parent);
    QModelIndex getChild(const QString& subkey,
            const QModelIndex& parent = QModelIndex()) const;
    oskar_SettingsItem* getItem(const QModelIndex& index) const;
    void loadFromParentIndex(const QModelIndex& parent);
    int numModified(const QModelIndex& parent) const;
    void restoreAll(const QModelIndex& parent = QModelIndex());
    void saveFromParentIndex(const QModelIndex& parent);

    QSettings* settings_;
    oskar_SettingsItem* rootItem_;
    QHash<QString, oskar_SettingsItem*> itemHash_;
    QStringList iterationKeys_;
    QStringList outputKeys_;
};

class OSKAR_WIDGETS_EXPORT oskar_SettingsModelFilter : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    oskar_SettingsModelFilter(QObject* parent = 0);
    virtual ~oskar_SettingsModelFilter();
    QVariant data(const QModelIndex& index, int role) const;
    bool hideUnsetItems() const;

public slots:
    void setFilterText(QString value);
    void setHideUnsetItems(bool value);

protected:
    bool filterAcceptsChildren(int sourceRow,
            const QModelIndex& sourceParent) const;
    bool filterAcceptsCurrentRow(const QModelIndex& idx) const;
    bool filterAcceptsCurrentRow(int sourceRow,
            const QModelIndex& sourceParent) const;
    virtual bool filterAcceptsRow(int sourceRow,
            const QModelIndex& sourceParent) const;

private:
    bool hideUnsetItems_;
    QString filterText_;
};

#endif /* OSKAR_SETTINGS_MODEL_H_ */
