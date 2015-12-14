/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_version.h>
#include <QtCore/QAbstractItemModel>
#include <QtCore/QDateTime>
#include <QtCore/QHash>
#include <QtGui/QSortFilterProxyModel>

class QModelIndex;
class QStringList;
class QSettings;
class QVariant;
class oskar_SettingsItem;

class oskar_SettingsModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    enum OSKAR_SETTINGS_MODEL_ROLE
    {
        KeyRole = Qt::UserRole, // 32
        ValueRole,
        TypeRole,
        RequiredRole,
        VisibleRole,
        LoadRole,
        OptionsRole,
        DefaultRole,
        DependencyKeyRole,
        DependencyValueRole,
        DependentKeyRole,
        DisabledRole,
        CheckExternalChangesRole
    };

public:
    oskar_SettingsModel(QObject* parent = 0);
    virtual ~oskar_SettingsModel();

    int columnCount(const QModelIndex& parent = QModelIndex()) const;
    QVariant data(const QModelIndex& index, int role) const;
    void declare(const QString& key, const QString& label, int type,
            const QVariant& defaultValue = QVariant(), bool required = false);
    void declare(const QString& key, const QString& label,
            const QStringList& options, int defaultIndex = 0,
            bool required = false);
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
    int rowCount(const QModelIndex& parent = QModelIndex()) const;
    void saveSettingsFile(const QString& filename);
    bool setData(const QModelIndex& index, const QVariant& value,
            int role = Qt::EditRole);
    void setDefault(const QString& key, const QVariant& value);
    void setDependency(const QString& key, const QString& dependency_key,
            const QVariant& dependency_value);
    void setLabel(const QString& key, const QString& label);
    void setTooltip(const QString& key, const QString& tooltip);
    void setValue(const QString& key, const QVariant& value);
    //QHash<QString, QVariant> settings() const;
    QString version() const;
    void setVersion(const QString& value = QString(OSKAR_VERSION_STR));

signals:
    void fileReloaded();

private:
    void append(const QString& key, const QString& subkey, int type,
            const QString& label, bool required, const QVariant& defaultValue,
            const QStringList& options, const QModelIndex& parent);
    QModelIndex getChild(const QString& subkey,
            const QModelIndex& parent = QModelIndex()) const;
    oskar_SettingsItem* getItem(const QModelIndex& index) const;
    void loadFromParentIndex(const QModelIndex& parent);
    int numModified(const QModelIndex& parent) const;
    void saveFromParentIndex(const QModelIndex& parent);
    void writeVersion();

    QSettings* settings_;
    oskar_SettingsItem* rootItem_;
    QHash<QString, oskar_SettingsItem*> itemHash_;
    QString version_;
    QString filename_;
    QDateTime lastModified_;
};

class oskar_SettingsModelFilter : public QSortFilterProxyModel
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
