/*
 * Copyright (c) 2015-2020, The University of Oxford
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

#include <QAbstractItemModel>
#include <QDateTime>
#include <QString>
#include <QSortFilterProxyModel>
#include <QIcon>

class QModelIndex;
class QStringList;
class QVariant;

namespace oskar {

class SettingsTree;
class SettingsNode;

class SettingsModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    enum ModelRole
    {
        KeyRole = Qt::UserRole,
        ValueRole,
        DefaultRole,
        TypeRole,
        OptionsRole,
        CheckExternalChangesRole,
        DisplayKeysRole,
        ItemTypeRole,
        RangeRole,
        ExtRangeRole,
        ResetGroupRole
    };

public:
    SettingsModel(SettingsTree* settings, QObject* parent = 0);
    virtual ~SettingsModel();
    void beginReset();
    void endReset();
    int columnCount(const QModelIndex& parent = QModelIndex()) const;
    QVariant data(const QModelIndex& index, int role) const;
    Qt::ItemFlags flags(const QModelIndex& index) const;
    QVariant headerData(int section, Qt::Orientation orientation,
            int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column,
                      const QModelIndex& parent = QModelIndex()) const;
    void load_settings_file(const QString& filename = QString());
    void save_settings_file(const QString& filename = QString());
    QModelIndex parent(const QModelIndex& index) const;
    void refresh();
    int rowCount(const QModelIndex& parent = QModelIndex()) const;
    bool setData(const QModelIndex& index, const QVariant& value,
            int role = Qt::EditRole);

signals:
    void fileReloaded();

private:
    const SettingsNode* get_node(const QModelIndex& index) const;
    void refresh(const QModelIndex& parent);
    void reset_group_(const SettingsNode* node);

    SettingsTree* settings_;
    QIcon iconOpen_, iconSave_;
    QString filename_;
    QDateTime lastModified_;
    bool displayKey_;
};


class SettingsModelFilter : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    SettingsModelFilter(QObject* parent = 0);
    ~SettingsModelFilter();
    QVariant data(const QModelIndex& index, int role) const;

public slots:
    void setFilterRegExp(const QString& pattern);

protected:
    bool filterAcceptsChildren(int sourceRow,
            const QModelIndex& sourceParent) const;
    bool filterAcceptsCurrentRow(const QModelIndex& idx) const;
    bool filterAcceptsCurrentRow(int sourceRow,
            const QModelIndex& sourceParent) const;
    virtual bool filterAcceptsRow(int sourceRow,
            const QModelIndex& sourceParent) const;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_MODEL_H_ */
