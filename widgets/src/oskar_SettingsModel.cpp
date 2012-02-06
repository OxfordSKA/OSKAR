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

#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsItem.h"
#include <QtGui/QApplication>
#include <QtGui/QFontMetrics>
#include <QtCore/QVector>
#include <QtCore/QSize>
#include <QtCore/QVariant>
#include <cstdio>

oskar_SettingsModel::oskar_SettingsModel(QObject* parent)
: QAbstractItemModel(parent)
{
    settings_ = NULL;
    rootItem_ = new oskar_SettingsItem(QString(), QString(),
            oskar_SettingsItem::CAPTION_ONLY, "Setting", QVariant());
    rootItem_->setValue("Value");
}

oskar_SettingsModel::~oskar_SettingsModel()
{
    // Delete any existing settings object.
    if (settings_)
    {
        settings_->sync();
        delete settings_;
    }
    delete rootItem_;
}

void oskar_SettingsModel::append(const QString& key,
        const QString& subkey, int type, const QString& caption,
        const QVariant& defaultValue, const QModelIndex& parent)
{
    oskar_SettingsItem *parentItem = getItem(parent);

    beginInsertRows(parent, rowCount(), rowCount());
    oskar_SettingsItem* item = new oskar_SettingsItem(key, subkey, type,
            caption, defaultValue, parentItem);
    parentItem->appendChild(item);
    endInsertRows();
    hash_.insert(key, item);
}

void oskar_SettingsModel::clearIteration(const QString& key)
{
    int i = iterationKeys_.indexOf(key);
    if (i >= 0)
    {
        iterationKeys_.removeAt(i);
        foreach (QString k, iterationKeys_)
        {
            QModelIndex idx = getIndex(k);
            emit dataChanged(index(idx.row(), 0, parent(idx)),
                    index(idx.row(), columnCount(), parent(idx)));
        }
        QModelIndex idx = getIndex(key);
        emit dataChanged(index(idx.row(), 0, parent(idx)),
                index(idx.row(), columnCount(), parent(idx)));
    }
}

int oskar_SettingsModel::columnCount(const QModelIndex& /*parent*/) const
{
    return 2;
}

QVariant oskar_SettingsModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    oskar_SettingsItem* item = getItem(index);

    if (role == Qt::FontRole)
    {
        int iterIndex = iterationKeys_.indexOf(item->key());
        if (iterIndex >= 0)
        {
            QFont font = QApplication::font();
            font.setBold(true);
            return font;
        }
    }

    if (index.column() == 0)
    {
        if (role == Qt::DisplayRole)
        {
            QString caption = item->caption();
            int iterIndex = iterationKeys_.indexOf(item->key());
            if (iterIndex >= 0)
                caption.prepend(QString("[%1] ").arg(iterIndex + 1));
            return caption;
        }
    }
    else if (index.column() == 1)
    {
        if (role == Qt::DisplayRole || role == Qt::EditRole)
        {
            return item->value();
        }
        else if (role == Qt::CheckStateRole &&
                item->type() == oskar_SettingsItem::BOOL)
        {
            return item->value().toBool() ? Qt::Checked : Qt::Unchecked;
        }
        else if (role == Qt::SizeHintRole)
        {
            int width = QApplication::fontMetrics().width(item->caption()) + 10;
            return QSize(width, 24);
        }
    }

    return QVariant();
}

Qt::ItemFlags oskar_SettingsModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    oskar_SettingsItem* item = getItem(index);

    if (index.column() == 0 ||
            item->type() == oskar_SettingsItem::CAPTION_ONLY)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    }
    else if (index.column() == 1 &&
            item->type() == oskar_SettingsItem::BOOL)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable |
                Qt::ItemIsUserCheckable;
    }

    return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

oskar_SettingsItem* oskar_SettingsModel::getItem(const QModelIndex& index) const
{
    if (index.isValid())
    {
        oskar_SettingsItem* item =
                static_cast<oskar_SettingsItem*>(index.internalPointer());
        if (item) return item;
    }
    return rootItem_;
}

oskar_SettingsItem* oskar_SettingsModel::getItem(const QString& key) const
{
    return hash_.value(key);
}

QVariant oskar_SettingsModel::headerData(int section,
        Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        if (section == 0)
            return rootItem_->caption();
        else if (section == 1)
            return rootItem_->value();
    }

    return QVariant();
}

QModelIndex oskar_SettingsModel::index(int row, int column,
        const QModelIndex& parent) const
{
    if (parent.isValid() && parent.column() != 0)
        return QModelIndex();

    oskar_SettingsItem* parentItem = getItem(parent);
    oskar_SettingsItem* childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

int oskar_SettingsModel::itemType(const QModelIndex& index) const
{
    return getItem(index)->type();
}

const QList<QString>& oskar_SettingsModel::iterationKeys() const
{
    return iterationKeys_;
}

QModelIndex oskar_SettingsModel::parent(const QModelIndex& index) const
{
    if (!index.isValid())
        return QModelIndex();

    oskar_SettingsItem* childItem = getItem(index);
    oskar_SettingsItem* parentItem = childItem->parent();

    if (parentItem == rootItem_)
        return QModelIndex();

    return createIndex(parentItem->childNumber(), 0, parentItem);
}

void oskar_SettingsModel::registerSetting(const QString& key,
        const QString& caption, int type, const QVariant& defaultValue,
        const QStringList& /*options*/)
{
    QStringList keys = key.split('/');

    // Find the parent, creating groups as necessary.
    QModelIndex parent, child;
    for (int k = 0; k < keys.size() - 1; ++k)
    {
        child = getChild(keys[k], parent);
        if (child.isValid())
            parent = child;
        else
        {
            // Append the group and set it as the new parent.
            append(key, keys[k], oskar_SettingsItem::CAPTION_ONLY, keys[k],
                    QVariant(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Append the actual setting.
    append(key, keys.last(), type, caption, defaultValue, parent);
}

int oskar_SettingsModel::rowCount(const QModelIndex& parent) const
{
    return getItem(parent)->childCount();
}

void oskar_SettingsModel::setCaption(const QString& key, const QString& caption)
{
    QModelIndex idx = getIndex(key);
    setData(idx, caption);
}

bool oskar_SettingsModel::setData(const QModelIndex& index,
        const QVariant& value, int role)
{
    if (!index.isValid())
        return false;

    oskar_SettingsItem* item = getItem(index);

    QVariant data;
    if (role == Qt::EditRole)
        data = value;
    else if (role == Qt::CheckStateRole)
        data = value.toBool() ? QString("true") : QString("false");

    if (index.column() == 0)
    {
        item->setCaption(data.toString());
        emit dataChanged(index, index);
        return true;
    }
    else if (index.column() == 1)
    {
        item->setValue(data);
        emit dataChanged(index, index);
        if (settings_)
        {
            if (value.toString().isEmpty())
                settings_->remove(item->key());
            else
                settings_->setValue(item->key(), data);
        }
        return true;
    }

    return false;
}

void oskar_SettingsModel::setFile(const QString& filename)
{
    if (!filename.isEmpty())
    {
        // Delete any existing settings object.
        if (settings_)
        {
            settings_->sync();
            delete settings_;
        }

        // Create new settings object from supplied filename.
        settings_ = new QSettings(filename, QSettings::IniFormat);

        // Display the contents of the file.
        beginResetModel();
        QModelIndex parent;
        loadFromParentIndex(parent);
        endResetModel();
    }
}

void oskar_SettingsModel::setIteration(const QString& key)
{
    if (!iterationKeys_.contains(key))
    {
        iterationKeys_.append(key);
        QModelIndex idx = getIndex(key);
        emit dataChanged(index(idx.row(), 0, parent(idx)),
                index(idx.row(), columnCount(), parent(idx)));
    }
}

// Private methods.

QModelIndex oskar_SettingsModel::getChild(const QString& subkey,
        const QModelIndex& parent) const
{
    // Search this parent's children.
    oskar_SettingsItem* item = getItem(parent);
    for (int i = 0; i < item->childCount(); ++i)
    {
        if (item->child(i)->subkey() == subkey)
            return index(i, 0, parent);
    }
    return QModelIndex();
}

QModelIndex oskar_SettingsModel::getIndex(const QString& key)
{
    QStringList keys = key.split('/');

    // Find the parent, creating groups as necessary.
    QModelIndex parent, child;
    for (int k = 0; k < keys.size() - 1; ++k)
    {
        child = getChild(keys[k], parent);
        if (child.isValid())
            parent = child;
        else
        {
            // Append the group and set it as the new parent.
            append(key, keys[k], oskar_SettingsItem::CAPTION_ONLY, keys[k],
                    QVariant(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Return the model index.
    child = getChild(keys.last(), parent);
    if (!child.isValid())
    {
        append(key, keys.last(), oskar_SettingsItem::CAPTION_ONLY, keys.last(),
                QVariant(), parent);
        child = index(rowCount(parent) - 1, 0, parent);
    }
    return child;
}

void oskar_SettingsModel::loadFromParentIndex(const QModelIndex& parent)
{
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            oskar_SettingsItem* item = getItem(idx);
            if (item->type() != oskar_SettingsItem::CAPTION_ONLY)
            {
                QVariant value = settings_->value(item->key(),
                        item->defaultValue());
                item->setValue(value);
                emit dataChanged(idx, idx);
            }
            loadFromParentIndex(idx);
        }
    }
}
