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

#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsItem.h"
#include <QtGui/QApplication>
#include <QtGui/QFontMetrics>
#include <QtGui/QIcon>
#include <QtCore/QSize>
#include <QtCore/QVariant>
#include <cfloat>

oskar_SettingsModel::oskar_SettingsModel(QObject* parent)
: QAbstractItemModel(parent),
  settings_(NULL),
  rootItem_(NULL)
{
    // Set up the root item.
    rootItem_ = new oskar_SettingsItem(QString(), QString(),
            oskar_SettingsItem::LABEL, "Setting", "Value");
}

oskar_SettingsModel::~oskar_SettingsModel()
{
    // Delete any existing settings object.
    if (settings_)
        delete settings_;
    delete rootItem_;
}

int oskar_SettingsModel::columnCount(const QModelIndex& /*parent*/) const
{
    return 2;
}

QVariant oskar_SettingsModel::data(const QModelIndex& index, int role) const
{
    // Check for roles that do not depend on the index.
    if (role == IterationKeysRole)
    {
        // Check if keys are actually enabled before returning the list.
        QStringList keys;
        for (int i = 0; i < iterationKeys_.size(); ++i)
        {
            if (getItem(iterationKeys_[i])->enabled())
                keys.append(iterationKeys_[i]);
        }
        return keys;
    }
    else if (role == OutputKeysRole)
    {
        // Check if keys are actually enabled before returning the list.
        QStringList keys;
        for (int i = 0; i < outputKeys_.size(); ++i)
        {
            if (getItem(outputKeys_[i])->enabled())
                keys.append(outputKeys_[i]);
        }
        return keys;
    }

    // Get a pointer to the item.
    if (!index.isValid())
        return QVariant();
    oskar_SettingsItem* item = getItem(index);

    // Check for roles common to all columns.
    if (role == Qt::FontRole)
    {
        if (iterationKeys_.contains(item->key()))
        {
            QFont font = QApplication::font();
            font.setBold(true);
            return font;
        }
    }
    else if (role == Qt::ForegroundRole)
    {
        if (item->hidden())
            return QColor(Qt::gray);
        if (item->enabled())
        {
            if (item->critical())
                return QColor(Qt::white);
            if (item->valueSet())
                return QColor(Qt::blue);
            else
                return QColor(64, 64, 64);
        }
        else
            return QColor(Qt::red);
    }
    else if (role == Qt::BackgroundRole)
    {
        if (item->critical())
        {
            if (index.column() == 0)
                return QColor(0, 48, 255, 160);
            else if (item->type() != oskar_SettingsItem::LABEL)
                return QColor(255, 64, 64, 255);
        }
        if (index.column() == 1)
            return QColor(16, 16, 16, 16);
    }
    else if (role == Qt::ToolTipRole)
    {
        QString tooltip = item->tooltip();
        if (item->critical() && !tooltip.isEmpty())
            tooltip.append(" [Required]");
        return tooltip;
    }
    else if (role == DefaultRole)
        return item->defaultValue();
    else if (role == KeyRole)
        return item->key();
    else if (role == ValueRole)
        return item->value();
    else if (role == TypeRole)
        return item->type();
    else if (role == RequiredRole)
        return item->required();
    else if (role == HiddenRole)
        return item->hidden();
    else if (role == VisibleRole)
        return item->valueSet() || item->required();
    else if (role == EnabledRole)
        return item->enabled();
    else if (role == OptionsRole)
        return item->options();
    else if (role == IterationNumRole)
        return item->iterationNum();
    else if (role == IterationIncRole)
        return item->iterationInc();
    else if (role == Qt::DecorationRole)
    {
        if (index.column() == 0)
        {
            if (item->type() == oskar_SettingsItem::INPUT_FILE_NAME ||
                    item->type() == oskar_SettingsItem::INPUT_FILE_LIST ||
                    item->type() == oskar_SettingsItem::TELESCOPE_DIR_NAME)
            {
                return QIcon(":/icons/open.png");
            }
            else if (item->type() == oskar_SettingsItem::OUTPUT_FILE_NAME)
            {
                return QIcon(":/icons/save.png");
            }
        }
    }

    // Check for roles in specific columns.
    if (index.column() == 0)
    {
        if (role == Qt::DisplayRole)
        {
            QString label = item->label();
            int iterIndex = iterationKeys_.indexOf(item->key());
            if (iterIndex >= 0)
                label.prepend(QString("[%1] ").arg(iterIndex + 1));
            return label;
        }
    }
    else if (index.column() == 1)
    {
        if (role == Qt::DisplayRole)
        {
            QVariant val = item->valueOrDefault();
            if (item->type() == oskar_SettingsItem::INPUT_FILE_LIST ||
                    val.type() == QVariant::StringList)
            {
                QStringList list = val.toStringList();
                return list.join(",");
            }
            if (!item->enabled())
            {
                QString str = val.toString();
                QString defaultString = item->defaultValue().toString();
                if (!defaultString.isEmpty())
                {
                    str.append(QString(" [using %1]").arg(defaultString));
                    return str;
                }
            }
            return val;
        }
        else if (role == Qt::EditRole)
        {
            QVariant val = item->valueOrDefault();
            if (val.type() == QVariant::StringList)
            {
                QStringList list = val.toStringList();
                return list.join(",");
            }
            return val;
        }
        else if (role == Qt::CheckStateRole &&
                item->type() == oskar_SettingsItem::BOOL)
        {
            QVariant val = item->valueOrDefault();
            return val.toBool() ? Qt::Checked : Qt::Unchecked;
        }
        else if (role == Qt::SizeHintRole)
        {
            int width = QApplication::fontMetrics().width(item->label()) + 10;
            return QSize(width, 26);
        }
    }

    return QVariant();
}

void oskar_SettingsModel::declare(const QString& key, const QString& label,
        int type, const QVariant& defaultValue, bool required)
{
    // Find the parent, creating groups as necessary.
    QStringList keys = key.split('/');
    QModelIndex parent, child;
    for (int k = 0; k < keys.size() - 1; ++k)
    {
        child = getChild(keys[k], parent);
        if (child.isValid())
            parent = child;
        else
        {
            // Append the group and set it as the new parent.
            append(key, keys[k], oskar_SettingsItem::LABEL, keys[k],
                    required, QVariant(), QStringList(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Append the actual setting.
    append(key, keys.last(), type, label, required, defaultValue,
            QStringList(), parent);

    // Check if this is an output file.
    if (type == oskar_SettingsItem::OUTPUT_FILE_NAME)
        outputKeys_.append(key);
}

void oskar_SettingsModel::declare(const QString& key, const QString& label,
        const QStringList& options, int defaultIndex, bool required)
{
    // Get the default value.
    QVariant defaultValue;
    if (defaultIndex < options.size())
        defaultValue = options[defaultIndex];

    // Find the parent, creating groups as necessary.
    QStringList keys = key.split('/');
    QModelIndex parent, child;
    for (int k = 0; k < keys.size() - 1; ++k)
    {
        child = getChild(keys[k], parent);
        if (child.isValid())
            parent = child;
        else
        {
            // Append the group and set it as the new parent.
            append(key, keys[k], oskar_SettingsItem::LABEL, keys[k],
                    required, QVariant(), QStringList(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Append the actual setting.
    append(key, keys.last(), oskar_SettingsItem::OPTIONS, label, required,
            defaultValue, options, parent);
}

Qt::ItemFlags oskar_SettingsModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    oskar_SettingsItem* item = getItem(index);
    if (!item->enabled() || item->hidden())
        return Qt::ItemIsSelectable;

    if (index.column() == 0 || item->type() == oskar_SettingsItem::LABEL)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    }
    else if (index.column() == 1 && item->type() == oskar_SettingsItem::BOOL)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable |
                Qt::ItemIsUserCheckable;
    }

    return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

const oskar_SettingsItem* oskar_SettingsModel::getItem(const QString& key) const
{
    return itemHash_.value(key);
}

QVariant oskar_SettingsModel::headerData(int section,
        Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        if (section == 0)
            return rootItem_->label();
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

QModelIndex oskar_SettingsModel::index(const QString& key)
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
            append(key, keys[k], oskar_SettingsItem::LABEL, keys[k],
                    false, QVariant(), QStringList(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Return the model index.
    child = getChild(keys.last(), parent);
    if (!child.isValid())
    {
        append(key, keys.last(), oskar_SettingsItem::LABEL, keys.last(),
                false, QVariant(), QStringList(), parent);
        child = index(rowCount(parent) - 1, 0, parent);
    }
    return child;
}

bool oskar_SettingsModel::isModified() const
{
    return numModified(QModelIndex()) > 0;
}

void oskar_SettingsModel::loadSettingsFile(const QString& filename)
{
    if (!filename.isEmpty())
    {
        // Check if any settings are currently disabled, and enable them if so.
        restoreAll();

        // Delete any existing settings object.
        if (settings_)
            delete settings_;

        // Create new settings object from supplied filename.
        settings_ = new QSettings(filename, QSettings::IniFormat);

        // Display the contents of the file.
        beginResetModel();
        loadFromParentIndex(QModelIndex());
        endResetModel();
    }
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

int oskar_SettingsModel::rowCount(const QModelIndex& parent) const
{
    return getItem(parent)->childCount();
}

void oskar_SettingsModel::saveSettingsFile(const QString& filename)
{
    if (!filename.isEmpty())
    {
        // Check if any settings are currently disabled, and enable them if so.
        restoreAll();

        // Delete any existing settings object.
        if (settings_)
            delete settings_;

        // Create new settings object from supplied filename.
        settings_ = new QSettings(filename, QSettings::IniFormat);

        // Set the contents of the file.
        saveFromParentIndex(QModelIndex());
    }
}

bool oskar_SettingsModel::setData(const QModelIndex& idx,
        const QVariant& value, int role)
{
    if (!idx.isValid())
        return false;

    // Get a pointer to the item.
    oskar_SettingsItem* item = getItem(idx);

    // Get model indexes for the row.
    QModelIndex topLeft = idx.sibling(idx.row(), 0);
    QModelIndex bottomRight = idx.sibling(idx.row(), columnCount() - 1);

    // Check for role type.
    if (role == Qt::ToolTipRole)
    {
        item->setTooltip(value.toString());
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == DefaultRole)
    {
        item->setDefaultValue(value);
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == DependencyKeyRole)
    {
        item->setDependencyKey(value.toString());
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == DependencyValueRole)
    {
        item->setDependencyValue(value);
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == DependentKeyRole)
    {
        item->addDependent(value.toString());
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == HiddenRole)
    {
        item->setHidden(value.toBool());
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == EnabledRole)
    {
        item->setEnabled(value.toBool());
        if (settings_)
        {
            if (value.toBool())
            {
                if (item->value().isValid())
                    settings_->setValue(item->key(), item->value());
            }
            else
                settings_->remove(item->key());
            settings_->sync();
        }
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == IterationNumRole)
    {
        item->setIterationNum(value.toInt());
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == IterationIncRole)
    {
        item->setIterationInc(value);
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == SetIterationRole)
    {
        if (!iterationKeys_.contains(item->key()))
        {
            iterationKeys_.append(item->key());
            emit dataChanged(topLeft, bottomRight);
            return true;
        }
        return false;
    }
    else if (role == ClearIterationRole)
    {
        int i = iterationKeys_.indexOf(item->key());
        if (i >= 0)
        {
            iterationKeys_.removeAt(i);
            emit dataChanged(topLeft, bottomRight);
            foreach (QString k, iterationKeys_)
            {
                QModelIndex idx = index(k);
                emit dataChanged(idx, idx.sibling(idx.row(), columnCount()-1));
            }
            return true;
        }
        return false;
    }
    else if (role == Qt::EditRole || role == Qt::CheckStateRole ||
            role == LoadRole)
    {
        QVariant data = value;
        if (role == Qt::CheckStateRole)
            data = value.toBool() ? QString("true") : QString("false");

        if (idx.column() == 0)
        {
            item->setLabel(data.toString());
            emit dataChanged(idx, idx);
            return true;
        }
        else if (idx.column() == 1)
        {
            // Set the data in the settings file.
            if ((role != LoadRole) && settings_)
            {
                if (data.isNull())
                    settings_->remove(item->key());
                else
                    settings_->setValue(item->key(), data);
                settings_->sync();
            }

            // Set the item data.
            item->setValue(data);
            QModelIndex i(idx);
            while (i.isValid())
            {
                emit dataChanged(i.sibling(i.row(), 0),
                        i.sibling(i.row(), columnCount()-1));
                i = i.parent();
            }

            // Check for dependents.
            for (int i = 0; i < item->dependentKeys().size(); ++i)
            {
                QModelIndex idx_dependent = index(item->dependentKeys().at(i));
                oskar_SettingsItem* dependent = getItem(idx_dependent);
                if (dependent->dependencyValue() == item->valueOrDefault()
                        && dependent->hidden())
                {
                    setData(idx_dependent, false, HiddenRole);
                }
                else if (dependent->dependencyValue() != item->valueOrDefault()
                        && !dependent->hidden())
                {
                    setData(idx_dependent, true, HiddenRole);
                }
            }
            return true;
        }
    }

    return false;
}

void oskar_SettingsModel::setDefault(const QString& key, const QVariant& value)
{
    QModelIndex idx = index(key);
    setData(idx, value, DefaultRole);
}

void oskar_SettingsModel::setDependency(const QString& key,
        const QString& dependency_key, const QVariant& dependency_value)
{
    // Check that both keys have been registered, and return immediately if not.
    if (!itemHash_.contains(key) || !itemHash_.contains(dependency_key))
        return;

    // Set dependencies of this key.
    QModelIndex idx_dependent = index(key);
    setData(idx_dependent, dependency_key, DependencyKeyRole);
    setData(idx_dependent, dependency_value, DependencyValueRole);

    // Add this key to the list of dependents (of the dependency key!).
    QModelIndex idx_dependency = index(dependency_key);
    setData(idx_dependency, key, DependentKeyRole);

    // Do the initial check and set the flag to indicate whether hidden or not.
    oskar_SettingsItem* dependency = getItem(idx_dependency);
    if (dependency->valueOrDefault() == dependency_value)
        setData(idx_dependent, false, HiddenRole);
    else
        setData(idx_dependent, true, HiddenRole);
}

void oskar_SettingsModel::setLabel(const QString& key, const QString& label)
{
    QModelIndex idx = index(key);
    setData(idx, label);
}

void oskar_SettingsModel::setTooltip(const QString& key, const QString& tooltip)
{
    QModelIndex idx = index(key);
    setData(idx, tooltip, Qt::ToolTipRole);
}

void oskar_SettingsModel::setValue(const QString& key, const QVariant& value)
{
    // Get the model index.
    QModelIndex idx = index(key);
    idx = idx.sibling(idx.row(), 1);

    // Set the data.
    setData(idx, value, Qt::EditRole);
}

QHash<QString, QVariant> oskar_SettingsModel::settings() const
{
    QHash<QString, QVariant> hash;
    foreach (oskar_SettingsItem* item, itemHash_)
    {
        hash.insert(item->key(), item->valueOrDefault());
    }
    return hash;
}


// Private methods.

void oskar_SettingsModel::append(const QString& key, const QString& subkey,
        int type, const QString& label, bool required,
        const QVariant& defaultValue, const QStringList& options,
        const QModelIndex& parent)
{
    oskar_SettingsItem *parentItem = getItem(parent);

    beginInsertRows(parent, rowCount(), rowCount());
    oskar_SettingsItem* item = new oskar_SettingsItem(key, subkey, type,
            label, QVariant(), required, defaultValue, options, parentItem);
    parentItem->appendChild(item);
    endInsertRows();
    itemHash_.insert(key, item);
}

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

void oskar_SettingsModel::loadFromParentIndex(const QModelIndex& parent)
{
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            const oskar_SettingsItem* item = getItem(idx);
            setData(idx.sibling(idx.row(), 1),
                    settings_->value(item->key()), LoadRole);
            loadFromParentIndex(idx);
        }
    }
}

int oskar_SettingsModel::numModified(const QModelIndex& parent) const
{
    int num_modified = 0;
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            const oskar_SettingsItem* item = getItem(idx);
            if (!item->value().isNull())
                ++num_modified;
            num_modified += numModified(idx);
        }
    }
    return num_modified;
}

void oskar_SettingsModel::restoreAll(const QModelIndex& parent)
{
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            const oskar_SettingsItem* item = getItem(idx);
            if (!item->enabled())
                setData(idx.sibling(idx.row(), 1), true, EnabledRole);
            restoreAll(idx);
        }
    }
}

void oskar_SettingsModel::saveFromParentIndex(const QModelIndex& parent)
{
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            const oskar_SettingsItem* item = getItem(idx);
            if (!item->value().isNull())
                settings_->setValue(item->key(), item->value());
            saveFromParentIndex(idx);
        }
    }
}


oskar_SettingsModelFilter::oskar_SettingsModelFilter(QObject* parent)
: QSortFilterProxyModel(parent),
  hideUnsetItems_(false)
{
    setDynamicSortFilter(true);
}

oskar_SettingsModelFilter::~oskar_SettingsModelFilter()
{
}

QVariant oskar_SettingsModelFilter::data(const QModelIndex& index,
        int role) const
{
    if (!filterText_.isEmpty())
    {
        if (role == Qt::BackgroundRole && index.column() == 0)
        {
            QString label = QSortFilterProxyModel::data(index,
                    Qt::DisplayRole).toString();
            if (label.contains(filterText_, Qt::CaseInsensitive))
                return QColor("#FFFF9F");
        }
    }
    return QSortFilterProxyModel::data(index, role);
}

bool oskar_SettingsModelFilter::hideUnsetItems() const
{
    return hideUnsetItems_;
}


// Public slots.

void oskar_SettingsModelFilter::setFilterText(QString value)
{
    filterText_ = value;
    invalidate();
}

void oskar_SettingsModelFilter::setHideUnsetItems(bool value)
{
    if (value != hideUnsetItems_)
    {
        hideUnsetItems_ = value;
        invalidate();
    }
}

// Protected methods.

bool oskar_SettingsModelFilter::filterAcceptsChildren(int sourceRow,
        const QModelIndex& sourceParent) const
{
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    if (!idx.isValid())
        return false;

    int childCount = idx.model()->rowCount(idx);
    for (int i = 0; i < childCount; ++i)
    {
        if (filterAcceptsCurrentRow(i, idx))
            return true;
        if (filterAcceptsChildren(i, idx))
            return true;
    }
    return false;
}

bool oskar_SettingsModelFilter::filterAcceptsCurrentRow(
            const QModelIndex& idx) const
{
    bool visible = !(hideUnsetItems_ && !(sourceModel()->data(idx,
            oskar_SettingsModel::VisibleRole).toBool()));
    QString labelCurrent = sourceModel()->data(idx, Qt::DisplayRole).toString();
    bool pass = (labelCurrent.contains(filterText_, Qt::CaseInsensitive) ||
            false) && visible;
    return pass;
}

bool oskar_SettingsModelFilter::filterAcceptsCurrentRow(int sourceRow,
            const QModelIndex& sourceParent) const
{
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    return filterAcceptsCurrentRow(idx);
}

bool oskar_SettingsModelFilter::filterAcceptsRow(int sourceRow,
            const QModelIndex& sourceParent) const
{
    // Check if the item is hidden.
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    if (sourceModel()->data(idx, oskar_SettingsModel::HiddenRole).toBool())
        return false;

    // Check if filter accepts this row.
    if (filterAcceptsCurrentRow(idx))
        return true;

    // Check if filter accepts any parent.
    QModelIndex parent = sourceParent;
    while (parent.isValid() && !hideUnsetItems_)
    {
        if (filterAcceptsCurrentRow(parent.row(), parent.parent()))
            return true;
        parent = parent.parent();
    }

    // Check if filter accepts any child.
    if (filterAcceptsChildren(sourceRow, sourceParent))
        return true;

    return false;
}
