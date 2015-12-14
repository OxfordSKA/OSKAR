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

#include <oskar_SettingsModel.h>
#include <oskar_SettingsItem.h>
#include <oskar_version.h>

#include <QtGui/QApplication>
#include <QtGui/QFontMetrics>
#include <QtGui/QIcon>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QModelIndex>
#include <QtCore/QSettings>
#include <QtCore/QSize>
#include <QtCore/QStringList>
#include <QtCore/QVariant>
#include <cfloat>
#include <iostream>
using namespace std;

oskar_SettingsModel::oskar_SettingsModel(QObject* parent)
: QAbstractItemModel(parent),
  settings_(NULL),
  rootItem_(NULL),
  version_(OSKAR_VERSION_STR),
  lastModified_(QDateTime::currentDateTime())
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
    // Get a pointer to the item.
    if (!index.isValid())
        return QVariant();
    oskar_SettingsItem* item = getItem(index);

    // Check for roles common to all columns.
    if (role == Qt::ForegroundRole)
    {
        if (item->disabled())
            return QColor(Qt::lightGray);
        else if (item->critical())
            return QColor(Qt::white);
        else if (item->valueSet())
            return QColor(Qt::blue);
        else if (item->required())
            return QColor(Qt::red);
        else
            return QColor(64, 64, 64);
    }
    else if (role == Qt::BackgroundRole)
    {
        // TODO only set background role if not disabled and if no child items
        // are required and not set (if shown)

        // FIXME Recursively iterate over child to establish if any are not set,
        // required and not disabled... if so there is a critical child
        // so the background colour should be set...

//        if (item->key() == "telescope") // IF == HACK TO AVOID TOO MUCH PRINTING
//        {
//            // Loop over items that are children to this item and if they are
//            // unset, required and not disabled add 1 to the local critical counter.
//            for (int i = 0; i < item->childCount(); ++i)
//            {
//                oskar_SettingsItem* item_ = item->child(i);
//            }
//        }

        // Only set the critical/required background colour for items which
        // have their dependencies satisfied (and are therefore not disabled)
        if (item->critical() && !item->disabled())
        {
            if (index.column() == 0)
                return QColor(0, 48, 255, 160);
            else if (item->type() != oskar_SettingsItem::LABEL)
                return QColor(255, 64, 64, 255);
        }
        if (index.column() == 1)
            return QColor(0, 0, 192, 12);
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
    else if (role == DisabledRole)
        return item->disabled();
    else if (role == VisibleRole)
        return item->valueSet() || item->required();
    else if (role == OptionsRole)
        return item->options();
    // Note: Maybe icons should be disabled unless there is an icon
    // for everything. This would avoid indentation level problems with
    // option trees of depth greater than 1.
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
            return item->label();
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
    if (item->disabled())
        return Qt::ItemIsSelectable;

    if (index.column() == 0 || item->type() == oskar_SettingsItem::LABEL)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    }
    else if (index.column() == 1 && item->type() == oskar_SettingsItem::BOOL)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable |  Qt::ItemIsUserCheckable;
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
        filename_ = filename;

        // Delete any existing settings object.
        if (settings_)
            delete settings_;

        // Create new settings object from supplied filename.
        settings_ = new QSettings(filename, QSettings::IniFormat);

        // Display the contents of the file.
        loadFromParentIndex(QModelIndex());
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
        filename_ = filename;

        // Delete any existing settings object.
        if (settings_)
            delete settings_;

        // Create new settings object from supplied filename.
        settings_ = new QSettings(filename, QSettings::IniFormat);
        writeVersion();

        // Set the contents of the file.
        saveFromParentIndex(QModelIndex());
    }
}

bool oskar_SettingsModel::setData(const QModelIndex& idx,
        const QVariant& value, int role)
{
    // Check for roles that do not depend on the index.
    if (role == CheckExternalChangesRole)
    {
        if (!QFile::exists(filename_))
            return false;
        QFileInfo fileInfo(filename_);
        if (fileInfo.lastModified() > lastModified_.addMSecs(200))
        {
            loadSettingsFile(filename_);
            emit fileReloaded();
        }
        return true;
    }

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
    else if (role == DisabledRole)
    {
        item->setDisabled(value.toBool());
        emit dataChanged(topLeft, bottomRight);
        return true;
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
                writeVersion();
                if (data.isNull())
                    settings_->remove(item->key());
                else
                {
                    settings_->setValue(item->key(), data);
                    for (int i = 0; i < item->childCount(); ++i)
                    {
                        oskar_SettingsItem* child = item->child(i);
                        QModelIndex childIdx = index(child->key());
                        if (settings_->allKeys().contains(child->key())) continue;
                        childIdx = childIdx.sibling(childIdx.row(), columnCount() - 1);
                        setData(childIdx, child->value(), Qt::EditRole);
                    }
                }
                settings_->sync();
            }
            lastModified_ = QDateTime::currentDateTime();

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
                        && dependent->disabled())
                {
                    setData(idx_dependent, false, DisabledRole);
                }
                else if (dependent->dependencyValue() != item->valueOrDefault()
                        && !dependent->disabled())
                {
                    setData(idx_dependent, true, DisabledRole);
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

    // Do the initial check and set the flag to indicate whether disabled or not.
    oskar_SettingsItem* dependency = getItem(idx_dependency);
    if (dependency->valueOrDefault() == dependency_value)
        setData(idx_dependent, false, DisabledRole);
    else
        setData(idx_dependent, true, DisabledRole);
}

void oskar_SettingsModel::setLabel(const QString& key, const QString& label)
{
    QModelIndex idx = index(key);
    setData(idx, label);
}

void oskar_SettingsModel::setTooltip(const QString& key, const QString& tooltip)
{
    QModelIndex idx = index(key);
    setData(idx, "<p>" + tooltip + "</p>", Qt::ToolTipRole);
}

void oskar_SettingsModel::setValue(const QString& key, const QVariant& value)
{
    // Get the model index.
    QModelIndex idx = index(key);
    idx = idx.sibling(idx.row(), 1);

    // Set the data.
    setData(idx, value, Qt::EditRole);
}

//QHash<QString, QVariant> oskar_SettingsModel::settings() const
//{
//    QHash<QString, QVariant> hash;
//    foreach (oskar_SettingsItem* item, itemHash_)
//    {
//        hash.insert(item->key(), item->valueOrDefault());
//    }
//    return hash;
//}

QString oskar_SettingsModel::version() const
{
    if (settings_)
       return settings_->value("version").toString();
    return QString();
}

void oskar_SettingsModel::setVersion(const QString& value)
{
    version_ = value;
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

void oskar_SettingsModel::writeVersion()
{
    if (settings_)
    {
        // Write a version key only if it doesn't already exist in the file.
        if (!settings_->contains("version"))
            settings_->setValue("version", version_);
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
    // Check if the item is disabled.
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
#if 0
    if (sourceModel()->data(idx, oskar_SettingsModel::DisabledRole).toBool())
        return false;
#endif

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
