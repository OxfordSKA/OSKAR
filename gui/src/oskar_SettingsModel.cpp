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

#include "gui/oskar_SettingsModel.h"
#include "settings/oskar_settings_types.h"
#include "settings/oskar_SettingsTree.h"
#include "settings/oskar_SettingsNode.h"
#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_SettingsValue.h"
#include <QApplication>
#include <QFile>
#include <QFileInfo>
#include <QFontMetrics>
#include <QIcon>
#include <QModelIndex>
#include <QPalette>
#include <QSize>
#include <QStringList>
#include <QVariant>

using namespace std;

namespace oskar {

SettingsModel::SettingsModel(SettingsTree* settings, QObject* parent)
: QAbstractItemModel(parent),
  settings_(settings),
  iconOpen_(QIcon(":/icons/open.png")),
  iconSave_(QIcon(":/icons/save.png")),
  lastModified_(QDateTime::currentDateTime()),
  displayKey_(false)
{
}

SettingsModel::~SettingsModel()
{
}

void SettingsModel::beginReset()
{
    beginResetModel();
}

void SettingsModel::endReset()
{
    endResetModel();
}

int SettingsModel::columnCount(const QModelIndex& /*parent*/) const
{
    return 2;
}

QVariant SettingsModel::data(const QModelIndex& index, int role) const
{
    // Get a pointer to the item.
    if (!index.isValid())
        return QVariant();

    const SettingsNode* node = get_node(index);
    const SettingsValue& value = node->settings_value();
    const SettingsValue::TypeId type = value.type();
    const char* key = node->key();

    // Check for roles common to all columns.
    switch (role)
    {
    case Qt::ForegroundRole:
    {
        QPalette palette = QApplication::palette((QWidget*) 0);
        if (!settings_->dependencies_satisfied(key))
            return palette.color(QPalette::Disabled, QPalette::Text);
        if (settings_->is_critical(key))
            return QColor(Qt::white);
        if (node->value_or_child_set())
            return palette.color(QPalette::Normal, QPalette::Link);
        return palette.color(QPalette::Normal, QPalette::Text);
    }
    case Qt::BackgroundRole:
    {
        bool disabled = !settings_->dependencies_satisfied(key);
        if (settings_->is_critical(key) && !disabled)
        {
            if (index.column() == 0)
                return QColor(0, 48, 255, 160);
            else if (node->item_type() != SettingsItem::LABEL)
                return QColor(255, 64, 64, 255);
        }
        if (index.column() == 1)
            return QColor(0, 0, 192, 12);
        break;
    }
    case Qt::ToolTipRole:
    {
        QString tooltip = QString(node->description());
        if (!tooltip.isEmpty())
        {
            tooltip = "<p>" + tooltip + "</p>";
            if (node->is_required())
                tooltip.append(" [Required]");
//            if (node->item_type() == SettingsItem::SETTING)
//                tooltip.append(" [" + QString::fromStdString(
//                        node->value().type_name()) + "]");
        }
        return tooltip;
    }
    case Qt::EditRole:
    {
        if (node->item_type() == SettingsItem::SETTING)
            return QString(node->value());
        break;
    }
    case KeyRole:
        return QString(key);
    case ValueRole:
        return QString(node->value());
    case DefaultRole:
        return QString(node->default_value());
    case TypeRole:
        return type;
    case ItemTypeRole:
        return node->item_type();
    case RangeRole:
    {
        QList<QVariant> range;
        switch (type)
        {
        case SettingsValue::INT_RANGE:
        {
            range.append(value.get<IntRange>().min());
            range.append(value.get<IntRange>().max());
            return range;
        }
        case SettingsValue::DOUBLE_RANGE:
        {
            range.append(value.get<DoubleRange>().min());
            range.append(value.get<DoubleRange>().max());
            return range;
        }
        default:
            break;
        }
        return range;
    }
    case ExtRangeRole:
    {
        QList<QVariant> range;
        switch (type)
        {
        case SettingsValue::INT_RANGE_EXT:
        {
            range.append(value.get<IntRangeExt>().min());
            range.append(value.get<IntRangeExt>().max());
            range.append(QString(value.get<IntRangeExt>().ext_min()));
            range.append(QString(value.get<IntRangeExt>().ext_max()));
            return range;
        }
        case SettingsValue::DOUBLE_RANGE_EXT:
        {
            range.append(value.get<DoubleRangeExt>().min());
            range.append(value.get<DoubleRangeExt>().max());
            range.append(QString(value.get<DoubleRangeExt>().ext_min()));
            range.append(QString(value.get<DoubleRangeExt>().ext_max()));
            return range;
        }
        default:
            break;
        }
        return range;
    }
    case OptionsRole:
    {
        QStringList options;
        if (type == SettingsValue::OPTION_LIST)
        {
            const OptionList& l = value.get<OptionList>();
            for (int i = 0; i < l.size(); ++i)
                options.push_back(QString(l.option(i)));
        }
        return options;
    }
    default:
        break;
    }

    // Check for roles in specific columns.
    if (index.column() == 0)
    {
        switch (role)
        {
        case Qt::SizeHintRole:
        {
            int width = QApplication::fontMetrics().width(
                    (data(index, Qt::DisplayRole)).toString()) + 20;
            return QSize(width, 26);
        }
        case Qt::DisplayRole:
            return displayKey_ ? QString(key) : QString(node->label());
        default:
            break;
        }
    }
    else if (index.column() == 1)
    {
        switch (role)
        {
        case Qt::DisplayRole:
        {
            if (node->item_type() == SettingsItem::SETTING)
                return QString(node->value());
            break;
        }
        case Qt::CheckStateRole:
        {
            if (type == SettingsValue::BOOL)
                return value.get<Bool>().value() ? Qt::Checked : Qt::Unchecked;
            break;
        }
        case Qt::DecorationRole:
        {
            if (type == SettingsValue::INPUT_FILE ||
                    type == SettingsValue::INPUT_FILE_LIST ||
                    type == SettingsValue::INPUT_DIRECTORY)
                return iconOpen_;
            else if (type == SettingsValue::OUTPUT_FILE)
                return iconSave_;
            break;
        }
        default:
            break;
        }
    }

    return QVariant();
}

Qt::ItemFlags SettingsModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    const SettingsNode* node = get_node(index);
    if (!settings_->dependencies_satisfied(node->key()))
        return Qt::ItemIsSelectable;
    int column = index.column();
    if (column == 0 || node->item_type() == SettingsItem::LABEL)
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    if (column == 1 && node->settings_value().type() == SettingsValue::BOOL)
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable |  Qt::ItemIsUserCheckable;

    return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant SettingsModel::headerData(int section,
        Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        if (section == 0)
            return "Setting";
        else if (section == 1)
            return "Value";
    }

    return QVariant();
}

QModelIndex SettingsModel::index(int row, int column,
        const QModelIndex& parent) const
{
    if (parent.isValid() && parent.column() != 0)
        return QModelIndex();

    const SettingsNode* child_node = get_node(parent)->child(row);
    // This is from the Qt documentation: const_cast is unavoidable here.
    SettingsNode* node = const_cast<SettingsNode*>(child_node);
    if (child_node)
        return createIndex(row, column, static_cast<void*>(node));
    else
        return QModelIndex();
}

void SettingsModel::load_settings_file(const QString& filename)
{
    if (!filename.isEmpty()) filename_ = filename;
    lastModified_ = QDateTime::currentDateTime();
    settings_->load(filename.toLatin1().constData());
    refresh(QModelIndex());
}

void SettingsModel::save_settings_file(const QString& filename)
{
    if (!filename.isEmpty()) filename_ = filename;
    lastModified_ = QDateTime::currentDateTime();
    settings_->save(filename.toLatin1().constData());
}

QModelIndex SettingsModel::parent(const QModelIndex& index) const
{
    if (!index.isValid())
        return QModelIndex();

    const SettingsNode* parent = get_node(index)->parent();
    if (parent == settings_->root_node())
        return QModelIndex();

    SettingsNode* node = const_cast<SettingsNode*>(parent);
    return createIndex(parent->child_number(), 0, static_cast<void*>(node));
}

void SettingsModel::refresh()
{
    refresh(QModelIndex());
}

int SettingsModel::rowCount(const QModelIndex& parent) const
{
    return get_node(parent)->num_children();
}

bool SettingsModel::setData(const QModelIndex& idx, const QVariant& value,
                            int role)
{
    // Check for roles that do not depend on the index.
    if (role == CheckExternalChangesRole)
    {
        if (!QFile::exists(filename_)) return false;
        QFileInfo fileInfo(filename_);
        if (fileInfo.lastModified() > lastModified_.addMSecs(200))
        {
            load_settings_file(filename_);
            lastModified_ = QDateTime::currentDateTime();
            emit fileReloaded();
        }
        return true;
    }
    else if (role == DisplayKeysRole)
    {
        displayKey_ = value.toBool();
        refresh(QModelIndex());
        return true;
    }

    if (!idx.isValid())
        return false;

    // Get a pointer to the item.
    const SettingsNode* node = get_node(idx);

    // Get model indexes for the row.
    QModelIndex topLeft = idx.sibling(idx.row(), 0);
    QModelIndex bottomRight = idx.sibling(idx.row(), columnCount() - 1);

    // Check for role type.
    if (role == Qt::EditRole || role == Qt::CheckStateRole)
    {
        QVariant data = value;
        if (role == Qt::CheckStateRole)
            data = value.toBool() ? QString("true") : QString("false");

        if (idx.column() == 1)
        {
            lastModified_ = QDateTime::currentDateTime();
            QString value;
            if (node->settings_value().type() == SettingsValue::INPUT_FILE_LIST)
            {
                QStringList l = data.toStringList();
                for (int i = 0; i < l.size(); ++i) {
                    value += l[i];
                    if (i < l.size()) value += ",";
                }
            }
            else {
                value = data.toString();
            }
            settings_->set_value(node->key(), value.toLatin1().constData());

            QModelIndex i(idx);
            while (i.isValid())
            {
                emit dataChanged(i.sibling(i.row(), 0),
                                 i.sibling(i.row(), columnCount()-1));
                i = i.parent();
            }
            emit dataChanged(topLeft, bottomRight);
            return true;
        }
    }
    else if (role == SettingsModel::ResetGroupRole)
    {
        if (idx.column() == 1) {
            reset_group_(node);
            lastModified_ = QDateTime::currentDateTime();
            // TODO(BM) call dataChanged on all children and parents too.
            // seems to work at the moment on the basis of luck or the right
            // click action used to call reset calling a redraw.
            emit dataChanged(topLeft, bottomRight);
//            QModelIndex i(idx);
//            while (i.isValid())
//            {
//                emit dataChanged(i.sibling(i.row(), 0),
//                                 i.sibling(i.row(), columnCount()-1));
//                i = i.parent();
//            }
            return true;
        }
    }
    return false;
}


// Private methods.

void SettingsModel::reset_group_(const SettingsNode* node)
{
    for (int i = 0; i < node->num_children(); ++i) {
        const SettingsNode* child = node->child(i);
        settings_->set_value(child->key(), child->default_value());
        reset_group_(child);
    }
}

const SettingsNode* SettingsModel::get_node(const QModelIndex& index) const
{
    if (index.isValid())
    {
        SettingsNode* node = static_cast<SettingsNode*>(index.internalPointer());
        if (node) return node;
    }
    return settings_->root_node();
}

void SettingsModel::refresh(const QModelIndex& parent)
{
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            emit dataChanged(idx, idx.sibling(idx.row(), 1));
            refresh(idx);
        }
    }
}


SettingsModelFilter::SettingsModelFilter(QObject* parent)
: QSortFilterProxyModel(parent)
{
    setDynamicSortFilter(true);
}

SettingsModelFilter::~SettingsModelFilter()
{
}

QVariant SettingsModelFilter::data(const QModelIndex& index, int role) const
{
    if (!filterRegExp().isEmpty())
    {
        if (role == Qt::BackgroundRole && index.column() == 0)
        {
            QString label = QSortFilterProxyModel::data(index,
                    Qt::DisplayRole).toString();
            if (label.contains(filterRegExp().pattern(), Qt::CaseInsensitive))
                return QColor("#FFFF9F");
        }
    }
    return QSortFilterProxyModel::data(index, role);
}

void SettingsModelFilter::setFilterRegExp(const QString& pattern)
{
    QSortFilterProxyModel::setFilterRegExp(pattern);
    invalidate();
}

// Protected methods.

bool SettingsModelFilter::filterAcceptsChildren(int sourceRow,
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

bool SettingsModelFilter::filterAcceptsCurrentRow(const QModelIndex& idx) const
{
    QString labelCurrent = sourceModel()->data(idx, Qt::DisplayRole).toString();
    return labelCurrent.contains(filterRegExp().pattern(), Qt::CaseInsensitive);
}

bool SettingsModelFilter::filterAcceptsCurrentRow(int sourceRow,
            const QModelIndex& sourceParent) const
{
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    return filterAcceptsCurrentRow(idx);
}

bool SettingsModelFilter::filterAcceptsRow(int sourceRow,
            const QModelIndex& sourceParent) const
{
    // Check if filter accepts this row.
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    if (filterAcceptsCurrentRow(idx))
        return true;

    // Check if filter accepts any parent.
    QModelIndex parent = sourceParent;
    while (parent.isValid())
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

} // namespace oskar
