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

    // Set up the root item.
    rootItem_ = new oskar_SettingsItem(QString(), QString(),
            oskar_SettingsItem::CAPTION_ONLY, "Setting", "Value");

    // Simulator settings.
    setCaption("simulator", "Simulator settings");
    registerSetting("simulator/double_precision", "Use double precision", oskar_SettingsItem::BOOL);
    setTooltip("simulator/double_precision", "Determines whether double precision arithmetic is used for the simulation");
    registerSetting("simulator/max_sources_per_chunk", "Max. number of sources per chunk", oskar_SettingsItem::INT);
    registerSetting("simulator/cuda_device_ids", "CUDA device IDs to use", oskar_SettingsItem::INT_CSV_LIST);

    // Sky model file settings.
    setCaption("sky", "Sky model settings");
    registerSetting("sky/oskar_source_file", "Input OSKAR source file", oskar_SettingsItem::INPUT_FILE_NAME);
    setCaption("sky/oskar_source_file/filter", "Filter settings");
    registerSetting("sky/oskar_source_file/filter/flux_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/oskar_source_file/filter/flux_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/oskar_source_file/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/oskar_source_file/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/gsm_file", "Input Global Sky Model file", oskar_SettingsItem::INPUT_FILE_NAME);
    setCaption("sky/gsm_file/filter", "Filter settings");
    registerSetting("sky/gsm_file/filter/flux_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/gsm_file/filter/flux_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/gsm_file/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/gsm_file/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/output_sky_file", "Output OSKAR source file", oskar_SettingsItem::OUTPUT_FILE_NAME);

    // Sky model generator settings.
    setCaption("sky/generator", "Generators");
    setCaption("sky/generator/random_power_law", "Random, power-law in flux");
    registerSetting("sky/generator/random_power_law/num_sources", "Number of sources", oskar_SettingsItem::INT);
    registerSetting("sky/generator/random_power_law/flux_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_power_law/flux_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_power_law/power", "Power law index", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_power_law/seed", "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setCaption("sky/generator/random_power_law/filter", "Filter settings");
    registerSetting("sky/generator/random_power_law/filter/flux_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_power_law/filter/flux_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_power_law/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_power_law/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    setCaption("sky/generator/random_broken_power_law", "Random, broken power-law in flux");
    registerSetting("sky/generator/random_broken_power_law/num_sources", "Number of sources", oskar_SettingsItem::INT);
    registerSetting("sky/generator/random_broken_power_law/flux_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/flux_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/power1", "Power law index 1", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/power2", "Power law index 2", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/threshold", "Threshold (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/seed", "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setCaption("sky/generator/random_broken_power_law/filter", "Filter settings");
    registerSetting("sky/generator/random_broken_power_law/filter/flux_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/filter/flux_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/random_broken_power_law/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    setCaption("sky/generator/healpix", "HEALPix (uniform, all sky) grid");
    registerSetting("sky/generator/healpix/nside", "Nside", oskar_SettingsItem::INT);
    setCaption("sky/generator/healpix/filter", "Filter settings");
    registerSetting("sky/generator/healpix/filter/flux_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/healpix/filter/flux_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/healpix/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("sky/generator/healpix/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);

    // Telescope model settings.
    setCaption("telescope", "Telescope model settings");
    registerSetting("telescope/station_positions_file", "Station positions file", oskar_SettingsItem::INPUT_FILE_NAME);
    registerSetting("telescope/station_layout_directory", "Station layout directory", oskar_SettingsItem::INPUT_DIR_NAME);
    registerSetting("telescope/longitude_deg", "Longitude (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("telescope/latitude_deg", "Latitude (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("telescope/altitude_m", "Altitude (m)", oskar_SettingsItem::DOUBLE);
    setCaption("telescope/station", "Station settings");
    registerSetting("telescope/station/enable_beam", "Enable", oskar_SettingsItem::BOOL);
    registerSetting("telescope/station/normalise_beam", "Normalise", oskar_SettingsItem::BOOL);
    registerSetting("telescope/station/element_amp_gain", "Element amplitude gain", oskar_SettingsItem::DOUBLE);
    registerSetting("telescope/station/element_amp_error", "Element amplitude standard deviation", oskar_SettingsItem::DOUBLE);
    registerSetting("telescope/station/element_phase_offset_deg", "Element phase offset (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("telescope/station/element_phase_error_deg", "Element phase standard deviation (deg)", oskar_SettingsItem::DOUBLE);

    // Observation settings.
    setCaption("observation", "Observation settings");
    registerSetting("observation/num_channels", "Number of channels", oskar_SettingsItem::INT);
    registerSetting("observation/start_frequency_hz", "Start frequency (Hz)", oskar_SettingsItem::DOUBLE);
    registerSetting("observation/frequency_inc_hz", "Frequency increment (Hz)", oskar_SettingsItem::DOUBLE);
    registerSetting("observation/channel_bandwidth_hz", "Channel bandwidth (Hz)", oskar_SettingsItem::DOUBLE);
    registerSetting("observation/phase_centre_ra_deg", "Phase centre RA (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("observation/phase_centre_dec_deg", "Phase centre Dec (deg)", oskar_SettingsItem::DOUBLE);
    registerSetting("observation/num_vis_dumps", "Number of visibility dumps", oskar_SettingsItem::INT);
    registerSetting("observation/num_vis_ave", "Number of visibility averages", oskar_SettingsItem::INT);
    registerSetting("observation/num_fringe_ave", "Number of fringe averages", oskar_SettingsItem::INT);
    registerSetting("observation/start_time_utc", "Start time (UTC)", oskar_SettingsItem::DATE_TIME);
    registerSetting("observation/length", "Observation length (H:M:S)", oskar_SettingsItem::TIME);
    registerSetting("observation/oskar_vis_filename", "Output OSKAR visibility file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    registerSetting("observation/ms_filename", "Output Measurement Set name", oskar_SettingsItem::OUTPUT_FILE_NAME);
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

int oskar_SettingsModel::columnCount(const QModelIndex& /*parent*/) const
{
    return 2;
}

QVariant oskar_SettingsModel::data(const QModelIndex& index, int role) const
{
    // Check for roles that do not depend on the index.
    if (role == IterationKeysRole)
        return iterationKeys_;
    else if (role == OutputKeysRole)
        return outputKeys_;

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
    else if (role == Qt::ToolTipRole)
        return item->tooltip();
    else if (role == KeyRole)
        return item->key();
    else if (role == TypeRole)
        return item->type();
    else if (role == VisibleRole)
        return item->visible();
    else if (role == IterationNumRole)
        return item->iterationNum();
    else if (role == IterationIncRole)
        return item->iterationInc();

    // Check for roles in specific columns.
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

const oskar_SettingsItem* oskar_SettingsModel::getItem(const QString& key) const
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

QMap<int, QVariant> oskar_SettingsModel::itemData(const QModelIndex& index) const
{
    QMap<int, QVariant> d;
    d.insert(KeyRole, data(index, KeyRole));
    d.insert(TypeRole, data(index, TypeRole));
    d.insert(VisibleRole, data(index, VisibleRole));
    d.insert(IterationNumRole, data(index, IterationNumRole));
    d.insert(IterationIncRole, data(index, IterationIncRole));
    d.insert(IterationKeysRole, data(index, IterationKeysRole));
    d.insert(OutputKeysRole, data(index, OutputKeysRole));
    return d;
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

    // Check if this is an output file.
    if (type == oskar_SettingsItem::OUTPUT_FILE_NAME)
        outputKeys_.append(key);
}

int oskar_SettingsModel::rowCount(const QModelIndex& parent) const
{
    return getItem(parent)->childCount();
}

void oskar_SettingsModel::setCaption(const QString& key, const QString& caption)
{
    QModelIndex idx = index(key);
    setData(idx, caption);
}

void oskar_SettingsModel::setTooltip(const QString& key, const QString& tooltip)
{
    QModelIndex idx = index(key);
    setData(idx, tooltip, Qt::ToolTipRole);
}

bool oskar_SettingsModel::setData(const QModelIndex& idx,
        const QVariant& value, int role)
{
    if (!idx.isValid())
        return false;

    // Get a pointer to the item.
    oskar_SettingsItem* item = getItem(idx);

    // Check for roles common to all columns.
    if (role == Qt::ToolTipRole)
    {
        item->setTooltip(value.toString());
        emit dataChanged(idx, idx);
        return true;
    }
    else if (role == VisibleRole)
    {
        item->setVisible(value.toBool());
        emit dataChanged(idx, idx);
        return true;
    }
    else if (role == IterationNumRole)
    {
        item->setIterationNum(value.toInt());
        emit dataChanged(idx, idx);
        return true;
    }
    else if (role == IterationIncRole)
    {
        item->setIterationInc(value);
        emit dataChanged(idx, idx);
        return true;
    }
    else if (role == SetIterationRole)
    {
        if (!iterationKeys_.contains(item->key()))
        {
            iterationKeys_.append(item->key());
            emit dataChanged(idx, idx);
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
            emit dataChanged(idx, idx);
            foreach (QString k, iterationKeys_)
            {
                QModelIndex idx = index(k);
                emit dataChanged(index(idx.row(), 0, parent(idx)),
                        index(idx.row(), columnCount(), parent(idx)));
            }
            return true;
        }
        return false;
    }

    QVariant data;
    if (role == Qt::EditRole)
        data = value;
    else if (role == Qt::CheckStateRole)
        data = value.toBool() ? QString("true") : QString("false");

    if (idx.column() == 0)
    {
        item->setCaption(data.toString());
        emit dataChanged(idx, idx);
        return true;
    }
    else if (idx.column() == 1)
    {
        item->setValue(data);
        if (settings_)
        {
            if (value.toString().isEmpty())
                settings_->remove(item->key());
            else
                settings_->setValue(item->key(), data);
            settings_->sync();
        }
        emit dataChanged(idx, idx);
        return true;
    }

    return false;
}

void oskar_SettingsModel::setSettingsFile(const QString& filename)
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

void oskar_SettingsModel::setValue(const QString& key, const QVariant& value)
{
    // Get the model index.
    QModelIndex idx = index(key);
    idx = idx.sibling(idx.row(), 1);

    // Set the data.
    setData(idx, value, Qt::EditRole);
}

// Private methods.

void oskar_SettingsModel::append(const QString& key, const QString& subkey,
        int type, const QString& caption, const QVariant& defaultValue,
        const QModelIndex& parent)
{
    oskar_SettingsItem *parentItem = getItem(parent);

    beginInsertRows(parent, rowCount(), rowCount());
    oskar_SettingsItem* item = new oskar_SettingsItem(key, subkey, type,
            caption, defaultValue, parentItem);
    parentItem->appendChild(item);
    endInsertRows();
    hash_.insert(key, item);
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
            oskar_SettingsItem* item = getItem(idx);
            item->setVisible(false);
            if (item->type() != oskar_SettingsItem::CAPTION_ONLY)
            {
                if (settings_->contains(item->key()))
                {
                    item->setValue(settings_->value(item->key()));
                    item->setVisible(true);
                }
                else
                {
                    item->setValue(item->defaultValue());
                }
                emit dataChanged(idx, idx);
            }
            loadFromParentIndex(idx);
        }
    }
}


oskar_SettingsFilterModel::oskar_SettingsFilterModel(QObject* parent)
: QSortFilterProxyModel(parent)
{
}

oskar_SettingsFilterModel::~oskar_SettingsFilterModel()
{
}

bool oskar_SettingsFilterModel::filterAcceptsRow(int sourceRow,
            const QModelIndex& sourceParent) const
{
    return true;
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    return sourceModel()->data(idx, oskar_SettingsModel::VisibleRole).toBool();
}
