#include "widgets/oskar_SettingsItem.h"

#include <QtCore/QStringList>
#include <cstdio>

oskar_SettingsItem::oskar_SettingsItem(const QString& key, const QString& keyShort,
        int type, const QVector<QVariant>& data, oskar_SettingsItem* parent)
{
    key_ = key;
    keyShort_ = keyShort;
    type_ = type;
    parentItem_ = parent;
    itemData_ = data;
}

oskar_SettingsItem::~oskar_SettingsItem()
{
    qDeleteAll(childItems_);
}

void oskar_SettingsItem::appendChild(oskar_SettingsItem* item)
{
    childItems_.append(item);
}

oskar_SettingsItem* oskar_SettingsItem::child(int row)
{
    return childItems_.value(row);
}

int oskar_SettingsItem::childCount() const
{
    return childItems_.count();
}

int oskar_SettingsItem::childNumber() const
{
    if (parentItem_)
        return parentItem_->childItems_.indexOf(const_cast<oskar_SettingsItem*>(this));
    return 0;
}

int oskar_SettingsItem::columnCount() const
{
    return itemData_.count();
}

QVariant oskar_SettingsItem::data(int column) const
{
    return itemData_.value(column);
}

bool oskar_SettingsItem::insertColumns(int position, int columns)
{
    if (position < 0 || position > itemData_.size())
        return false;

    for (int column = 0; column < columns; ++column)
        itemData_.insert(position, QVariant());

    foreach (oskar_SettingsItem* child, childItems_)
        child->insertColumns(position, columns);

    return true;
}

oskar_SettingsItem* oskar_SettingsItem::parent()
{
    return parentItem_;
}

bool oskar_SettingsItem::setData(int column, const QVariant &value)
{
    if (column < 0 || column >= itemData_.size())
        return false;

    itemData_[column] = value;
    return true;
}

QString oskar_SettingsItem::key() const
{
    return key_;
}

QString oskar_SettingsItem::keyShort() const
{
    return keyShort_;
}

int oskar_SettingsItem::type() const
{
    return type_;
}
