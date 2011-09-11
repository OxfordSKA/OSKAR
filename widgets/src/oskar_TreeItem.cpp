#include "widgets/oskar_TreeItem.h"

#include <QtCore/QStringList>

oskar_TreeItem::oskar_TreeItem(const QList<QVariant>& data, oskar_TreeItem* parent)
{
    _parentItem = parent;
    _itemData = data;
}

oskar_TreeItem::~oskar_TreeItem()
{
    qDeleteAll(_childItems);
}


void oskar_TreeItem::appendChild(oskar_TreeItem* item)
{
    _childItems.append(item);
}


oskar_TreeItem* oskar_TreeItem::child(int row)
{
    return _childItems.value(row);
}


int oskar_TreeItem::childCount() const
{
    return _childItems.count();
}

int oskar_TreeItem::columnCount() const
{
    return _itemData.count();
}

QVariant oskar_TreeItem::data(int column) const
{
    return _itemData.value(column);
}

oskar_TreeItem* oskar_TreeItem::parent()
{
    return _parentItem;
}

int oskar_TreeItem::row() const
{
    if (_parentItem)
        return _parentItem->_childItems.indexOf(const_cast<oskar_TreeItem*>(this));
    return 0;
}

