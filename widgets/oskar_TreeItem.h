#ifndef OSKAR_TREEITEM_H_
#define OSKAR_TREEITEM_H_

#include <QtCore/QList>
#include <QtCore/QVariant>

class oskar_TreeItem
{
    public:
        oskar_TreeItem(const QList<QVariant>& data, oskar_TreeItem* parent = NULL);
        ~oskar_TreeItem();

        oskar_TreeItem* child(int row);
        int childCount() const;
        int columnCount() const;
        QVariant data(int column) const;
        void appendChild(oskar_TreeItem* child);
        int row() const;
        oskar_TreeItem* parent();

    private:
        QList<oskar_TreeItem*> _childItems;
        QList<QVariant> _itemData;
        oskar_TreeItem* _parentItem;
};


#endif // OSKAR_TREEITEM_H_
