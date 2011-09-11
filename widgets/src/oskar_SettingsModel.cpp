#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_TreeItem.h"


oskar_SettingsModel::oskar_SettingsModel(const QString& data, QObject* parent)
: QAbstractItemModel(parent)
{
    QList<QVariant> rootData;
    rootData << "Title" << "Summary";
    _rootItem = new oskar_TreeItem(rootData);
    _setupModelData(data.split(QString("\n")), _rootItem);
}

oskar_SettingsModel::~oskar_SettingsModel()
{
    delete _rootItem;
}



int oskar_SettingsModel::columnCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return static_cast<oskar_TreeItem*>(parent.internalPointer())->columnCount();
    else
        return _rootItem->columnCount();
}

QVariant oskar_SettingsModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (role != Qt::DisplayRole)
        return QVariant();

    oskar_TreeItem* item = static_cast<oskar_TreeItem*>(index.internalPointer());

    return item->data(index.column());
}


Qt::ItemFlags oskar_SettingsModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant oskar_SettingsModel::headerData(int section, Qt::Orientation orientation,
        int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return _rootItem->data(section);

    return QVariant();
}

QModelIndex oskar_SettingsModel::index(int row, int column,
        const QModelIndex& parent) const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    oskar_TreeItem* parentItem;

    if (!parent.isValid())
        parentItem = _rootItem;
    else
        parentItem = static_cast<oskar_TreeItem*>(parent.internalPointer());

    oskar_TreeItem* childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}


QModelIndex oskar_SettingsModel::parent(const QModelIndex& index) const
{
    if (!index.isValid())
        return  QModelIndex();

    oskar_TreeItem* childItem = static_cast<oskar_TreeItem*>(index.internalPointer());
    oskar_TreeItem* parentItem = childItem->parent();

    if (parentItem == _rootItem)
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}


int oskar_SettingsModel::rowCount(const QModelIndex& parent) const
{
    oskar_TreeItem* parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = _rootItem;
    else
        parentItem = static_cast<oskar_TreeItem*>(parent.internalPointer());

    return parentItem->childCount();
}



void oskar_SettingsModel::_setupModelData(const QStringList& lines,
        oskar_TreeItem* parent)
{
    QList<oskar_TreeItem*> parents;
    QList<int> indentations;
    parents << parent;
    indentations << 0;

    int number = 0;

    while (number < lines.count())
    {
        int position = 0;
        while (position < lines[number].length())
        {
            if (lines[number].mid(position, 1) != " ")
                break;
            position++;
        }

        QString lineData = lines[number].mid(position).trimmed();

        if (!lineData.isEmpty())
        {
            // Read the column data from the rest of the line.
            QStringList columnStrings = lineData.split("\t", QString::SkipEmptyParts);
            QList<QVariant> columnData;
            for (int column = 0; column < columnStrings.count(); ++column)
                columnData << columnStrings[column];

            if (position > indentations.last())
            {
                // The last child of the current parent is now the new parent
                // unless the current parent has no children.
                if (parents.last()->childCount() > 0)
                {
                    parents << parents.last()->child(parents.last()->childCount()-1);
                    indentations << position;
                }
            }
            else
            {
                while (position < indentations.last() && parents.count() > 0)
                {
                    parents.pop_back();
                    indentations.pop_back();
                }
            }

            // Append a new item to the current parent's list of children.
            parents.last()->appendChild(new TreeItem(columnData, parents.last()));
        }

        number++;
    }
}
