#include "widgets/ConfigOptionsTable.h"

#include <iostream>
using namespace std;

namespace oskar {

/**
 * @details
 */
ConfigOptionsTable::ConfigOptionsTable(QWidget *parent)
: QTableWidget(parent)
{
    setHeaders(QStringList() << "Key" << "Value");
}


ConfigOptionsTable::~ConfigOptionsTable()
{
}


/**
 * @details
 */
void ConfigOptionsTable::setHeaders(const QStringList & headers)
{
    setColumnCount(headers.size());
    setHorizontalHeaderLabels(headers);
}


/// Sets the configuration option for the specified key optionally
/// setting the associated delegate.
void ConfigOptionsTable::setConfigOption(const QString& key,
        const QVariant& value, QAbstractItemDelegate* delegate,
        unsigned column)
{
    // Check the key doesn't already exist.
    QList<QTableWidgetItem*> items = findItems(key, Qt::MatchExactly);

    if (items.size() == 0)
    {
        insertRow(rowCount());
        const int row = rowCount() - 1;

        if (delegate)
            setItemDelegateForRow(row, delegate);

        QTableWidgetItem * keyItem = new QTableWidgetItem(key);
        keyItem->setFlags(Qt::ItemIsEnabled);

        QFont font;
        font.setWeight(QFont::DemiBold);
        keyItem->setFont(font);

        setItem(row, 0, keyItem);
        QTableWidgetItem * tableItem = new QTableWidgetItem;
        tableItem->setData(Qt::EditRole, value);
        setItem(row, column, tableItem);
    }
    else if (items.size() == 1)
    {
        QTableWidgetItem * tableItem = item(items.at(0)->row(), column);
        tableItem->setData(Qt::EditRole, value);
    }
    else
        throw QString("ConvFuncTable::setConfigOption()");
}


/// Returns the configuration option specified by the key.
QVariant ConfigOptionsTable::getConfigOption(const QString& key,
        unsigned column) const
{
    // Match key based on the match flag and return a list of items found.
    QList<QTableWidgetItem*> items = findItems(key, Qt::MatchExactly);

    // If more than one item is found there is a problem.
    if (items.size() > 1)
    {
        throw QString("ConvFuncTable::getConfigOption(): "
                "More than one value found for key %1.").arg(key);
    }

    // Return the value associated with the item key.
    return item(items.at(0)->row(), column)->data(Qt::EditRole);
}


} // namespace oskar
