#ifndef CONFIG_OPTIONS_TABLE_H
#define CONFIG_OPTIONS_TABLE_H

#include <QtGui/QWidget>
#include <QtGui/QTableWidget>
#include <QtCore/QVariant>
#include <QtCore/QStringList>

namespace oskar {

/**
 * @class ConfigOptionsTable
 *
 * @brief
 *
 * @details
 */

class ConfigOptionsTable : public QTableWidget
{
    Q_OBJECT

    public:
        /// Construct the Configuration options table widget.
        ConfigOptionsTable(QWidget * parent = 0);

        ~ConfigOptionsTable();

        /// Sets the headers of table columns.
        void setHeaders(const QStringList & headers);

    public slots:
        /// Sets the configuration option for the specified key optionally
        /// setting the associated delegate.
        void setConfigOption(const QString & key, const QVariant & value,
            QAbstractItemDelegate * delegate = 0, unsigned column = 1);

        /// Returns the configuration option specified by the key.
        QVariant getConfigOption(const QString & key, unsigned column = 1) const;
};

} // namespace oskar
#endif // CONFIG_OPTIONS_TABLE_H
