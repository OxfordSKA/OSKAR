#ifndef OSKAR_SETTINGSMODEL_H_
#define OSKAR_SETTINGSMODEL_H_

#include <QtCore/QAbstractItemModel>
#include <QtCore/QModelIndex>
#include <QtCore/QVariant>
#include <QtCore/QStringList>

class oskar_TreeItem;

class oskar_SettingsModel : public QAbstractItemModel
{
        Q_OBJECT

    public:
        oskar_SettingsModel(const QString& data, QObject* parent);
        virtual ~oskar_SettingsModel();

        QVariant data(const QModelIndex& index, int role) const;
        Qt::ItemFlags flags(const QModelIndex& index) const;
        QVariant headerData(int section, Qt::Orientation orientation,
                int role = Qt::DisplayRole) const;
        QModelIndex index(int row, int column,
                const QModelIndex& parent = QModelIndex()) const;
        QModelIndex parent(const QModelIndex& index) const;
        int rowCount(const QModelIndex& parent = QModelIndex()) const;
        int columnCount(const QModelIndex& parent = QModelIndex()) const;

    private:
        void _setupModelData(const QStringList& lines, oskar_TreeItem* parent);

    private:
        oskar_TreeItem* _rootItem;
};


#endif // OSKAR_SETTINGSMODEL_H_
