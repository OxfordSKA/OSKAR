#ifndef QTEST_CONFIG_OPTIONS_TABLE_H
#define QTEST_CONFIG_OPTIONS_TABLE_H

/**
 * @file QTestConfigOptionsTable.h
 */

#include <QtGui/QApplication>
#include <QtCore/QObject>
#include <QtTest/QtTest>
#include <QtCore/QVariant>
#include "widgets/ConfigOptionsTable.h"

using namespace oskar;

/**
 * @class QTestConfigOptionsTable
 *
 * @brief
 * Unit testing for the ConfigOptionsTable class.
 *
 * @details
 */

class QTestConfigOptionsTable : public QObject
{
    Q_OBJECT

    public:
        QTestConfigOptionsTable() { _table = new ConfigOptionsTable; }
        ~QTestConfigOptionsTable() { delete _table; }

    private slots:
        void test()
        {
            _table->show();
            _table->setConfigOption("doubleOption", 2.1);
            _table->setConfigOption("intOption", 4);
            _table->setConfigOption("boolOption", true);

            QCOMPARE(_table->getConfigOption("boolOption"), QVariant(true));
            QCOMPARE(_table->getConfigOption("intOption"), QVariant(int(4)));
            QCOMPARE(_table->getConfigOption("doubleOption"), QVariant(double(2.1)));

            _table->setConfigOption("boolOption", false);
            QCOMPARE(_table->getConfigOption("boolOption").toBool(), false);
        }

    private:
        ConfigOptionsTable* _table;
};


//QTEST_MAIN(oskar::QTestConfigOptionsTable)

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QTestConfigOptionsTable test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}


#endif // QTEST_CONFIG_OPTIONS_TABLE_H
