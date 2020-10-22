/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MAIN_WINDOW_H_
#define OSKAR_MAIN_WINDOW_H_

#include <QMainWindow>
#include <QString>
#include <QHash>

class QComboBox;
class QLineEdit;
class QWidget;
class QNetworkAccessManager;
class QNetworkReply;

namespace oskar {

class SettingsModel;
class SettingsView;
class SettingsTree;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = 0);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event);

public slots:
    void clear();
    void open(QString filename = QString());
    void save(QString filename = QString());

private slots:
    void about();
    void appChanged(QString text = QString());
    void setAppDir(bool hint = false);
    void changeDir();
    void checkForUpdate();
    void systemInfo();
    void helpDoc();
    void processNetworkReply(QNetworkReply*);
    void runButton();

private:
    void notFound(const QString& app_name);
    void run(QString app, const QStringList& args = QStringList(),
            bool allow_auto_close = true);
    void setApp(const QString& app, bool refresh = true);

private:
    bool checked_update_;
    QString title_;
    QLineEdit* current_dir_label_;
    QLineEdit* filter_;
    QComboBox* selector_;
    SettingsTree* settings_;
    SettingsView* view_;
    SettingsModel* model_;
    QHash<QString, QString> files_;
    QString current_dir_;
    QString current_app_;
    QString current_app_version_;
    QString app_dir_;

    QNetworkAccessManager* net_;
    QString version_url_;
};

} /* namespace oskar */

#endif /* include guard */
