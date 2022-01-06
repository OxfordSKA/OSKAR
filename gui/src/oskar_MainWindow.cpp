/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "gui/oskar_About.h"
#include "gui/oskar_AppLocation.h"
#include "gui/oskar_Docs.h"
#include "gui/oskar_MainWindow.h"
#include "gui/oskar_RunDialog.h"
#include "gui/oskar_SettingsDelegate.h"
#include "gui/oskar_SettingsModel.h"
#include "gui/oskar_SettingsView.h"
#include "settings/oskar_SettingsDeclareXml.h"
#include "settings/oskar_SettingsFileHandlerIni.h"
#include "settings/oskar_SettingsTree.h"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QSettings>
#include <QTimer>
#include <QVBoxLayout>

#include <exception>

namespace oskar {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent),
        checked_update_(false)
{
    // Set the window title.
    title_ = QString("OSKAR");
    setWindowTitle(title_);
    setWindowIcon(QIcon(":/icons/oskar.ico"));

    // Create the central widget and main layout.
    QWidget* widget = new QWidget(this);
    setCentralWidget(widget);
    QVBoxLayout* v_layout = new QVBoxLayout(widget);
    QGridLayout* gridLayout = new QGridLayout;

    // Create the settings model.
    settings_ = new SettingsTree;
    model_ = new SettingsModel(settings_, this);
    QSortFilterProxyModel* modelProxy_ = new SettingsModelFilter(this);
    modelProxy_->setSourceModel(model_);

    // Create application selector, current directory view and search box.
    QLabel* label1 = new QLabel("Application", widget);
    selector_ = new QComboBox(widget);
    selector_->addItem("oskar_sim_interferometer");
    selector_->addItem("oskar_sim_beam_pattern");
    selector_->addItem("oskar_imager");
    selector_->addItem("oskar_fit_element_data");
    QSizePolicy policy = selector_->sizePolicy();
    policy.setHorizontalStretch(2);
    selector_->setSizePolicy(policy);
    QPushButton* runner = new QPushButton(widget);
    runner->setText("Run");
    runner->setToolTip("Run the selected application "
            "with the current settings.");
    QLabel* label2 = new QLabel("Current Dir", widget);
    current_dir_label_ = new QLineEdit(widget);
    current_dir_label_->setReadOnly(true);
    current_dir_label_->setToolTip("The current working directory. "
            "All paths specified in the settings below must be relative "
            "to this location.");
    QPushButton* change_dir = new QPushButton(widget);
    change_dir->setText("Change...");
    change_dir->setToolTip("Change the current working directory.");
    filter_ = new QLineEdit(widget);
    filter_->setPlaceholderText("Search Filter");
    gridLayout->addWidget(label1, 0, 0);
    gridLayout->addWidget(selector_, 0, 1);
    gridLayout->addWidget(runner, 0, 2);
    gridLayout->addWidget(label2, 1, 0);
    gridLayout->addWidget(current_dir_label_, 1, 1);
    gridLayout->addWidget(change_dir, 1, 2);
    gridLayout->addWidget(filter_, 2, 0, 1, 3);
    connect(runner, SIGNAL(clicked()), SLOT(runButton()));
    connect(change_dir, SIGNAL(clicked()), SLOT(changeDir()));
    connect(selector_, SIGNAL(currentIndexChanged(QString)),
            SLOT(appChanged(QString)));
    connect(filter_, SIGNAL(textChanged(QString)),
            modelProxy_, SLOT(setFilterRegExp(QString)));

    // Create and set up the settings view.
    view_ = new SettingsView(widget);
    SettingsDelegate* delegate = new SettingsDelegate(view_, widget);
    view_->setModel(modelProxy_);
    view_->setItemDelegate(delegate);
    view_->resizeColumnToContents(0);
    view_->setUniformRowHeights(true);
    v_layout->addLayout(gridLayout);
    v_layout->addWidget(view_);
    connect(model_, SIGNAL(fileReloaded()), view_, SLOT(fileReloaded()));

    // Create and set up the actions.
    QAction* actOpen = new QAction("&Open...", this);
    QAction* actSaveAs = new QAction("&Save As...", this);
    QAction* actClear = new QAction("&Clear (Unload)", this);
    QAction* actExit = new QAction("E&xit", this);
    QAction* actAppDir = new QAction("&App Location...", this);
    QAction* actShowFirstLevel = new QAction("Show &First Level", this);
    QAction* actExpandAll = new QAction("&Expand All", this);
    QAction* actCollapseAll = new QAction("&Collapse All", this);
    QAction* actDisplayKeys = new QAction("Display &Keys", this);
    QAction* actDisplayLabels = new QAction("Display &Labels", this);
    QAction* actHelpDoc = new QAction("Documentation...", this);
    QAction* actAboutQt = new QAction("&Qt Info...", this);
    QAction* actSystemInfo = new QAction("System Info...", this);
    QAction* actAbout = new QAction("&About OSKAR...", this);
    connect(actOpen, SIGNAL(triggered()), this, SLOT(open()));
    connect(actSaveAs, SIGNAL(triggered()), this, SLOT(save()));
    connect(actClear, SIGNAL(triggered()), this, SLOT(clear()));
    connect(actExit, SIGNAL(triggered()), this, SLOT(close()));
    connect(actAppDir, SIGNAL(triggered()), this, SLOT(setAppDir()));
    connect(actShowFirstLevel, SIGNAL(triggered()),
            view_, SLOT(showFirstLevel()));
    connect(actDisplayKeys, SIGNAL(triggered()), view_, SLOT(displayKeys()));
    connect(actDisplayLabels, SIGNAL(triggered()),
            view_, SLOT(displayLabels()));
    connect(actExpandAll, SIGNAL(triggered()), view_, SLOT(expandSettingsTree()));
    connect(actCollapseAll, SIGNAL(triggered()), view_, SLOT(collapseAll()));
    connect(actAbout, SIGNAL(triggered()), this, SLOT(about()));
    connect(actAboutQt, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
    connect(actSystemInfo, SIGNAL(triggered()), this, SLOT(systemInfo()));
    connect(actHelpDoc, SIGNAL(triggered()), this, SLOT(helpDoc()));

    // Set up keyboard shortcuts.
    actClear->setShortcut(QKeySequence::Close);
    actOpen->setShortcut(QKeySequence::Open);
    actExit->setShortcut(QKeySequence::Quit);

    // Create the menus.
    setUnifiedTitleAndToolBarOnMac(true);
    QMenu* menuFile = menuBar()->addMenu("&File");
    QMenu* menuView = menuBar()->addMenu("&View");
    QMenu* menuHelp = menuBar()->addMenu("&Help");
    menuFile->addAction(actOpen);
    menuFile->addAction(actSaveAs);
    menuFile->addSeparator();
    menuFile->addAction(actClear);
    menuFile->addSeparator();
    menuFile->addAction(actAppDir);
    menuFile->addSeparator();
    menuFile->addAction(actExit);
    menuView->addAction(actShowFirstLevel);
    menuView->addAction(actExpandAll);
    menuView->addAction(actCollapseAll);
    menuView->addSeparator();
    menuView->addAction(actDisplayKeys);
    menuView->addAction(actDisplayLabels);
    menuHelp->addAction(actHelpDoc);
    menuHelp->addSeparator();
    menuHelp->addAction(actAboutQt);
    menuHelp->addAction(actSystemInfo);
    menuHelp->addAction(actAbout);

    // Create the network access manager to check the current version.
    version_url_ = "/current_oskar_version.txt";
    net_ = new QNetworkAccessManager(this);
    connect(net_, SIGNAL(finished(QNetworkReply*)),
                this, SLOT(processNetworkReply(QNetworkReply*)));

    // Load the settings.
    QSettings settings;
    restoreGeometry(settings.value("main_window/geometry").toByteArray());
    restoreState(settings.value("main_window/state").toByteArray());
    app_dir_ = settings.value("app_dir", "").toString();
#ifndef Q_OS_WIN
    if (app_dir_.isEmpty())
        app_dir_ = QApplication::applicationDirPath();
#endif
    current_app_ = settings.value("current_app").toString();
    current_dir_ = settings.value("current_dir").toString();
    QDir d(current_dir_);
    if (QCoreApplication::arguments().length() > 1)
        current_dir_ = QDir::toNativeSeparators(QDir::currentPath());
    else if (current_dir_.isEmpty() || !d.exists())
    {
        current_dir_ = QDir::toNativeSeparators(QDir::homePath());
        QDir::setCurrent(current_dir_);
    }
    else
        QDir::setCurrent(current_dir_);
    current_dir_label_->setText(current_dir_);
    if (current_app_.isEmpty())
        current_app_ = selector_->itemText(0);
    settings.beginGroup("files");
    QStringList keys = settings.childKeys();
    Q_FOREACH(QString key, keys)
    {
        files_[key] = settings.value(key).toString();
    }
    settings.endGroup();

    // Set the focus policy.
    setFocusPolicy(Qt::StrongFocus);

    // Set the selected application.
    setApp(current_app_);

    // Restore the scroll bar position.
    // A single-shot timer is used to do this after the main event loop starts.
    QTimer::singleShot(0, view_, SLOT(restorePosition()));
}

MainWindow::~MainWindow()
{
    delete settings_;
}

void MainWindow::clear()
{
    if (files_[current_app_].isEmpty() && settings_->is_modified())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("This action will discard unsaved modifications.");
        msgBox.setInformativeText("Do you want to proceed?");
        msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
        if (msgBox.exec() == QMessageBox::Cancel)
            return;
    }
    files_[current_app_] = QString();
    if (settings_->file_handler())
        settings_->file_handler()->set_file_name("");
    settings_->set_defaults();
    model_->refresh();
    setWindowTitle(title_);
}

void MainWindow::open(QString filename)
{
    char* temp_string;
    if (files_[current_app_].isEmpty() && settings_->is_modified())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("Opening a new file will discard unsaved modifications.");
        msgBox.setInformativeText("Do you want to proceed?");
        msgBox.setStandardButtons(QMessageBox::Open | QMessageBox::Cancel);
        if (msgBox.exec() == QMessageBox::Cancel)
            return;
    }

    // Check if the supplied filename is empty, and prompt to open file if so.
    if (filename.isEmpty())
        filename = QFileDialog::getOpenFileName(this, "Open Settings",
                files_[current_app_]);
    if (!settings_->file_handler() || filename.isEmpty()) return;

    // Check the file exists.
    if (!QFile::exists(filename))
    {
        QMessageBox::critical(this, title_,
                QString("Settings file '%1' does not exist.").arg(filename));
        return;
    }
    else
    {
        QFileInfo fi(filename);
        filename = QDir::toNativeSeparators(fi.canonicalFilePath());
    }

    // Open the file by selecting the application.
    temp_string = settings_->file_handler()->read(
            filename.toLatin1().constData(), "app");
    QString app = temp_string ? QString(temp_string) : QString();
    free(temp_string);
    int app_index = -1;
    if (!app.isEmpty())
        app_index = selector_->findText(app);
    if (app_index >= 0)
    {
        files_[app] = filename;
        selector_->blockSignals(true);
        selector_->setCurrentIndex(-1);
        selector_->blockSignals(false);
        selector_->setCurrentIndex(app_index);
    }
    else
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.setText("Unknown OSKAR application for this settings file.");
        msgBox.setInformativeText("Make sure the 'app=' key is set correctly "
                "in the file.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        return;
    }

    // Check the version of the application that created the file.
    temp_string = settings_->file_handler()->read(
            filename.toLatin1().constData(), "version");
    QString ver_file = temp_string ? QString(temp_string) : QString();
    free(temp_string);
    QStringList vers_file = ver_file.split('.');
    QStringList vers_app = current_app_version_.split('.');
    int major_file = (vers_file.size() > 0) ? vers_file[0].toInt() : 0;
    int minor_file = (vers_file.size() > 1) ? vers_file[1].toInt() : 0;
    int major_app = (vers_app.size() > 0) ? vers_app[0].toInt() : 0;
    int minor_app = (vers_app.size() > 1) ? vers_app[1].toInt() : 0;
    if (ver_file.isEmpty())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("This file was created by an unknown version of OSKAR.");
        msgBox.setInformativeText("Please check all settings "
                "carefully, as the keys may have changed.");
        msgBox.exec();
    }
    else if (major_file != major_app || minor_file != minor_app)
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText(QString("This file was created by OSKAR %1.").arg(ver_file));
        msgBox.setInformativeText("Please check all settings "
                "carefully, as the keys may have changed since that version.");
        msgBox.exec();
    }
}

void MainWindow::save(QString filename)
{
    // Check if the supplied filename is empty, and prompt to save file if so.
    if (filename.isEmpty())
        filename = QFileDialog::getSaveFileName(this, "Save Settings",
                files_[current_app_]);
    if (filename.isEmpty()) return;

    // Remove any existing file with this name.
    if (QFile::exists(filename))
    {
        if (!QFile::remove(filename))
        {
            QMessageBox::critical(this, title_,
                    QString("Could not overwrite file at %1").arg(filename));
            return;
        }
    }

    // Try to open the file for writing.
    filename = QDir::toNativeSeparators(filename);
    QFile file(filename);
    bool result = file.open(QFile::ReadWrite);
    file.close();
    if (!result)
    {
        QMessageBox::critical(this, title_,
                QString("Could not save file at %1").arg(filename));
        return;
    }

    files_[current_app_] = filename;
    model_->save_settings_file(filename);
    setWindowTitle(title_ + " [" + filename + "]");
}

// =========================================================  Protected methods.

void MainWindow::closeEvent(QCloseEvent* event)
{
    // Save the state of the tree view.
    view_->saveExpanded(current_app_);
    view_->savePosition();

    QSettings settings;
    settings.setValue("main_window/geometry", saveGeometry());
    settings.setValue("main_window/state", saveState());
    settings.setValue("app_dir", app_dir_);
    settings.setValue("current_app", current_app_);
    settings.setValue("current_dir", current_dir_);
    settings.beginGroup("files");
    QHash<QString, QString>::const_iterator i = files_.constBegin();
    while (i != files_.constEnd())
    {
         settings.setValue(i.key(), i.value());
         ++i;
     }
    settings.endGroup();

    if (files_[current_app_].isEmpty() && settings_->is_modified())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("This action will discard unsaved modifications.");
        msgBox.setInformativeText("Do you want to proceed?");
        msgBox.setStandardButtons(QMessageBox::Discard | QMessageBox::Cancel |
                QMessageBox::Save);
        msgBox.setDefaultButton(QMessageBox::Discard);
        int ret = msgBox.exec();
        if (ret == QMessageBox::Discard)
            event->accept();
        else if (ret == QMessageBox::Save)
        {
            save();

            // Check if the save failed for any reason.
            if (files_[current_app_].isEmpty())
                event->ignore();
            else
                event->accept();
        }
        else
            event->ignore();
    }
}

// =========================================================  Private slots.

void MainWindow::about()
{
    About dialog(current_app_, current_app_version_, this);
    dialog.exec();
}

void MainWindow::appChanged(QString text)
{
    // Return if no application was selected.
    if (text.isEmpty()) return;

    // Check if this is a potentially destructive action.
    if (files_[current_app_].isEmpty() && settings_->is_modified())
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setText("This action will discard unsaved modifications.");
        msgBox.setInformativeText("Do you want to proceed?");
        msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
        if (msgBox.exec() == QMessageBox::Cancel)
        {
            setApp(current_app_, false);
            return;
        }
    }

    // Clear the search filter text.
    filter_->clear();

    // Check that the selected application exists.
    QString app = text;
    if (!app_dir_.isEmpty())
    {
        QDir dir(app_dir_);
        app = QDir::cleanPath(dir.filePath(app));
    }
    QProcess process;
    process.start(app, QStringList() << "--version");
    if (!process.waitForStarted())
    {
        notFound(app);
        return;
    }
    process.waitForFinished();
    current_app_version_ = process.readAllStandardOutput().trimmed();

    // Save the current settings view and set the new current application.
    if (isVisible())
        view_->saveExpanded(current_app_);
    current_app_ = text;

    // Get settings for selected application.
    process.start(app, QStringList() << "--settings");
    process.waitForFinished();
    QByteArray settings = process.readAllStandardOutput().trimmed();
    if (settings.isEmpty() || !settings.startsWith('<'))
    {
        notFound(app);
        return;
    }

    // Set the file handler for the application.
    SettingsFileHandlerIni* handler = new SettingsFileHandlerIni(
            current_app_.toLatin1().constData(),
            current_app_version_.toLatin1().constData());
    settings_->set_file_handler(handler);

    // Declare application settings, load settings file, and reset the model.
    model_->beginReset();
    try
    {
        oskar_settings_declare_xml(settings_, settings.constData());
    }
    catch (std::exception& e)
    {
        model_->endReset();
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.setText(QString("Caught exception '%1'").arg(e.what()));
        msgBox.setDetailedText(
                QString("Tried to use settings:\n") + settings.constData());
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
        return;
    }
    QString filename = files_[current_app_];
    title_ = QString("OSKAR (%1)").arg(current_app_version_);
    if (QFile::exists(filename))
        setWindowTitle(title_ + " [" + filename + "]");
    else
    {
        setWindowTitle(title_);
        files_[current_app_] = QString();
    }
    model_->load_settings_file(files_[current_app_]);
    model_->endReset();

    // Restore the view for this application.
    view_->restoreExpanded(current_app_);
    view_->resizeColumnToContents(0);

    // Check for updates.
    //if (!checked_update_) checkForUpdate();
}

void MainWindow::setAppDir(bool hint)
{
    AppLocation dialog(this);
    dialog.setDir(hint ? QApplication::applicationDirPath() : app_dir_);
    if (dialog.exec() == QDialog::Accepted)
    {
        app_dir_ = dialog.dir();
        setApp(current_app_);
    }
}

void MainWindow::changeDir()
{
    QString dir = QFileDialog::getExistingDirectory(this,
            "Set work directory", QString(), QFileDialog::ShowDirsOnly);
    if (!dir.isEmpty())
    {
        current_dir_ = QDir::toNativeSeparators(dir);
        current_dir_label_->setText(current_dir_);
        QDir::setCurrent(current_dir_);
    }
}

void MainWindow::checkForUpdate()
{
    QNetworkRequest request;
    request.setUrl(QUrl(version_url_));
    request.setRawHeader("User-Agent", "OSKAR");
    net_->get(request);
    checked_update_ = true;
}

void MainWindow::systemInfo()
{
    run("oskar_system_info", QStringList(), false);
}

void MainWindow::helpDoc()
{
    Docs dialog(this);
    dialog.exec();
}

void MainWindow::processNetworkReply(QNetworkReply* reply)
{
    QString currentVerString;
    // Obtain the current version of OSKAR from the server.
    if (reply->request().url().toString() == version_url_)
    {
        QByteArray data = reply->readLine();
        if (data.endsWith('\n')) data.chop(1);
        if (data.isEmpty() || reply->error() != QNetworkReply::NoError)
            return;
        currentVerString = QString(data);
    }

    // Mark the reply for deletion.
    reply->deleteLater();

    // Split version strings into components.
    QRegExp rx("\\.|-");
    QStringList currentVer = currentVerString.split(rx);
    QStringList thisVer = QString(current_app_version_).split(rx);

    // Version strings should be of the format major.minor.patch-flag
    // where flag is optional. Therefore they will be of length 3 or 4 components.
    bool validVer = currentVer.size() == 3 || currentVer.size() == 4;
    validVer = validVer && (thisVer.size() == 3 || thisVer.size() == 4);
    if (!validVer) return;

    // If the server (current) version is flagged (i.e. has a hyphen tag after
    // its version number) - *NEVER* prompt for an update.
    // This avoids asking people to download release candidates, for example.
    if (currentVer.size() == 4)
        return;

    // If the current version is newer than the version of this code
    // notify of the update.
    bool isNewMajorVer   = currentVer[0].toInt() >  thisVer[0].toInt();
    bool isEqualMajorVer = currentVer[0].toInt() == thisVer[0].toInt();
    bool isNewMinorVer   = currentVer[1].toInt() >  thisVer[1].toInt();
    bool isEqualMinorVer = currentVer[1].toInt() == thisVer[1].toInt();
    bool isNewPatchVer   = currentVer[2].toInt() >  thisVer[2].toInt();

    // Conditions to prompt for update:
    // 1) Major version is out of date.
    // 2) Major version not changed but minor version out of date.
    // 3) Major and minor versions not changed but patch out of date.
    if (isNewMajorVer || (isEqualMajorVer && isNewMinorVer) ||
            (isEqualMajorVer && isEqualMinorVer && isNewPatchVer))
    {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(title_);
        msgBox.setIcon(QMessageBox::Information);
        msgBox.setTextFormat(Qt::RichText);
        msgBox.setText(QString("A newer version of OSKAR (%1) is "
                "available.").arg(currentVerString));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
    }
}

void MainWindow::runButton()
{
    // Save settings if they are not already saved.
    if (files_[current_app_].isEmpty())
    {
        save(QString());
        if (files_[current_app_].isEmpty())
            return;
    }

    // Set application working directory to the file path, if not already set.
    QFileInfo fi(QDir::currentPath());
    if (fi.isRoot())
    {
        QFileInfo fi(files_[current_app_]);
        current_dir_ = QDir::toNativeSeparators(fi.path());
        current_dir_label_->setText(current_dir_);
        QDir::setCurrent(current_dir_);
    }

    // Run with relative path of settings file.
    QDir d(current_dir_);
    QString relative_path = d.relativeFilePath(files_[current_app_]);
    run(current_app_, QStringList() << relative_path);
}

// =========================================================  Private methods.

void MainWindow::notFound(const QString& app_name)
{
    QMessageBox msgBox(this);
    msgBox.setWindowTitle(title_);
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.setText(
            QString("Application '%1' or its settings could not be found.")
            .arg(app_name));
    msgBox.setInformativeText("Please edit the location and try again.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
    setApp(current_app_, false);
    setAppDir(true);
}

void MainWindow::run(QString app, const QStringList& args,
        bool allow_auto_close)
{
    if (!app_dir_.isEmpty())
    {
        QDir dir(app_dir_);
        app = QDir::cleanPath(dir.filePath(app));
    }
    QProcess process;
    process.start(app, QStringList() << "--version");
    if (!process.waitForStarted())
    {
        notFound(app);
        return;
    }
    process.waitForFinished();
    RunDialog dialog(app, this);
    dialog.setAllowAutoClose(allow_auto_close);
    dialog.start(args);
    dialog.exec();
}

void MainWindow::setApp(const QString& app, bool refresh)
{
    int app_index = selector_->findText(app);
    if (app_index >= 0)
    {
        if (!refresh)
        {
            selector_->blockSignals(true);
            selector_->setCurrentIndex(app_index);
            selector_->blockSignals(false);
        }
        else
        {
            selector_->blockSignals(true);
            selector_->setCurrentIndex(-1);
            selector_->blockSignals(false);
            selector_->setCurrentIndex(app_index);
        }
    }
}

} /* namespace oskar */
