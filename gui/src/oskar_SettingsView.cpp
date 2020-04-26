/*
 * Copyright (c) 2012-2020, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gui/oskar_SettingsView.h"
#include "gui/oskar_SettingsModel.h"
#include <QApplication>
#include <QMouseEvent>
#include <QMessageBox>
#include <QSettings>
#include <QScrollBar>

namespace oskar {

SettingsView::SettingsView(QWidget* parent) : QTreeView(parent)
{
    connect(this, SIGNAL(expanded(const QModelIndex&)),
            this, SLOT(resizeAfterExpand(const QModelIndex&)));
    connect(this, SIGNAL(collapsed(const QModelIndex&)),
            this, SLOT(updateAfterCollapsed(const QModelIndex&)));
    connect(qApp, SIGNAL(focusChanged(QWidget*, QWidget*)),
            this, SLOT(focusChanged(QWidget*, QWidget*)));
    setAlternatingRowColors(true);
    setUniformRowHeights(true);
}

void SettingsView::displayLabels()
{
    if (!model()) return;
    model()->setData(QModelIndex(), false, SettingsModel::DisplayKeysRole);
}

void SettingsView::displayKeys()
{
    if (!model()) return;
    model()->setData(QModelIndex(), true, SettingsModel::DisplayKeysRole);
}

void SettingsView::restoreExpanded(const QString& app)
{
    if (!model()) return;
    QSettings s;
    QStringList expanded =
            s.value("settings_view/expanded_items/" + app).toStringList();
    saveRestoreExpanded(QModelIndex(), expanded, 1);
}

void SettingsView::restorePosition()
{
    QSettings settings;
    QScrollBar* verticalScroll = verticalScrollBar();
    verticalScroll->setValue(settings.value("settings_view/position").toInt());
}

void SettingsView::saveExpanded(const QString& app)
{
    if (!model()) return;
    QSettings s;
    QStringList expanded;
    saveRestoreExpanded(QModelIndex(), expanded, 0);
    s.setValue("settings_view/expanded_items/" + app, expanded);
}

void SettingsView::savePosition()
{
    QSettings settings;
    settings.setValue("settings_view/position", verticalScrollBar()->value());
}

void SettingsView::showFirstLevel()
{
    expandToDepth(0);
    resizeColumnToContents(0);
    update();
}

void SettingsView::expandSettingsTree()
{
    expandAll();
    resizeColumnToContents(0);
    update();
}

void SettingsView::resizeAfterExpand(const QModelIndex& /*index*/)
{
    resizeColumnToContents(0);
    update();
}


void SettingsView::updateAfterCollapsed(const QModelIndex& /*index*/)
{
    update();
}

void SettingsView::focusChanged(QWidget* old, QWidget* now)
{
    if (!old && now && model())
    {
        // OSKAR has gained focus.
        // Check if the settings file has been modified more recently than the
        // last known modification date.
        model()->setData(QModelIndex(), QVariant(),
                SettingsModel::CheckExternalChangesRole);
    }
}

void SettingsView::fileReloaded()
{
    QMessageBox msgBox(this);
    msgBox.setWindowTitle(parentWidget()->windowTitle());
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText("The settings file was updated by another application.");
    msgBox.setInformativeText("It has now been re-loaded.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}

void SettingsView::mouseDoubleClickEvent(QMouseEvent* event)
{
    QModelIndex index = indexAt(event->pos());
    QModelIndex sibling = index.sibling(index.row(), 1);
    if (model()->flags(sibling) & Qt::ItemIsEditable)
    {
        setCurrentIndex(sibling);
        edit(sibling, QAbstractItemView::DoubleClicked, event);
    }
    else
    {
        QTreeView::mouseDoubleClickEvent(event);
    }
}

void SettingsView::saveRestoreExpanded(const QModelIndex& parent,
        QStringList& list, int restore)
{
    if (!model()) return;
    for (int i = 0; i < model()->rowCount(parent); ++i)
    {
        QModelIndex idx = model()->index(i, 0, parent);
        QString key = idx.data(SettingsModel::KeyRole).toString();
        if (restore)
        {
            if (list.contains(key)) expand(idx);
        }
        else
        {
            if (isExpanded(idx)) list.append(key);
        }

        // Recursion.
        if (model()->rowCount(idx) > 0)
            saveRestoreExpanded(idx, list, restore);
    }
}

} // namespace oskar
