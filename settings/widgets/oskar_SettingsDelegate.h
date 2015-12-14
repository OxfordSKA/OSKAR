/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#ifndef OSKAR_SETTINGS_DELEGATE_H_
#define OSKAR_SETTINGS_DELEGATE_H_

/**
 * @file oskar_SettingsDelegate.h
 */

#include <QtGui/QStyledItemDelegate>

class QModelIndex;
class QWidget;
class oskar_SettingsItem;
class oskar_SettingsModel;

class oskar_SettingsDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    oskar_SettingsDelegate(QWidget* view, QObject* parent = 0);

    /**
     * @details
     * Returns the widget used to edit the item specified by index for editing.
     */
    QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option,
            const QModelIndex& index) const;

    /**
     * @details
     * Used to display the file dialog.
     */
    bool editorEvent(QEvent* event, QAbstractItemModel* model,
            const QStyleOptionViewItem& option, const QModelIndex& index);

    /**
     * @details
     * Sets the data to be displayed and edited by the editor.
     */
    void setEditorData(QWidget* editor, const QModelIndex& index) const;

    /**
     * @details
     * Gets data from the editor widget and stores it in the model.
     */
    void setModelData(QWidget* editor, QAbstractItemModel* model,
            const QModelIndex& index) const;

    /**
     * @details
     * Updates the editor for the item specified by index.
     */
    void updateEditorGeometry(QWidget* editor,
            const QStyleOptionViewItem& option, const QModelIndex& index) const;

private slots:
    void commitAndCloseEditor(int /*index*/);

private:
    QWidget* view_;
};

#endif /* OSKAR_SETTINGS_DELEGATE_H_ */
