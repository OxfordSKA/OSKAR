/*
 * Copyright (c) 2011, The University of Oxford
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

#include "widgets/oskar_SettingsDelegate.h"
#include "widgets/oskar_SettingsItem.h"
#include "widgets/oskar_SettingsModel.h"
#include <QtCore/QEvent>
#include <QtGui/QMenu>
#include <QtGui/QMouseEvent>
#include <QtGui/QFileDialog>
#include <QtGui/QLineEdit>
#include <QtGui/QSpinBox>
#include <QtGui/QDateTimeEdit>
#include <QtGui/QDoubleSpinBox>
#include <cstdio>
#include <climits>
#include <cfloat>

oskar_SettingsDelegate::oskar_SettingsDelegate(QObject* parent)
: QStyledItemDelegate(parent)
{
}

QWidget* oskar_SettingsDelegate::createEditor(QWidget* parent,
    const QStyleOptionViewItem& /*option*/, const QModelIndex& index) const
{
    // Get the setting type.
    int type = ((const oskar_SettingsModel*)index.model())->itemType(index);

    // Create the appropriate editor.
    QWidget* editor = 0;
    switch (type)
    {
        case oskar_SettingsItem::INT:
        {
            // Spin box editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-INT_MAX, INT_MAX);
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::DOUBLE:
        {
            // Double spin box editors.
            QDoubleSpinBox* spinner = new QDoubleSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-DBL_MAX, DBL_MAX);
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Double spin box editors.
            QDateTimeEdit* spinner = new QDateTimeEdit(parent);
            spinner->setFrame(false);
            spinner->setDisplayFormat("dd-MM-yyyy hh:mm:ss.zzz");
            spinner->setCalendarPopup(true);
            editor = spinner;
            break;
        }
        default:
        {
            // Line editors.
            QLineEdit* line = new QLineEdit(parent);
            line->setFrame(false);
            editor = line;
        }
    }

    editor->setAutoFillBackground(true);
    return editor;
}

bool oskar_SettingsDelegate::editorEvent(QEvent* event,
        QAbstractItemModel* model, const QStyleOptionViewItem& option,
        const QModelIndex& index)
{
    // Check for mouse double-click events.
    if (event->type() == QEvent::MouseButtonDblClick)
    {
        // Get the setting type.
        int type = ((const oskar_SettingsModel*)index.model())->itemType(index);

        // Get a pointer to the item.
        oskar_SettingsItem* item = ((oskar_SettingsModel*)model)->getItem(index);
        QWidget* parent = 0;

        if (type == oskar_SettingsItem::INPUT_FILE_NAME)
        {
            QString dir = item->data().toString();
            QString value = QFileDialog::getOpenFileName(parent,
                    "Input file name", dir);
            if (!value.isNull())
            {
                value = QDir::current().relativeFilePath(value);
                model->setData(index, value, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        else if (type == oskar_SettingsItem::INPUT_DIR_NAME)
        {
            QString dir = item->data().toString();
            QString value = QFileDialog::getExistingDirectory(parent,
                    "Directory", dir);
            if (!value.isNull())
            {
                value = QDir::current().relativeFilePath(value);
                model->setData(index, value, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        else if (type == oskar_SettingsItem::OUTPUT_FILE_NAME)
        {
            QString dir = item->data().toString();
            QString value = QFileDialog::getSaveFileName(parent,
                    "Output file name", dir);
            if (!value.isNull())
            {
                value = QDir::current().relativeFilePath(value);
                model->setData(index, value, Qt::EditRole);
            }
            event->accept();
            return true;
        }
    }

    // Check for mouse right-click events.
    else if (event->type() == QEvent::MouseButtonRelease)
    {
        QMouseEvent* mouseEvent = (QMouseEvent*)event;
        if (mouseEvent->button() == Qt::RightButton && index.column() == 1)
        {
            QMenu menu;
            menu.addAction("Clear");
            if (menu.exec(mouseEvent->globalPos()))
                model->setData(index, "", Qt::EditRole);
            event->accept();
            return true;
        }
    }

    return QStyledItemDelegate::editorEvent(event, model, option, index);
}

void oskar_SettingsDelegate::setEditorData(QWidget* editor,
        const QModelIndex& index) const
{
    // Get the setting type.
    int type = ((const oskar_SettingsModel*)index.model())->itemType(index);

    // Set the editor data.
    QVariant value = index.model()->data(index, Qt::EditRole);
    switch (type)
    {
        case oskar_SettingsItem::INT:
        {
            // Spin box editors.
            static_cast<QSpinBox*>(editor)->setValue(value.toInt());
            break;
        }
        case oskar_SettingsItem::DOUBLE:
        {
            // Double spin box editors.
            static_cast<QDoubleSpinBox*>(editor)->setValue(value.toDouble());
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Date and time editors.
            QDateTime date = QDateTime::fromString(value.toString(),
                    "dd-MM-yyyy h:m:s.z");
            static_cast<QDateTimeEdit*>(editor)->setDateTime(date);
            break;
        }
        default:
        {
            // Line editors.
            static_cast<QLineEdit*>(editor)->setText(value.toString());
        }
    }
}

void oskar_SettingsDelegate::setModelData(QWidget* editor,
        QAbstractItemModel* model, const QModelIndex& index) const
{
    // Get the setting type.
    int type = ((const oskar_SettingsModel*)index.model())->itemType(index);

    // Get the editor data.
    QVariant value;
    switch (type)
    {
        case oskar_SettingsItem::INT:
        {
            // Spin box editors.
            value = static_cast<QSpinBox*>(editor)->value();
            break;
        }
        case oskar_SettingsItem::DOUBLE:
        {
            // Double spin box editors.
            value = static_cast<QDoubleSpinBox*>(editor)->value();
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Date and time editors.
            QDateTime date = static_cast<QDateTimeEdit*>(editor)->dateTime();
            value = date.toString("dd-MM-yyyy hh:mm:ss.zzz");
            break;
        }
        default:
        {
            value = static_cast<QLineEdit*>(editor)->text();
        }
    }

    model->setData(index, value, Qt::EditRole);
}

void oskar_SettingsDelegate::updateEditorGeometry(QWidget* editor,
        const QStyleOptionViewItem& option,
        const QModelIndex& /*index*/) const
{
    editor->setGeometry(option.rect);
}
