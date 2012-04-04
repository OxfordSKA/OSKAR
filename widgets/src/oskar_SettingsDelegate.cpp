/*
 * Copyright (c) 2012, The University of Oxford
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

#include "widgets/oskar_DoubleSpinBox.h"
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
#include <QtGui/QTimeEdit>
#include <QtGui/QComboBox>

#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QFormLayout>
#include <cstdio>
#include <climits>
#include <cfloat>

oskar_SettingsDelegate::oskar_SettingsDelegate(QWidget* view, QObject* parent)
: QStyledItemDelegate(parent)
{
    view_ = view;
}

QWidget* oskar_SettingsDelegate::createEditor(QWidget* parent,
    const QStyleOptionViewItem& /*option*/, const QModelIndex& index) const
{
    // Get the setting type.
    int type = index.model()->data(index, oskar_SettingsModel::TypeRole).toInt();

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
        case oskar_SettingsItem::INT_UNSIGNED:
        {
            // Spin box editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(0, INT_MAX);
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::INT_POSITIVE:
        {
            // Spin box editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(1, INT_MAX);
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::DOUBLE:
        {
            // Double spin box editors.
            oskar_DoubleSpinBox* spinner = new oskar_DoubleSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-DBL_MAX, DBL_MAX);
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Date and time editors.
            QDateTimeEdit* spinner = new QDateTimeEdit(parent);
            spinner->setFrame(false);
            spinner->setDisplayFormat("dd-MM-yyyy hh:mm:ss.zzz");
            spinner->setCalendarPopup(true);
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::TIME:
        {
            // Time editors.
            QTimeEdit* spinner = new QTimeEdit(parent);
            spinner->setFrame(false);
            spinner->setDisplayFormat("hh:mm:ss.zzz");
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::RANDOM_SEED:
        {
            // Random seed editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-1, INT_MAX);
            spinner->setSpecialValueText("time");
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::AXIS_RANGE:
        {
            // Random seed editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-1, INT_MAX);
            spinner->setSpecialValueText("max");
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::OPTIONS:
        {
            // Options list.
            QComboBox* list = new QComboBox(parent);
            QStringList options;
            options = index.model()->data(index,
                    oskar_SettingsModel::OptionsRole).toStringList();
            list->setFrame(false);
            list->addItems(options);
            connect(list, SIGNAL(activated(int)),
                    this, SLOT(commitAndCloseEditor(int)));
            editor = list;
            break;
        }
        default:
        {
            // Line editors.
            QLineEdit* line = new QLineEdit(parent);
            line->setFrame(false);
            editor = line;
            break;
        }
    }

    editor->setAutoFillBackground(true);
    return editor;
}

bool oskar_SettingsDelegate::editorEvent(QEvent* event,
        QAbstractItemModel* mod, const QStyleOptionViewItem& option,
        const QModelIndex& index)
{
    // Check for events only in column 1.
    if (index.column() != 1)
        return QStyledItemDelegate::editorEvent(event, mod, option, index);

    // Get item type and value.
    int type = index.model()->data(index, oskar_SettingsModel::TypeRole).toInt();
    QVariant value = index.model()->data(index, Qt::EditRole);

    // Check for mouse double-click events.
    if (event->type() == QEvent::MouseButtonDblClick)
    {
        if (type == oskar_SettingsItem::TELESCOPE_DIR_NAME)
        {
            QString name = QFileDialog::getExistingDirectory(view_,
                    "Telescope directory", value.toString());
            if (!name.isNull())
            {
                name = QDir::current().relativeFilePath(name);
                mod->setData(index, name, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        else if (type == oskar_SettingsItem::INPUT_FILE_NAME)
        {
            QString name = QFileDialog::getOpenFileName(view_,
                    "Input file name", value.toString());
            if (!name.isNull())
            {
                name = QDir::current().relativeFilePath(name);
                mod->setData(index, name, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        else if (type == oskar_SettingsItem::OUTPUT_FILE_NAME)
        {
            QString name = QFileDialog::getSaveFileName(view_,
                    "Output file name", value.toString());
            if (!name.isNull())
            {
                name = QDir::current().relativeFilePath(name);
                mod->setData(index, name, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        return QStyledItemDelegate::editorEvent(event, mod, option, index);
    }

    // Check for mouse right-click events.
    else if (event->type() == QEvent::MouseButtonRelease)
    {
        QMouseEvent* mouseEvent = (QMouseEvent*)event;
        if (mouseEvent->button() == Qt::RightButton &&
                type != oskar_SettingsItem::LABEL)
        {
            // Get the iteration keys.
            QStringList iterationKeys = mod->data(index,
                    oskar_SettingsModel::IterationKeysRole).toStringList();

            // Set up the context menu.
            QMenu menu;
            QString strClearValue = "Clear Value";
            QString strDisable = "Disable";
            QString strEnable = "Enable";
            QString strClearIteration = "Clear Iteration";
            QString strEditIteration = "Edit Iteration Parameters";
            QString strIterate = QString("Iterate (Dimension %1)...").
                    arg(iterationKeys.size() + 1);
            menu.addAction(strClearValue);
            if (mod->data(index, oskar_SettingsModel::EnabledRole).toBool())
                menu.addAction(strDisable);
            else
                menu.addAction(strEnable);
            if (type == oskar_SettingsItem::INT ||
                    type == oskar_SettingsItem::INT_UNSIGNED ||
                    type == oskar_SettingsItem::INT_POSITIVE ||
                    type == oskar_SettingsItem::DOUBLE)
            {
                QString key = mod->data(index,
                        oskar_SettingsModel::KeyRole).toString();
                menu.addSeparator();
                if (iterationKeys.contains(key))
                {
                    menu.addAction(strEditIteration);
                    menu.addAction(strClearIteration);
                }
                else
                    menu.addAction(strIterate);
            }

            // Display the context menu.
            QAction* action = menu.exec(mouseEvent->globalPos());

            // Check which action was selected.
            if (action)
            {
                if (action->text() == strClearValue)
                    mod->setData(index, QVariant(), Qt::EditRole);
                else if (action->text() == strDisable)
                    mod->setData(index, false, oskar_SettingsModel::EnabledRole);
                else if (action->text() == strEnable)
                    mod->setData(index, true, oskar_SettingsModel::EnabledRole);
                else if (action->text() == strIterate ||
                        action->text() == strEditIteration)
                    setIterations(mod, index);
                else if (action->text() == strClearIteration)
                    mod->setData(index, 0, oskar_SettingsModel::ClearIterationRole);
            }
            event->accept();
            return true;
        }
    }

    return QStyledItemDelegate::editorEvent(event, mod, option, index);
}

void oskar_SettingsDelegate::setEditorData(QWidget* editor,
        const QModelIndex& index) const
{
    // Get the setting type.
    int type = index.model()->data(index, oskar_SettingsModel::TypeRole).toInt();

    // Set the editor data.
    QVariant value = index.model()->data(index, Qt::EditRole);
    switch (type)
    {
        case oskar_SettingsItem::INT:
        case oskar_SettingsItem::INT_UNSIGNED:
        case oskar_SettingsItem::INT_POSITIVE:
        {
            // Spin box editors.
            static_cast<QSpinBox*>(editor)->setValue(value.toInt());
            break;
        }
        case oskar_SettingsItem::DOUBLE:
        {
            // Double spin box editors.
            static_cast<oskar_DoubleSpinBox*>(editor)->setValue(value.toDouble());
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
        case oskar_SettingsItem::TIME:
        {
            // Time editors.
            QTime time = QTime::fromString(value.toString(), "h:m:s.z");
            static_cast<QTimeEdit*>(editor)->setTime(time);
            break;
        }
        case oskar_SettingsItem::RANDOM_SEED:
        {
            // Random seed editors.
            if (value.toString().toUpper() == "TIME" || value.toInt() < 0)
                static_cast<QSpinBox*>(editor)->setValue(-1);
            else
                static_cast<QSpinBox*>(editor)->setValue(value.toInt());
            break;
        }
        case oskar_SettingsItem::AXIS_RANGE:
        {
            // Random seed editors.
            if (value.toString().toUpper() == "MAX" || value.toInt() < 0)
                static_cast<QSpinBox*>(editor)->setValue(-1);
            else
                static_cast<QSpinBox*>(editor)->setValue(value.toInt());
            break;
        }
        case oskar_SettingsItem::OPTIONS:
        {
            // Options list.
            QString str = value.toString();
            int i = static_cast<QComboBox*>(editor)->findText(str,
                    Qt::MatchFixedString);
            if (i < 0) i = 0;
            static_cast<QComboBox*>(editor)->setCurrentIndex(i);
            break;
        }
        default:
        {
            // Line editors.
            static_cast<QLineEdit*>(editor)->setText(value.toString());
            break;
        }
    }
}

void oskar_SettingsDelegate::setModelData(QWidget* editor,
        QAbstractItemModel* model, const QModelIndex& index) const
{
    // Get the setting type.
    int type = index.model()->data(index, oskar_SettingsModel::TypeRole).toInt();

    // Get the editor data.
    QVariant value;
    switch (type)
    {
        case oskar_SettingsItem::INT:
        case oskar_SettingsItem::INT_UNSIGNED:
        case oskar_SettingsItem::INT_POSITIVE:
        {
            // Spin box editors.
            value = static_cast<QSpinBox*>(editor)->value();
            break;
        }
        case oskar_SettingsItem::DOUBLE:
        {
            // Double spin box editors.
            value = static_cast<oskar_DoubleSpinBox*>(editor)->value();
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Date and time editors.
            QDateTime date = static_cast<QDateTimeEdit*>(editor)->dateTime();
            value = date.toString("dd-MM-yyyy hh:mm:ss.zzz");
            break;
        }
        case oskar_SettingsItem::TIME:
        {
            // Time editors.
            QTime date = static_cast<QTimeEdit*>(editor)->time();
            value = date.toString("hh:mm:ss.zzz");
            break;
        }
        case oskar_SettingsItem::RANDOM_SEED:
        {
            // Random seed editors.
            int val = static_cast<QSpinBox*>(editor)->value();
            if (val < 0) value = "time"; else value = val;
            break;
        }
        case oskar_SettingsItem::AXIS_RANGE:
        {
            // Random seed editors.
            int val = static_cast<QSpinBox*>(editor)->value();
            if (val < 0) value = "max"; else value = val;
            break;
        }
        case oskar_SettingsItem::OPTIONS:
        {
            // Options list.
            value = static_cast<QComboBox*>(editor)->currentText();
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

// Private slots.

void oskar_SettingsDelegate::commitAndCloseEditor(int /*index*/)
{
    QWidget* editor = static_cast<QWidget*>(sender());
    emit commitData(editor);
    closeEditor(editor);
}

// Private members.

void oskar_SettingsDelegate::setIterations(QAbstractItemModel* model,
        const QModelIndex& index)
{
    // Get the item type.
    int type = model->data(index, oskar_SettingsModel::TypeRole).toInt();

    // Set up the dialog.
    QDialog* dialog = new QDialog(view_);
    dialog->setWindowTitle("Iteration Parameters");
    QFormLayout* layout = new QFormLayout(dialog);
    QSpinBox* iterNum = new QSpinBox(dialog);
    iterNum->setMinimum(1);
    iterNum->setMaximum(INT_MAX);
    layout->addRow("Iterations", iterNum);
    QSpinBox* iterIncInt = NULL;
    oskar_DoubleSpinBox* iterIncDbl = NULL;
    if (type == oskar_SettingsItem::INT ||
            type == oskar_SettingsItem::INT_UNSIGNED ||
            type == oskar_SettingsItem::INT_POSITIVE)
    {
        iterIncInt = new QSpinBox(dialog);
        iterIncInt->setRange(-INT_MAX, INT_MAX);
        layout->addRow("Increment", iterIncInt);
    }
    else if (type == oskar_SettingsItem::DOUBLE)
    {
        iterIncDbl = new oskar_DoubleSpinBox(dialog);
        iterIncDbl->setRange(-DBL_MAX, DBL_MAX);
        layout->addRow("Increment", iterIncDbl);
    }

    // Add the buttons and connect them.
    QDialogButtonBox* buttons = new QDialogButtonBox(
            (QDialogButtonBox::Ok | QDialogButtonBox::Cancel),
            Qt::Horizontal, dialog);
    layout->setWidget(2, QFormLayout::SpanningRole, buttons);
    connect(buttons, SIGNAL(accepted()), dialog, SLOT(accept()));
    connect(buttons, SIGNAL(rejected()), dialog, SLOT(reject()));

    // Fill the widget data from the item.
    int num = model->data(index, oskar_SettingsModel::IterationNumRole).toInt();
    QVariant inc = model->data(index, oskar_SettingsModel::IterationIncRole);
    iterNum->setValue(num);
    if (type == oskar_SettingsItem::INT ||
            type == oskar_SettingsItem::INT_UNSIGNED ||
            type == oskar_SettingsItem::INT_POSITIVE)
        iterIncInt->setValue(inc.toInt());
    else if (type == oskar_SettingsItem::DOUBLE)
        iterIncDbl->setValue(inc.toDouble());

    if (dialog->exec() == QDialog::Accepted)
    {
        // Set the iteration data.
        model->setData(index, iterNum->value(),
                oskar_SettingsModel::IterationNumRole);
        if (type == oskar_SettingsItem::INT ||
                type == oskar_SettingsItem::INT_UNSIGNED ||
                type == oskar_SettingsItem::INT_POSITIVE)
            model->setData(index, iterIncInt->value(),
                    oskar_SettingsModel::IterationIncRole);
        else if (type == oskar_SettingsItem::DOUBLE)
            model->setData(index, iterIncDbl->value(),
                    oskar_SettingsModel::IterationIncRole);

        // Set the iteration flag.
        model->setData(index, 0, oskar_SettingsModel::SetIterationRole);
    }
    delete dialog;
}
