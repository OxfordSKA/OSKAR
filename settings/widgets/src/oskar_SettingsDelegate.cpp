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

#include <oskar_DoubleSpinBox.h>
#include <oskar_SettingsDelegate.h>
#include <oskar_SettingsItem.h>
#include <oskar_SettingsModel.h>

#include <QtCore/QEvent>
#include <QtCore/QModelIndex>
#include <QtGui/QMenu>
#include <QtGui/QMouseEvent>
#include <QtGui/QFileDialog>
#include <QtGui/QLineEdit>
#include <QtGui/QSpinBox>
#include <QtGui/QDateTimeEdit>
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
        case oskar_SettingsItem::DOUBLE_MAX:
        {
            oskar_DoubleSpinBox* spinner = new oskar_DoubleSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-DBL_MIN, DBL_MAX);
            spinner->setMinText("max");
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::DOUBLE_MIN:
        {
            oskar_DoubleSpinBox* spinner = new oskar_DoubleSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-DBL_MIN, DBL_MAX);
            spinner->setMinText("min");
            editor = spinner;
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Date and time editors.
            QLineEdit* line = new QLineEdit(parent);
            line->setFrame(false);
            editor = line;
            break;
        }
        case oskar_SettingsItem::TIME:
        {
            // Time editors.
            QLineEdit* line = new QLineEdit(parent);
            line->setFrame(false);
            editor = line;
            break;
        }
        case oskar_SettingsItem::RANDOM_SEED:
        {
            // Random seed editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(0, INT_MAX);
            spinner->setSpecialValueText("time");
            editor = spinner;
            break;
        }
        // Better name? (0 -> INT_MAX or max)
        case oskar_SettingsItem::AXIS_RANGE:
        {
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
        case oskar_SettingsItem::DOUBLE_CSV_LIST:
        {
            // Floating-point arrays.
            QLineEdit* line = new QLineEdit(parent);
            QValidator* validator = new QRegExpValidator(
                    QRegExp("^[+-]?\\d+(?:\\.\\d+)?(,[+-]?\\d+(?:\\.\\d+)?)*$"),
                    line);
            line->setFrame(false);
            line->setValidator(validator);
            editor = line;
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
        else if (type == oskar_SettingsItem::INPUT_FILE_LIST)
        {
            QStringList list = value.toStringList();
            QString dir;
            if (!list.isEmpty())
                dir = list[0];
            list = QFileDialog::getOpenFileNames(view_,
                    "Input file name(s)", dir);
            if (!list.isEmpty())
            {
                for (int i = 0; i < list.size(); ++i)
                {
                    list[i] = QDir::current().relativeFilePath(list[i]);
                }
                mod->setData(index, list, Qt::EditRole);
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
            // Set up the context menu.
            QMenu menu;
            QString strResetValue = "Reset";

            // Add reset action if value is not null.
            QVariant val = mod->data(index, oskar_SettingsModel::ValueRole);
            if (!val.isNull())
                menu.addAction(strResetValue);

            // Return if the menu is empty.
            if (menu.isEmpty())
            {
                return QStyledItemDelegate::editorEvent(event, mod,
                        option, index);
            }

            // Display the context menu.
            QAction* action = menu.exec(mouseEvent->globalPos());

            // Check which action was selected.
            if (action && action->text() == strResetValue)
                mod->setData(index, QVariant(), Qt::EditRole);
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
        case oskar_SettingsItem::DOUBLE_MAX:
        {
            oskar_DoubleSpinBox* e = static_cast<oskar_DoubleSpinBox*>(editor);
            if (value.toString().toUpper() == "MAX" || value.toDouble() < 0.0)
                e->setValue(e->rangeMin());
            else
                e->setValue(value.toDouble());
            break;
        }
        case oskar_SettingsItem::DOUBLE_MIN:
        {
            oskar_DoubleSpinBox* e = static_cast<oskar_DoubleSpinBox*>(editor);
            if (value.toString().toUpper() == "MIN" || value.toDouble() < 0.0)
                e->setValue(e->rangeMin());
            else
                e->setValue(value.toDouble());
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Date and time editors.
            static_cast<QLineEdit*>(editor)->setText(value.toString());
            break;
        }
        case oskar_SettingsItem::TIME:
        {
            // Time editors.
            static_cast<QLineEdit*>(editor)->setText(value.toString());
            break;
        }
        case oskar_SettingsItem::RANDOM_SEED:
        {
            // Random seed editors.
            if (value.toString().toUpper() == "TIME" || value.toInt() < 1)
                static_cast<QSpinBox*>(editor)->setValue(0);
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
        case oskar_SettingsItem::DOUBLE_MAX:
        {
            // Double spin box editors.
            oskar_DoubleSpinBox* e = static_cast<oskar_DoubleSpinBox*>(editor);
            value = e->value();
            if (value.toDouble() <= e->rangeMin())
                value = e->minText();
            break;
        }
        case oskar_SettingsItem::DOUBLE_MIN:
        {
            // Double spin box editors.
            oskar_DoubleSpinBox* e = static_cast<oskar_DoubleSpinBox*>(editor);
            value = e->value();
            if (value.toDouble() <= e->rangeMin())
                value = e->minText();
            break;
        }
        case oskar_SettingsItem::DATE_TIME:
        {
            // Date and time editors.
            value = static_cast<QLineEdit*>(editor)->text();

            // Validate these strings...
            //      d-M-yyyy h:m:s.z
            //      yyyy/M/d/h:m:s.z
            //      yyyy-M-d h:m:s.z
            //      yyyy-M-dTh:m:s.z
            //      MJD.fraction
            QString h = "([01]?[0-9]|2[0-3])";       // 00-23
            QString m = "[0-5]?[0-9]";               // 00-59
            QString s = m;
            QString z = "(\\.\\d{1,3})?";            // 000-999
            QString d = "(0?[1-9]|[12][0-9]|3[01])"; // 01-31
            QString M = "(|1[0-2]|0?[1-9]|)";        // 01-12
            QString y = "\\d{4,4}";                  // yyyy
            QString rtime  = h+":"+m+":"+s+z;        // h:m:s.zzz
            QString rdate1 = "("+d+"-"+M+"-"+y+"\\s"+rtime+")";
            QString rdate2 = "("+y+"/"+M+"/"+d+"/"+rtime+")";
            QString rdate3 = "("+y+"-"+M+"-"+d+"\\s"+rtime+")";
            QString rdate4 = "("+y+"-"+M+"-"+d+"T"+rtime+")";
            QString rdate5 = "(\\d+\\.?\\d*)";
            QString rdatetime = rdate1+"|"+rdate2+"|"+rdate3+"|"+rdate4+"|"+rdate5;
            QRegExpValidator validator(QRegExp(rdatetime), 0);
            int pos = 0;
            QString v = value.toString();
            if (validator.validate(v, pos) != QValidator::Acceptable &&
                    !v.isEmpty())
                return;
            break;
        }
        case oskar_SettingsItem::TIME:
        {
            // Time editors.
            value = static_cast<QLineEdit*>(editor)->text();

            QString h = "([01]?[0-9]|2[0-3])";       // 00-23
            QString m = "[0-5]?[0-9]";               // 00-59
            QString s = m;
            QString z = "(\\.\\d{1,3})?";            // 000-999
            QString rtime  = h+":"+m+":"+s+z;        // h:m:s.zzz
            QString sec    = "(\\d+\\.?\\d*)";
            QRegExpValidator validator(QRegExp(rtime+"|"+sec), 0);
            int pos = 0;
            QString v = value.toString();
            if (validator.validate(v, pos) != QValidator::Acceptable &&
                    !v.isEmpty())
                return;
            break;
        }
        case oskar_SettingsItem::RANDOM_SEED:
        {
            // Random seed editors.
            int val = static_cast<QSpinBox*>(editor)->value();
            if (val < 1)
                value = static_cast<QSpinBox*>(editor)->specialValueText();
            else
                value = val;
            break;
        }
        case oskar_SettingsItem::AXIS_RANGE:
        {
            // Random seed editors.
            int val = static_cast<QSpinBox*>(editor)->value();
            if (val < 0)
                value = static_cast<QSpinBox*>(editor)->specialValueText();
            else
                value = val;
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
            break;
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
