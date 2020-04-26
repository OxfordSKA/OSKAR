/*
 * Copyright (c) 2015-2020, The University of Oxford
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

#include "gui/oskar_DoubleSpinBox.h"
#include "gui/oskar_SettingsDelegate.h"
#include "gui/oskar_SettingsModel.h"
#include "settings/oskar_SettingsItem.h"
#include "settings/oskar_SettingsValue.h"

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDateTimeEdit>
#include <QEvent>
#include <QFileDialog>
#include <QLineEdit>
#include <QMenu>
#include <QModelIndex>
#include <QMouseEvent>
#include <QPainter>
#include <QSpinBox>

#include <cstdio>
#include <climits>
#include <cfloat>
#include <iostream>

using namespace std;

namespace oskar {

SettingsDelegate::SettingsDelegate(QWidget* view, QObject* parent)
: QStyledItemDelegate(parent)
{
    view_ = view;
}

QWidget* SettingsDelegate::createEditor(QWidget* parent,
        const QStyleOptionViewItem& /*option*/, const QModelIndex& index) const
{
    // Get the setting type.
    int type = index.model()->data(index, SettingsModel::TypeRole).toInt();

    // Create the appropriate editor.
    QWidget* editor = 0;
    switch (type)
    {
        case SettingsValue::INT:
        {
            // Spin box editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-INT_MAX, INT_MAX);
            editor = spinner;
            break;
        }
        case SettingsValue::UNSIGNED_INT:
        {
            // Spin box editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(0, INT_MAX);
            editor = spinner;
            break;
        }
        case SettingsValue::INT_POSITIVE:
        {
            // Spin box editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(1, INT_MAX);
            editor = spinner;
            break;
        }
        case SettingsValue::UNSIGNED_DOUBLE:
        {
            // Double spin box editors.
            DoubleSpinBox* spinner = new DoubleSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(0, DBL_MAX);
            editor = spinner;
            break;
        }
        case SettingsValue::DOUBLE:
        {
            // Double spin box editors.
            DoubleSpinBox* spinner = new DoubleSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(-DBL_MAX, DBL_MAX);
            editor = spinner;
            break;
        }
        case SettingsValue::DOUBLE_RANGE:
        {
            // Double spin box editors.
            DoubleSpinBox* spinner = new DoubleSpinBox(parent);
            QVariant v = index.model()->data(index, SettingsModel::RangeRole);
            QList<QVariant> range = v.toList();
            double min_ = range[0].toDouble();
            double max_ = range[1].toDouble();
            spinner->setFrame(false);
            spinner->setRange(min_, max_);
            editor = spinner;
            break;
        }
        case SettingsValue::DOUBLE_RANGE_EXT:
        {
            QVariant v = index.model()->data(index, SettingsModel::ExtRangeRole);
            QList<QVariant> range = v.toList();
            double min_ = range[0].toDouble();
            double max_ = range[1].toDouble();
            QString ext_min_ = range[2].toString();
            DoubleSpinBox* spinner = new DoubleSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(min_, max_);
            spinner->setMinText(ext_min_);
            editor = spinner;
            break;
        }
        case SettingsValue::DATE_TIME:
        {
            // Date and time editors.
            QLineEdit* line = new QLineEdit(parent);
            line->setFrame(false);
            editor = line;
            break;
        }
        case SettingsValue::TIME:
        {
            // Time editors.
            QLineEdit* line = new QLineEdit(parent);
            line->setFrame(false);
            editor = line;
            break;
        }
        case SettingsValue::RANDOM_SEED:
        {
            // Random seed editors.
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(0, INT_MAX);
            spinner->setSpecialValueText("time");
            editor = spinner;
            break;
        }
        case SettingsValue::INT_RANGE_EXT:
        {
            QVariant v = index.model()->data(index, SettingsModel::ExtRangeRole);
            QList<QVariant> range = v.toList();
            int min_ = range[0].toInt();
            int max_ = range[1].toInt();
            QString ext_min_ = range[2].toString();
            QSpinBox* spinner = new QSpinBox(parent);
            spinner->setFrame(false);
            spinner->setRange(min_, max_);
            spinner->setSpecialValueText(ext_min_);
            editor = spinner;
            break;
        }
        case SettingsValue::OPTION_LIST:
        {
            // Options list.
            QComboBox* list = new QComboBox(parent);
            QVariant data = index.model()->data(index,
                                                SettingsModel::OptionsRole);
            QStringList options = data.toStringList();
            list->setFrame(false);
            list->addItems(options);
            connect(list, SIGNAL(activated(int)),
                    this, SLOT(commitAndCloseEditor(int)));
            editor = list;
            break;
        }
        case SettingsValue::DOUBLE_LIST:
        {
            // Floating-point arrays.
            QLineEdit* line = new QLineEdit(parent);
            QRegExp regex("^[+-]?\\d+(?:\\.\\d+)?(,[+-]?\\d+(?:\\.\\d+)?)*$");
            QValidator* validator = new QRegExpValidator(regex, line);
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

bool SettingsDelegate::editorEvent(QEvent* event,
        QAbstractItemModel* mod, const QStyleOptionViewItem& option,
        const QModelIndex& index)
{
    // Get item type and value.
    int type = index.model()->data(index, SettingsModel::TypeRole).toInt();
    QVariant value = index.model()->data(index, Qt::EditRole);
    int item_type = index.model()->data(index, SettingsModel::ItemTypeRole).toInt();

    // Check for mouse double-click events.
    if (event->type() == QEvent::MouseButtonDblClick && index.column() == 1)
    {
        if (type == SettingsValue::INPUT_DIRECTORY)
        {
            QString name = QFileDialog::getExistingDirectory(
                    view_, "Select directory", value.toString());
            if (!name.isNull())
            {
                name = QDir::current().relativeFilePath(name);
                mod->setData(index, name, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        else if (type == SettingsValue::INPUT_FILE)
        {
            QString name = QFileDialog::getOpenFileName(
                    view_, "Input file name", value.toString());
            if (!name.isNull())
            {
                name = QDir::current().relativeFilePath(name);
                mod->setData(index, name, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        else if (type == SettingsValue::INPUT_FILE_LIST)
        {
            QStringList list = value.toStringList();
            QString dir;
            if (!list.isEmpty())
                dir = list[0];
            list = QFileDialog::getOpenFileNames(
                    view_, "Input file name(s)", dir);
            if (!list.isEmpty())
            {
                for (int i = 0; i < list.size(); ++i)
                    list[i] = QDir::current().relativeFilePath(list[i]);
                mod->setData(index, list, Qt::EditRole);
            }
            event->accept();
            return true;
        }
        else if (type == SettingsValue::OUTPUT_FILE)
        {
            QString name = QFileDialog::getSaveFileName(
                    view_, "Output file name", value.toString());
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
                item_type != SettingsItem::LABEL)
        {
            // Get model index to value.
            QModelIndex valueIndex = index.sibling(index.row(), 1);

            // Set up the context menu.
            QMenu menu;
            QString strResetValue = "Reset";
            QString strCopyKey = "Copy setting key";

            // Add reset action if value is not null.
            QVariant val = mod->data(valueIndex, SettingsModel::ValueRole);
            if (!val.isNull())
                menu.addAction(strResetValue);

            menu.addAction(strCopyKey);

            // Display the context menu.
            QAction* action = menu.exec(mouseEvent->globalPos());

            // Check which action was selected.
            if (action && action->text() == strResetValue)
            {
                mod->setData(valueIndex, mod->data(valueIndex,
                        SettingsModel::DefaultRole), Qt::EditRole);
            }
            else if (action && action->text() == strCopyKey)
            {
                QVariant key = mod->data(valueIndex, SettingsModel::KeyRole);
                QApplication::clipboard()->setText(key.toString());
            }
            event->accept();
            return true;
        }
        else if (mouseEvent->button() == Qt::RightButton &&
                item_type == SettingsItem::LABEL)
        {
            QMenu menu;
            QString strResetGroup = "Reset Group";
            menu.addAction(strResetGroup);
            QAction* action = menu.exec(mouseEvent->globalPos());
            if (action && action->text() == strResetGroup)
                mod->setData(index, QVariant(), SettingsModel::ResetGroupRole);
            event->accept();
            return true;
        }
    }

    return QStyledItemDelegate::editorEvent(event, mod, option, index);
}

void SettingsDelegate::setEditorData(QWidget* editor,
        const QModelIndex& index) const
{
    // Get the setting type.
    int type = index.model()->data(index, SettingsModel::TypeRole).toInt();

    // Set the editor data.
    QVariant value = index.model()->data(index, Qt::EditRole);
    switch (type)
    {
        case SettingsValue::INT:
        case SettingsValue::UNSIGNED_INT:
        case SettingsValue::INT_POSITIVE:
        {
            // Spin box editors.
            static_cast<QSpinBox*>(editor)->setValue(value.toInt());
            break;
        }
        case SettingsValue::UNSIGNED_DOUBLE:
        case SettingsValue::DOUBLE:
        case SettingsValue::DOUBLE_RANGE:
        {
            // Double spin box editors.
            static_cast<DoubleSpinBox*>(editor)->setValue(
                    value.toString());
            break;
        }
        case SettingsValue::DOUBLE_RANGE_EXT:
        {
            DoubleSpinBox* e = static_cast<DoubleSpinBox*>(editor);
            QVariant v = index.model()->data(index, SettingsModel::ExtRangeRole);
            QList<QVariant> range = v.toList();
            double min_ = range[0].toDouble();
            QString ext_min_ = range[2].toString();
            if (value.toString().toUpper() == ext_min_ ||
                    value.toDouble() < min_)
                e->setValue(e->rangeMin());
            else
                e->setValue(value.toDouble());
            break;
        }
        case SettingsValue::DATE_TIME:
        {
            // Date and time editors.
            static_cast<QLineEdit*>(editor)->setText(value.toString());
            break;
        }
        case SettingsValue::TIME:
        {
            // Time editors.
            static_cast<QLineEdit*>(editor)->setText(value.toString());
            break;
        }
        case SettingsValue::RANDOM_SEED:
        {
            // Random seed editors.
            if (value.toString().toUpper() == "TIME" || value.toInt() < 1)
                static_cast<QSpinBox*>(editor)->setValue(0);
            else
                static_cast<QSpinBox*>(editor)->setValue(value.toInt());
            break;
        }
        case SettingsValue::INT_RANGE_EXT:  // AXIS_RANGE
        {
            QVariant v = index.model()->data(index, SettingsModel::ExtRangeRole);
            QList<QVariant> range = v.toList();
            int min_ = range[0].toInt();
            QString ext_min_ = range[2].toString();
            if (value.toString().toUpper() == ext_min_ || value.toInt() < min_)
                static_cast<QSpinBox*>(editor)->setValue(-1);
            else
                static_cast<QSpinBox*>(editor)->setValue(value.toInt());
            break;
        }
        case SettingsValue::OPTION_LIST:
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

void SettingsDelegate::setModelData(QWidget* editor,
        QAbstractItemModel* model, const QModelIndex& index) const
{
    // Get the setting type.
    int type = index.model()->data(index, SettingsModel::TypeRole).toInt();

    // Get the editor data.
    QVariant value;
    switch (type)
    {
        case SettingsValue::INT:
        case SettingsValue::UNSIGNED_INT:
        case SettingsValue::INT_POSITIVE:
        {
            // Spin box editors.
            value = static_cast<QSpinBox*>(editor)->value();
            break;
        }
        case SettingsValue::DOUBLE:
        case SettingsValue::UNSIGNED_DOUBLE:
        case SettingsValue::DOUBLE_RANGE:
        {
            // Double spin box editors.
            value = static_cast<DoubleSpinBox*>(editor)->cleanText();
            break;
        }
        case SettingsValue::DOUBLE_RANGE_EXT:
        {
            // Double spin box editors.
            DoubleSpinBox* e = static_cast<DoubleSpinBox*>(editor);
            double v = e->value();
            value = e->value();
            if (v <= e->rangeMin())
                value = e->minText();
            else
                value = e->cleanText();
            break;
        }
        case SettingsValue::DATE_TIME:
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
            QString rdate6 = "(\\d+\\.?\\d*[e|E]-?\\d{1,2})";
            QString rdatetime = rdate1+"|"+rdate2+"|"+rdate3+"|"+rdate4+
                            "|"+rdate5+"|"+rdate6;
            QRegExpValidator validator(QRegExp(rdatetime), 0);
            int pos = 0;
            QString v = value.toString();
            if (validator.validate(v, pos) != QValidator::Acceptable &&
                            !v.isEmpty()) {
                cerr << "WARNING: DateTime value failed to validate." << endl;
                return;
            }
            break;
        }
        case SettingsValue::TIME:
        {
            // Time editors.
            value = static_cast<QLineEdit*>(editor)->text();
            QString h = "(\\d+)";                    // >= 0
            QString m = "[0-5]?[0-9]";               // 00-59
            QString s = m;
            QString z = "(\\.\\d{1,8})?";            // 000-99999999
            QString rtime  = h+":"+m+":"+s+z;        // h:m:s.zzz
            QString sec    = "(\\d+\\.?\\d*)";
            QString exp_sec = "(\\d+\\.?\\d*[e|E]-?\\d{1,2})";
            QRegExpValidator validator(QRegExp(rtime+"|"+sec+"|"+exp_sec), 0);
            int pos = 0;
            QString v = value.toString();
            if (validator.validate(v, pos) != QValidator::Acceptable &&
                            !v.isEmpty()) {
                cerr << "WARNING: Time value failed to validate." << endl;
                return;
            }
            break;
        }
        case SettingsValue::RANDOM_SEED:
        {
            // Random seed editors.
            int val = static_cast<QSpinBox*>(editor)->value();
            if (val < 1)
                value = static_cast<QSpinBox*>(editor)->specialValueText();
            else
                value = val;
            break;
        }
        case SettingsValue::INT_RANGE_EXT:
        {
            // Random seed editors.
            int val = static_cast<QSpinBox*>(editor)->value();
            if (val < 0)
                value = static_cast<QSpinBox*>(editor)->specialValueText();
            else
                value = val;
            break;
        }
        case SettingsValue::OPTION_LIST:
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

void SettingsDelegate::updateEditorGeometry(QWidget* editor,
        const QStyleOptionViewItem& option, const QModelIndex& /*index*/) const
{
    editor->setGeometry(option.rect);
}

// Private slots.

void SettingsDelegate::commitAndCloseEditor(int /*index*/)
{
    QWidget* editor = static_cast<QWidget*>(sender());
    emit commitData(editor);
    closeEditor(editor);
}

} /* namespace oskar */
