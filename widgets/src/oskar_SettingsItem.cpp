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

#include "widgets/oskar_SettingsItem.h"

#include <QtCore/QStringList>
#include <cstdio>

oskar_SettingsItem::oskar_SettingsItem(const QString& key,
        const QString& subkey, int type, const QString& label,
        const QVariant& defaultValue, oskar_SettingsItem* parent)
{
    // Initialise constructed values.
    key_ = key;
    subkey_ = subkey;
    type_ = type;
    label_ = label;
    default_ = defaultValue;
    parentItem_ = parent;

    // Initialise user-defined, runtime values.
    iterNum_ = 1;
    visible_ = 0;
    if (key.isEmpty())
        value_ = defaultValue;
}

oskar_SettingsItem::~oskar_SettingsItem()
{
    qDeleteAll(childItems_);
}

void oskar_SettingsItem::appendChild(oskar_SettingsItem* item)
{
    childItems_.append(item);
}

oskar_SettingsItem* oskar_SettingsItem::child(int row)
{
    return childItems_.value(row);
}

int oskar_SettingsItem::childCount() const
{
    return childItems_.count();
}

int oskar_SettingsItem::childNumber() const
{
    if (parentItem_)
        return parentItem_->childItems_.indexOf(const_cast<oskar_SettingsItem*>(this));
    return 0;
}

const QVariant& oskar_SettingsItem::defaultValue() const
{
    return default_;
}

const QVariant& oskar_SettingsItem::iterationInc() const
{
    return iterInc_;
}

int oskar_SettingsItem::iterationNum() const
{
    return iterNum_;
}

const QString& oskar_SettingsItem::key() const
{
    return key_;
}

const QString& oskar_SettingsItem::label() const
{
    return label_;
}

oskar_SettingsItem* oskar_SettingsItem::parent()
{
    return parentItem_;
}

void oskar_SettingsItem::setIterationInc(const QVariant& value)
{
    iterInc_ = value;
}

void oskar_SettingsItem::setIterationNum(int value)
{
    iterNum_ = value;
}

void oskar_SettingsItem::setLabel(const QString& value)
{
    label_ = value;
}

void oskar_SettingsItem::setTooltip(const QString& value)
{
    tooltip_ = value;
}

void oskar_SettingsItem::setValue(const QVariant& value)
{
    if (type_ == LABEL)
        return;
    if (value_.isNull() != value.isNull())
        setVisible(!value.isNull());
    value_ = value;
}

const QString& oskar_SettingsItem::subkey() const
{
    return subkey_;
}

const QString& oskar_SettingsItem::tooltip() const
{
    return tooltip_;
}

int oskar_SettingsItem::type() const
{
    return type_;
}

const QVariant& oskar_SettingsItem::value() const
{
    return value_;
}

int oskar_SettingsItem::visible() const
{
    return visible_;
}

// Private members.

void oskar_SettingsItem::setVisible(bool value)
{
    if (value)
        ++visible_;
    else
        --visible_;
    if (parentItem_)
        parentItem_->setVisible(value);
}
