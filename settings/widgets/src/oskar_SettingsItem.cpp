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

#include <oskar_SettingsItem.h>

#include <QtCore/QStringList>

oskar_SettingsItem::oskar_SettingsItem(const QString& key,
        const QString& subkey, int type, const QString& label,
        const QVariant& value, bool required, const QVariant& defaultValue,
        const QStringList& options, oskar_SettingsItem* parent)
{
    // Set default defaults.
    if (type == DOUBLE || type == DOUBLE_MAX || type == DOUBLE_MIN ||
            type == DOUBLE_CSV_LIST)
        defaultValue_ = 0.0;
    else if (type == INT || type == INT_UNSIGNED)
        defaultValue_ = 0;
    else if (type == RANDOM_SEED)
        defaultValue_ = 1;
    else if (type == INT_POSITIVE)
        defaultValue_ = 1;
    else if (type == BOOL)
        defaultValue_ = false;

    // Initialise constructed values.
    key_ = key;
    subkey_ = subkey;
    type_ = type;
    label_ = label;
    value_ = value;
    if (!defaultValue.isNull())
        defaultValue_ = defaultValue;
    options_ = options;
    parentItem_ = parent;

    // Initialise user-defined, runtime values.
    critical_ = 0;
    valueSet_ = 0;
    disabled_ = false;

    // Set required flag of this and all parents if this option is required.
    required_ = false;
    setRequired(required);
    setCritical(required);
}

oskar_SettingsItem::~oskar_SettingsItem()
{
    qDeleteAll(childItems_);
}

void oskar_SettingsItem::addDependent(const QString& key)
{
    dependentKeys_.append(key);
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

int oskar_SettingsItem::critical() const
{
    return critical_;
}

const QVariant& oskar_SettingsItem::defaultValue() const
{
    return defaultValue_;
}

const QString& oskar_SettingsItem::dependencyKey() const
{
    return dependencyKey_;
}

const QVariant& oskar_SettingsItem::dependencyValue() const
{
    return dependencyValue_;
}

const QList<QString>& oskar_SettingsItem::dependentKeys() const
{
    return dependentKeys_;
}

bool oskar_SettingsItem::disabled() const
{
    if (!parentItem_) return disabled_;
    return disabled_ || (parentItem_->type() == BOOL &&
            parentItem_->valueOrDefault() == false) ||
            parentItem_->disabled();
}

const QString& oskar_SettingsItem::key() const
{
    return key_;
}

const QString& oskar_SettingsItem::label() const
{
    return label_;
}

const QStringList& oskar_SettingsItem::options() const
{
    return options_;
}

oskar_SettingsItem* oskar_SettingsItem::parent()
{
    return parentItem_;
}

bool oskar_SettingsItem::required() const
{
    return required_;
}

void oskar_SettingsItem::setDefaultValue(const QVariant& value)
{
    if (type_ == LABEL)
        return;
    defaultValue_ = value;
    if (type_ == DOUBLE)
        defaultValue_.convert(QVariant::Double);
}

void oskar_SettingsItem::setDisabled(bool value)
{
    disabled_ = value;
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
    bool nullValue = value.isNull();
    if (value_.isNull() != nullValue)
    {
        setValueSet(!nullValue);
        setCritical(nullValue);
    }
    value_ = value;
    if (type_ == DOUBLE)
        value_.convert(QVariant::Double);
}

void oskar_SettingsItem::setDependencyKey(const QString& key)
{
    dependencyKey_ = key;
}

void oskar_SettingsItem::setDependencyValue(const QVariant& value)
{
    dependencyValue_ = value;
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

const QVariant& oskar_SettingsItem::valueOrDefault() const
{
    return (value_.isNull() ? defaultValue_ : value_);
}

int oskar_SettingsItem::valueSet() const
{
    return valueSet_;
}

// Private members.

void oskar_SettingsItem::setCritical(bool value)
{
    // TODO don't increment critical for disabled settings
    if (!required_)
        return;
    if (value)
        ++critical_;
    else
        --critical_;

    if (parentItem_)
        parentItem_->setCritical(value);
}

void oskar_SettingsItem::setRequired(bool value)
{
    required_ = value;
    if (value && parentItem_)
        parentItem_->setRequired(value);
}

void oskar_SettingsItem::setValueSet(bool value)
{
    if (value)
        ++valueSet_;
    else
        --valueSet_;
    if (parentItem_)
        parentItem_->setValueSet(value);
}
