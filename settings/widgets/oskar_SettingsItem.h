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

#ifndef OSKAR_SETTINGS_ITEM_H_
#define OSKAR_SETTINGS_ITEM_H_

/**
 * @file oskar_SettingsItem.h
 */

#include <QtCore/QList>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QVariant>

class oskar_SettingsItem
{
public:
    enum type_id {
        UNDEF,
        LABEL,              /**< Text label only (no data). */
        BOOL,               /**< Boolean data type (true or false). */
        INT,                /**< Generic integer data type. */
        INT_UNSIGNED,       /**< Unsigned integer data type. */
        INT_POSITIVE,       /**< Positive integer data type. */
        DOUBLE,             /**< Generic double-precision data type. */
        DOUBLE_MAX,
        DOUBLE_MIN,
        STRING,             /**< Generic string data type. */
        TELESCOPE_DIR_NAME, /**< Telescope directory string data type. */
        OUTPUT_FILE_NAME,   /**< Output file name string data type. */
        INPUT_FILE_NAME,    /**< Input file name string data type. */
        INPUT_FILE_LIST,    /**< Input file string list data type. */
        INT_CSV_LIST,       /**< Integer list data type (comma-separated). */
        DOUBLE_CSV_LIST,    /**< Double-precision list data type (comma-separated). */
        OPTIONS,            /**< Option list data type. */
        RANDOM_SEED,        /**< Random seed data type. */
        AXIS_RANGE,
        DATE_TIME,          /**< Date and time data type. */
        TIME                /**< Time data type. */
    };

public:
    oskar_SettingsItem(const QString& key, const QString& subkey, int type,
            const QString& label, const QVariant& value, bool required = false,
            const QVariant& defaultValue = QVariant(),
            const QStringList& options = QStringList(),
            oskar_SettingsItem* parent = 0);
    ~oskar_SettingsItem();

    void addDependent(const QString& key);
    void appendChild(oskar_SettingsItem* child);
    oskar_SettingsItem* child(int row);
    int childCount() const;
    int childNumber() const;
    int critical() const;
    const QVariant& defaultValue() const;
    const QString& dependencyKey() const;
    const QVariant& dependencyValue() const;
    const QList<QString>& dependentKeys() const;
    bool disabled() const;
    const QString& key() const;
    const QString& label() const;
    const QStringList& options() const;
    oskar_SettingsItem* parent();
    bool required() const;
    void setDefaultValue(const QVariant& value);
    void setDisabled(bool value);
    void setLabel(const QString& value);
    void setTooltip(const QString& value);
    void setValue(const QVariant& value);
    void setDependencyKey(const QString& key);
    void setDependencyValue(const QVariant& value);
    const QString& subkey() const;
    const QString& tooltip() const;
    int type() const;
    const QVariant& value() const;
    const QVariant& valueOrDefault() const;
    int valueSet() const;

private:
    void setCritical(bool value);
    void setRequired(bool value);
    void setValueSet(bool value);

private:
    oskar_SettingsItem* parentItem_;
    QList<oskar_SettingsItem*> childItems_;
    QString key_;    // Full settings key for the item.
    QString subkey_; // Short settings key.
    int type_;       // Enumerated type.
    int valueSet_;
    bool disabled_;  // Flag to indicate an item is disabled, e.g. dependencies are not satisfied.
    bool required_;  // Flag to indicate an item is required, i.e. has no default and must be set.
    QString label_;
    QString tooltip_;
    QVariant value_;
    QVariant defaultValue_;
    QStringList options_;
    int critical_;
    QString dependencyKey_;
    QVariant dependencyValue_;
    QList<QString> dependentKeys_;
};

#endif /* OSKAR_SETTINGS_ITEM_H_ */
