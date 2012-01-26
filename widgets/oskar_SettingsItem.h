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

#ifndef OSKAR_SETTINGS_ITEM_H_
#define OSKAR_SETTINGS_ITEM_H_

/**
 * @file oskar_SettingsItem.h
 */

#include <QtCore/QList>
#include <QtCore/QVector>
#include <QtCore/QString>
#include <QtCore/QVariant>

class oskar_SettingsItem
{
public:
    enum {
        OSKAR_SETTINGS_CAPTION_ONLY, /* e.g. "Sky model settings". */
        OSKAR_SETTINGS_BOOL,         /* e.g. disable station beam. */
        OSKAR_SETTINGS_INT,
        OSKAR_SETTINGS_DOUBLE,
        OSKAR_SETTINGS_STRING,       /* e.g. output file names. */
        OSKAR_SETTINGS_DIR,          /* e.g. station directory. */
        OSKAR_SETTINGS_FILE,         /* e.g. sky model file. */
        OSKAR_SETTINGS_FILE_LIST,    /* e.g. MeerKAT antenna files. */
        OSKAR_SETTINGS_INT_CSV_LIST, /* e.g. CUDA device IDs. */
        OSKAR_SETTINGS_OPTIONS,      /* e.g. generator type, noise type. */
        OSKAR_SETTINGS_RANDOM_SEED,
        OSKAR_SETTINGS_DATETIME
    };

public:
    oskar_SettingsItem(const QString& key, const QString& keyShort, int type,
            const QVector<QVariant>& data, oskar_SettingsItem* parent = 0);
    ~oskar_SettingsItem();

    void appendChild(oskar_SettingsItem* child);
    oskar_SettingsItem* child(int row);
    int childCount() const;
    int childNumber() const;
    int columnCount() const;
    QVariant data(int column) const;
    bool insertColumns(int position, int columns);
    oskar_SettingsItem* parent();
    bool setData(int column, const QVariant &value);
    QString key() const;
    QString keyShort() const;
    int type() const;

private:
    oskar_SettingsItem* parentItem_;
    QString key_; // Full settings key for the item.
    QString keyShort_; // Short settings key.
    int type_; // Enumerated type.
    QList<oskar_SettingsItem*> childItems_;
    QVector<QVariant> itemData_;
};

#endif /* OSKAR_SETTINGS_ITEM_H_ */
