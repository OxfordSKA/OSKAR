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
        CAPTION_ONLY,       /* e.g. "Sky model settings". */
        BOOL,               /* e.g. disable station beam. */
        INT,
        DOUBLE,
        STRING,
        OUTPUT_FILE_NAME,   /* e.g. visibility data. */
        INPUT_DIR_NAME,     /* e.g. station directory. */
        INPUT_FILE_NAME,    /* e.g. sky model file. */
        INPUT_FILE_LIST,    /* e.g. MeerKAT antenna files. */
        INT_CSV_LIST,       /* e.g. CUDA device IDs. */
        OPTIONS,            /* e.g. generator type, noise type. */
        RANDOM_SEED,
        DATE_TIME,          /* e.g. observation start time. */
        TIME                /* e.g. observation duration. */
    };

public:
    oskar_SettingsItem(const QString& key, const QString& subkey, int type,
            const QString& caption, const QVariant& defaultValue,
            oskar_SettingsItem* parent = 0);
    ~oskar_SettingsItem();

    void appendChild(oskar_SettingsItem* child);
    const QString& caption() const;
    oskar_SettingsItem* child(int row);
    int childCount() const;
    int childNumber() const;
    const QVariant& defaultValue() const;
    const QVariant& iterationInc() const;
    int iterationNum() const;
    const QString& key() const;
    oskar_SettingsItem* parent();
    void setCaption(const QString& value);
    void setIterationInc(const QVariant& value);
    void setIterationNum(int value);
    void setValue(const QVariant& value);
    const QString& subkey() const;
    int type() const;
    const QVariant& value() const;

private:
    oskar_SettingsItem* parentItem_;
    QList<oskar_SettingsItem*> childItems_;
    QString key_; // Full settings key for the item.
    QString subkey_; // Short settings key.
    int type_; // Enumerated type.
    QString caption_;
    QVariant value_;
    QVariant default_;
    int iterNum_;
    QVariant iterInc_;
};

#endif /* OSKAR_SETTINGS_ITEM_H_ */
