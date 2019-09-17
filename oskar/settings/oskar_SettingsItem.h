/*
 * Copyright (c) 2015-2017, The University of Oxford
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

#ifndef OSKAR_SETTINGS_ITEM_HPP_
#define OSKAR_SETTINGS_ITEM_HPP_

/**
 * @file oskar_SettingsItem.h
 */

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus

namespace oskar {

class SettingsDependencyGroup;
class SettingsKey;
class SettingsValue;
struct SettingsItemPrivate;

/*!
 * @class SettingsItem
 *
 * @brief SettingsItem container class.
 *
 * @details
 * Container class for a setting.
 *
 * A setting has the following parameters:
 * - key: Key used to identify a setting.
 * - label: Brief description of the setting.
 * - description: Detailed description of the setting.
 * - value: The value of the setting.
 * - required flag: Flag indicating that the setting is required.
 * - priority level: Used to mark high priority settings.
 * - dependencies: Dependency tree used to determine if the setting is active.
 *
 * The settings value is a object of @class SettingsValue, this is a stack-
 * based discriminated union container which holds the value and default value
 * of the setting as a well defined type.
 *
 * Settings priority can be used to mark a setting as important. This can be
 * used for example to control if a setting which has not been set should
 * be written to the log or appear in the settings file.
 *
 * SettingsItem can depend on other settings items as specified by the
 * dependency tree.
 */

class OSKAR_SETTINGS_EXPORT SettingsItem
{
public:
    /*! Settings type enum. */
    enum ItemType { INVALID = -1, LABEL, SETTING };

    /*! Default constructor. */
    SettingsItem();

    /*! Constructor. */
    SettingsItem(const char* key,
                 const char* label = 0,
                 const char* description = 0,
                 const char* type_name = 0,
                 const char* type_default = 0,
                 const char* type_parameters = 0,
                 bool is_required = false,
                 int priority = 0);

    /*! Destructor. */
    virtual ~SettingsItem();

    /*! Adds a dependency to the current group */
    bool add_dependency(const char* dependency_key,
                        const char* value,
                        const char* logic = 0);

    /*! Begin a dependency group with a specified combination @p logic */
    bool begin_dependency_group(const char* logic);

    /*! Return the root of the settings dependency tree. */
    const SettingsDependencyGroup* dependency_tree() const;

    /*! Return the setting description as a string. */
    const char* description() const;

    /*! Return the default value as a string. */
    const char* default_value() const;

    /*! Close the current dependency group. */
    void end_dependency_group();

    /*! Return true if the setting is required. */
    bool is_required() const;

    /*! Return true if the setting has been set (i.e. is not default). */
    bool is_set() const;

    /*! Return the settings item type, label or setting. */
    SettingsItem::ItemType item_type() const;

    /*! Return the setting key as a string. */
    const char* key() const;

    /*! Return the setting label (short description). */
    const char* label() const;

    /*! Return the number of dependencies of the setting. */
    int num_dependencies() const;

    /*! Return the number of dependency groups of the setting. */
    int num_dependency_groups() const;

    /*! Return the priority of the setting. */
    int priority() const;

    /*! Return a reference to the settings key object. */
    const SettingsKey& settings_key() const;

    /*! Return a reference to the settings value object. */
    const SettingsValue& settings_value() const;

    /*! Set the value of the setting. */
    bool set_value(const char* value);

    /*! Return the setting value as a string. */
    const char* value() const;

protected:
    void set_priority(int value);

private:
    SettingsItem(const SettingsItem& item);
    SettingsItemPrivate* p;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_ITEM_HPP_ */
