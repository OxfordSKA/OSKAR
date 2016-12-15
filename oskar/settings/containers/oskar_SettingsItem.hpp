/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
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
 * @file oskar_SettingsItem.hpp
 */

#include <oskar_SettingsValue.hpp>
#include <oskar_SettingsKey.hpp>
#include <oskar_SettingsDependency.hpp>
#include <oskar_SettingsDependencyGroup.hpp>

#ifdef __cplusplus

namespace oskar {

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

class SettingsItem
{
public:
    /*! Settings type enum. */
    enum ItemType { INVALID = -1, LABEL, SETTING };

    /*! Default constructor. */
    SettingsItem();

    /*! Constructor. */
    SettingsItem(const std::string& key,
                 const std::string& label = std::string(),
                 const std::string& description = std::string(),
                 const std::string& type_name = std::string(),
                 const std::string& type_default = std::string(),
                 const std::string& type_parameters = std::string(),
                 bool is_required = false,
                 int priority = 0);

    /*! Destructor. */
    virtual ~SettingsItem();

    /*! Begin a dependency group with a specified combination @p logic */
    bool begin_dependency_group(const std::string& logic);

    /*! Close the current dependency group. */
    void end_dependency_group();

    /*! Adds a dependency to the current group */
    bool add_dependency(const std::string& dependency_key,
                        const std::string& value,
                        const std::string& logic = "EQ");

    /*! Return the settings item type, label or setting. */
    SettingsItem::ItemType item_type() const;

    /*! Return the setting key */
    const SettingsKey& key() const { return key_; }

    /*! Return the setting label (short description) */
    const std::string& label() const { return label_; }

    /*! Return the setting description */
    const std::string& description() const { return description_; }

    /*! Return true if the setting is required. */
    bool is_required() const { return required_; }

    /*! Set the value of the setting. */
    bool set_value(const std::string& value);

    /*! Return a reference to the settings value object. */
    const SettingsValue& value() const { return value_; }

    /*! Return the priority of the setting. */
    int priority() const { return priority_; }

    /*! Return the number of dependencies of the setting */
    int num_dependencies() const { return num_dependencies_; }

    /*! Return the number of dependency groups of the setting */
    int num_dependency_groups() const { return num_dependency_groups_; }

    /*! Return the root of the settings dependency tree. */
    const SettingsDependencyGroup* dependency_tree() const { return root_; }

protected:
    SettingsKey key_;
    std::string label_;
    std::string description_;
    SettingsValue value_;
    bool required_;
    int priority_;

    int num_dependency_groups_;
    int num_dependencies_;
    SettingsDependencyGroup* root_;
    SettingsDependencyGroup* current_group_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_ITEM_HPP_ */
