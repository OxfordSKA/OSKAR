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

class SettingsItem
{
public:
    enum ItemType { INVALID = -1, LABEL, SETTING };
    enum Priority { DEFAULT = 0, IMPORTANT = 1 };

    SettingsItem();

    SettingsItem(const std::string& key,
                 const std::string& label = std::string(),
                 const std::string& description = std::string(),
                 const std::string& type_name = std::string(),
                 const std::string& type_default = std::string(),
                 const std::string& type_parameters = std::string(),
                 bool is_required = false,
                 const std::string& priority = "DEFAULT");

    virtual ~SettingsItem();

    bool begin_dependency_group(const std::string& logic);
    void end_dependency_group();
    /* Pushes into the vector of dependencies in the dependency node
     * specified by the current group vector. */
    bool add_dependency(const std::string& dependency_key,
                        const std::string& value,
                        const std::string& logic = "EQ");

    SettingsItem::ItemType item_type() const;

    const SettingsKey& key() const { return key_; }
    const std::string& label() const { return label_; }
    const std::string& description() const { return description_; }
    bool is_required() const { return required_; }
    bool set_value(const std::string& value);
    const SettingsValue& value() const { return value_; }
    SettingsItem::Priority priority() const { return priority_; }

    int num_dependencies() const { return num_dependencies_; }
    int num_dependency_groups() const { return num_dependency_groups_; }
    const SettingsDependencyGroup* dependency_tree() const { return root_; }

protected:
    SettingsKey key_;
    std::string label_;
    std::string description_;
    SettingsValue value_;
    bool required_;
    Priority priority_;

    int num_dependency_groups_;
    int num_dependencies_;
    SettingsDependencyGroup* root_;
    SettingsDependencyGroup* current_group_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_ITEM_HPP_ */
