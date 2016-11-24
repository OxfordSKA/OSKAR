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

#include "oskar_SettingsKey.hpp"
#include "oskar_settings_types.hpp"
#include "oskar_SettingsItem.hpp"
#include "oskar_settings_utility_string.hpp"
#include <iostream>

using namespace std;

namespace oskar {

SettingsItem::SettingsItem()
: required_(false), num_dependency_groups_(0),
  num_dependencies_(0), root_(0), current_group_(0)
{
}

SettingsItem::SettingsItem(const std::string& key,
                           const std::string& label,
                           const std::string& description,
                           const std::string& type_name,
                           const std::string& type_default,
                           const std::string& type_parameters,
                           bool is_required,
                           const std::string& priority)
: key_(key), label_(label), description_(description), required_(is_required),
  num_dependency_groups_(0), num_dependencies_(0), root_(0), current_group_(0)
{
    if (!type_name.empty()) {
        SettingsValue v;
        bool ok = v.init(type_name, type_parameters);
        if (!ok) {
            cerr << "ERROR: Failed to initialise setting (key='" << key;
            cerr << "', type='" << type_name << "'" << ", parameters='";
            cerr << type_parameters << "')." << endl;
            return;
        }
        if (!type_default.empty()) {
            bool default_ok = v.set_default(type_default);
            if (!default_ok) {
                cerr << "ERROR: Failed setting default for (key='" << key;
                cerr << "', type='" << type_name << "', default='";
                cerr << type_default << "', type_parameters='";
                cerr << type_parameters << "')." << endl;
            }
            ok &= default_ok;
        }
        if (ok) {
            value_ = v;
        }
    }
    if (!priority.empty()) {
        if (oskar_settings_utility_string_starts_with(priority, "N", false) ||
                oskar_settings_utility_string_starts_with(priority, "D", false))
            priority_ = DEFAULT;
        else if (oskar_settings_utility_string_starts_with(priority, "I", false))
            priority_ = IMPORTANT;
        else {
            cerr << "ERROR: Failed setting priority for (key='" << key;
            cerr << ", priority='" << priority << "')" << endl;
        }
    }
}

SettingsItem::~SettingsItem()
{
}

bool SettingsItem::begin_dependency_group(const std::string& logic)
{
    num_dependency_groups_++;

    // If no groups have been previously created, create the root group.
    if (!root_) {
        root_ = new SettingsDependencyGroup(logic);
        current_group_ = root_;
    }
    // Add the group below the current group.
    else {
        current_group_ = current_group_->add_child(logic);
    }

    return true;
}

void SettingsItem::end_dependency_group()
{
    current_group_ = current_group_->parent();
}


bool SettingsItem::add_dependency(const std::string& dependency_key,
                                  const std::string& value,
                                  const std::string& logic)
{
    if (!current_group_) {
        bool ok = begin_dependency_group("AND");
        if (!ok) return false;
    }
    SettingsDependency d = SettingsDependency(dependency_key, value, logic);
    current_group_->add_dependency(d);
    num_dependencies_++;
    return true;
}

SettingsItem::ItemType SettingsItem::item_type() const
{
    if (key_.empty())
        return INVALID;
    else if (value_.type() == SettingsValue::UNDEF)
        return LABEL;
    else if (value_.type() >= 0)
        return SETTING;
    else
        return INVALID;
}

bool SettingsItem::set_value(const std::string& value)
{
    return value_.set_value(value);
}

} // namespace oskar

