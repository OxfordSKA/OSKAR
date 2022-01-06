/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <settings/oskar_SettingsDependency.h>
#include <settings/oskar_SettingsDependencyGroup.h>
#include "settings/oskar_SettingsItem.h"
#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_SettingsValue.h"
#include <iostream>
#include <string>

using namespace std;

namespace oskar {

struct SettingsItemPrivate
{
    bool required_;
    int num_dependency_groups_, num_dependencies_, priority_;
    string label_, description_;
    SettingsDependencyGroup *root_, *current_group_;
    SettingsKey key_;
    SettingsValue value_;
};

SettingsItem::SettingsItem()
{
    p = new SettingsItemPrivate;
    p->required_ = false;
    p->num_dependency_groups_ = 0;
    p->num_dependencies_ = 0;
    p->priority_ = 0;
    p->root_ = 0;
    p->current_group_ = 0;
}

SettingsItem::SettingsItem(const char* key,
                           const char* label,
                           const char* description,
                           const char* type_name,
                           const char* type_default,
                           const char* type_parameters,
                           bool is_required,
                           int priority)
{
    p = new SettingsItemPrivate;
    p->key_.from_string(key);
    p->label_ = label ? string(label) : string();
    p->description_ = description ? string(description) : string();
    p->required_ = is_required;
    p->priority_ = priority;
    p->num_dependency_groups_ = 0;
    p->num_dependencies_ = 0;
    p->root_ = 0;
    p->current_group_ = 0;
    string type_ = type_name ? string(type_name) : string();
    if (!type_.empty())
    {
        string params = type_parameters ? string(type_parameters) : string();
        string def = type_default ? string(type_default) : string();
        SettingsValue v;
        bool ok = v.init(type_name, type_parameters);
        if (!ok) {
            cerr << "ERROR: Failed to initialise setting (key='" << key;
            cerr << "', type='" << type_name << "'" << ", parameters='";
            cerr << params << "')." << endl;
            return;
        }
        if (!def.empty())
        {
            bool default_ok = v.set_default(type_default);
            if (!default_ok) {
                cerr << "ERROR: Failed setting default for (key='" << key;
                cerr << "', type='" << type_name << "', default='";
                cerr << type_default << "', type_parameters='";
                cerr << params << "')." << endl;
            }
            ok &= default_ok;
        }
        if (ok) p->value_ = v;
    }
}

SettingsItem::~SettingsItem()
{
    if (p->root_) delete p->root_;
    delete p;
}

bool SettingsItem::add_dependency(const char* dependency_key,
                                  const char* value,
                                  const char* logic)
{
    if (!p->current_group_)
    {
        bool ok = begin_dependency_group("AND");
        if (!ok) return false;
    }
    SettingsDependency d = SettingsDependency(dependency_key, value, logic);
    if (p->current_group_)
    {
        p->current_group_->add_dependency(d);
        p->num_dependencies_++;
    }
    return true;
}

bool SettingsItem::begin_dependency_group(const char* logic)
{
    p->num_dependency_groups_++;

    // If no groups have been previously created, create the root group.
    if (!p->root_) {
        p->root_ = new SettingsDependencyGroup(logic);
        p->current_group_ = p->root_;
    }
    // Add the group below the current group.
    else if (p->current_group_) {
        p->current_group_ = p->current_group_->add_child(logic);
    }

    return true;
}

const SettingsDependencyGroup* SettingsItem::dependency_tree() const
{
    return p->root_;
}

const char* SettingsItem::default_value() const
{
    return p->value_.get_default();
}

const char* SettingsItem::description() const
{
    return p->description_.c_str();
}

void SettingsItem::end_dependency_group()
{
    p->current_group_ = p->current_group_->parent();
}

bool SettingsItem::is_required() const
{
    return p->required_;
}

bool SettingsItem::is_set() const
{
    return p->value_.is_set();
}

SettingsItem::ItemType SettingsItem::item_type() const
{
    if (p->key_.empty()) {
        return INVALID;
    } else if (p->value_.type() == SettingsValue::UNDEF) {
        return LABEL;
    } else if (p->value_.type() >= 0) {
        return SETTING;
    } else {
        return INVALID;
    }
}

const char* SettingsItem::key() const
{
    return (const char*) (p->key_);
}

const char* SettingsItem::label() const
{
    return p->label_.c_str();
}

int SettingsItem::num_dependencies() const
{
    return p->num_dependencies_;
}

int SettingsItem::num_dependency_groups() const
{
    return p->num_dependency_groups_;
}

int SettingsItem::priority() const
{
    return p->priority_;
}

const SettingsKey& SettingsItem::settings_key() const
{
    return p->key_;
}

const SettingsValue& SettingsItem::settings_value() const
{
    return p->value_;
}

bool SettingsItem::set_value(const char* value)
{
    return p->value_.set_value(value);
}

const char* SettingsItem::value() const
{
    return p->value_.get_value();
}


/* Protected methods *********************************************************/

void SettingsItem::set_priority(int value)
{
    p->priority_ = value;
}

} // namespace oskar
