/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_SettingsDependency.h"
#include "settings/oskar_SettingsDependencyGroup.h"
#include <cstring>

namespace oskar {

SettingsDependencyGroup::SettingsDependencyGroup(const char* logic,
        SettingsDependencyGroup* parent)
: parent_(parent)
{
    logic_ = string_to_group_logic(logic);
}

SettingsDependencyGroup::~SettingsDependencyGroup()
{
    for (unsigned i = 0; i < children_.size(); ++i) delete children_.at(i);
}

int SettingsDependencyGroup::num_children() const
{
    return (int) children_.size();
}

SettingsDependencyGroup* SettingsDependencyGroup::add_child(
                const char* logic)
{
    children_.push_back(new SettingsDependencyGroup(logic, this));
    return children_.back();
}

SettingsDependencyGroup* SettingsDependencyGroup::get_child(int i)
{
    return children_.at(i);
}

const SettingsDependencyGroup* SettingsDependencyGroup::get_child(int i) const
{
    return children_.at(i);
}

SettingsDependencyGroup* SettingsDependencyGroup::parent()
{
    return parent_;
}

int SettingsDependencyGroup::num_dependencies() const
{
    return (int) dependencies_.size();
}

SettingsDependencyGroup::GroupLogic SettingsDependencyGroup::group_logic() const
{
    return logic_;
}

void SettingsDependencyGroup::add_dependency(const char* key,
        const char* value, const char* logic)
{
    dependencies_.push_back(SettingsDependency(key, value, logic));
}


void SettingsDependencyGroup::add_dependency(const SettingsDependency& dep)
{
    dependencies_.push_back(dep);
}

const SettingsDependency* SettingsDependencyGroup::get_dependency(
        unsigned int i) const
{
    return i < dependencies_.size() ? &dependencies_.at(i) : 0;
}

SettingsDependencyGroup::GroupLogic
SettingsDependencyGroup::string_to_group_logic(const char* s)
{
    if (!s || strlen(s) == 0)
    {
        return SettingsDependencyGroup::AND;
    }
    if (!strncmp(s, "AND", 3))
    {
        return SettingsDependencyGroup::AND;
    }
    else if (!strncmp(s, "OR", 2))
    {
        return SettingsDependencyGroup::OR;
    }
    return SettingsDependencyGroup::UNDEF;
}

} // namespace oskar
