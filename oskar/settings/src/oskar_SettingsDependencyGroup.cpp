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
        return SettingsDependencyGroup::AND;
    if (!strncmp(s, "AND", 3))
        return SettingsDependencyGroup::AND;
    else if (!strncmp(s, "OR", 2))
        return SettingsDependencyGroup::OR;
    return SettingsDependencyGroup::UNDEF;
}

} // namespace oskar
