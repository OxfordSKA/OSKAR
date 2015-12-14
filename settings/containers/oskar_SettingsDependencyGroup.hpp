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

#ifndef OSKAR_SETTINGS_DEPENDENCY_GROUP_HPP_
#define OSKAR_SETTINGS_DEPENDENCY_GROUP_HPP_

/**
 * @file oskar_SettingsDependencyGroup.hpp
 */

#include <oskar_SettingsDependency.hpp>
#include <oskar_SettingsKey.hpp>

#ifdef __cplusplus

#include <vector>

namespace oskar {

/* Dependency tree node. */
class SettingsDependencyGroup
{
 public:
    enum GroupLogic { UNDEF = -1, AND, OR };

    SettingsDependencyGroup(const std::string& group_logic,
                            SettingsDependencyGroup* parent = 0);
    ~SettingsDependencyGroup();

    int num_children() const;
    SettingsDependencyGroup* add_child(const std::string& logic = "AND");
    SettingsDependencyGroup* get_child(int i);
    const SettingsDependencyGroup* get_child(int i) const;
    SettingsDependencyGroup* parent();

    int num_dependencies() const;
    GroupLogic group_logic() const;
    void add_dependency(const std::string& key, const std::string& value,
                        const std::string& logic = std::string());
    void add_dependency(const SettingsDependency& dep);
    const SettingsDependency* get_dependency(unsigned int i) const;

    static SettingsDependencyGroup::GroupLogic string_to_group_logic(
                    const std::string& s);

 private:
    SettingsDependencyGroup* parent_;
    std::vector<SettingsDependencyGroup*> children_;

    /* Node payload combination logic and dependencies. */
    SettingsKey group_key_;
    SettingsDependencyGroup::GroupLogic logic_;
    std::vector<SettingsDependency> dependencies_;
};

} /* namespace oskar */

#endif /* __cplusplus */

/* C interface. */
struct oskar_SettingsDependencyGroup;
#ifndef OSKAR_SETTINGS_DEPENDENCY_GROUP_TYPEDEF_
#define OSKAR_SETTINGS_DEPENDENCY_GROUP_TYPEDEF_
typedef struct oskar_SettingsDependencyGroup oskar_SettingsDependencyGroup;
#endif /* OSKAR_SETTINGS_DEPENDENCY_GROUP_TYPEDEF_ */


#endif /* OSKAR_SETTINGS_DEPENDENCY_GROUP_HPP_ */
