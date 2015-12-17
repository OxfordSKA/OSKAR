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

#ifndef OSKAR_SETTINGS_NODE_HPP_
#define OSKAR_SETTINGS_NODE_HPP_

/**
 * @file oskar_SettingsNode.hpp
 */

#include <oskar_SettingsItem.hpp>

#ifdef __cplusplus

#include <vector>
#include <string>

namespace oskar {

/**
 * @class SettingsNode
 *
 * @brief A node with the settings tree, inherits @class SettingsItem.
 *
 * @details
 * Settings tree node class which specialises a settings item for use within
 * a tree-like structure.
 */

class SettingsNode : public SettingsItem
{
 public:
    SettingsNode();
    SettingsNode(const SettingsNode& node, SettingsNode* parent = 0);
    SettingsNode(const std::string& key,
                 const std::string& label = std::string(),
                 const std::string& description = std::string(),
                 const std::string& type_name = std::string(),
                 const std::string& type_default = std::string(),
                 const std::string& type_parameters = std::string(),
                 bool is_required = false,
                 const std::string& priority = "DEFAULT");
    virtual ~SettingsNode();

    int num_children() const;
    SettingsNode* add_child(const SettingsNode& node);
    SettingsNode* child(int i);
    const SettingsNode* child(int i) const;
    SettingsNode* child(const std::string& key);
    const SettingsNode* child(const std::string& key) const;
    const SettingsNode* parent() const;
    int child_number() const;
    /* Returns true, if the item or one its children has a non-default value. */
    bool value_or_child_set() const;
    bool set_value(const std::string& value);

 private:
    /* Increment or decrementing the value set counter. */
    void update_value_set_counter_(bool increment_counter);
    SettingsNode* parent_;
    std::vector<SettingsNode*> children_;
    /* Counter used to determine if the item or its children have been set.
     * Incremented by 1 for each item or child with a value not at default. */
    int value_set_counter_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* C interface. */
struct oskar_SettingsNode;
#ifndef OSKAR_SETTINGS_NODE_TYPEDEF_
#define OSKAR_SETTINGS_NODE_TYPEDEF_
typedef struct oskar_SettingsNode oskar_SettingsNode;
#endif /* OSKAR_SETTINGS_NODE_TYPEDEF_ */

int oskar_settings_node_num_children(const oskar_SettingsNode* node);
int oskar_settings_node_begin_dependency_group(oskar_SettingsNode* node,
        const char* logic);
void oskar_settings_node_end_dependency_group(oskar_SettingsNode* node);
int oskar_settings_node_add_dependency(oskar_SettingsNode* node,
        const char* dependency_key, const char* value, const char* logic);

int oskar_settings_node_type(const oskar_SettingsNode* node);
const char* oskar_settings_node_key(const oskar_SettingsNode* node);
const char* oskar_settings_node_label(const oskar_SettingsNode* node);
const char* oskar_settings_node_description(const oskar_SettingsNode* node);
int oskar_settings_node_is_required(const oskar_SettingsNode* node);
int oskar_settings_node_set_value(oskar_SettingsNode* node,
        const char* value);
const oskar_SettingsValue* oskar_settings_node_value(
        const oskar_SettingsNode* node);

int oskar_settings_node_num_dependencies(const oskar_SettingsNode* node);
int oskar_settings_node_num_dependency_groups(const oskar_SettingsNode* node);
const oskar_SettingsDependencyGroup* oskar_settings_node_dependency_tree(
        const oskar_SettingsNode* node);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SETTINGS_NODE_HPP_ */
