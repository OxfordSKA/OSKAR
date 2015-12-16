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

#include <oskar_SettingsTree.hpp>
#include <oskar_SettingsFileHandler.hpp>
#include <oskar_SettingsItem.hpp>
#include <oskar_SettingsKey.hpp>
#include <oskar_SettingsDependencyGroup.hpp>
#include <oskar_SettingsDependency.hpp>
#include <oskar_settings_utility_string.hpp>
#include <oskar_version.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace oskar;

namespace oskar {

SettingsTree::SettingsTree(char key_separator)
: root_(0), file_handler_(0), current_node_(0), sep_(key_separator),
  num_items_(0), num_settings_(0), modified_(0)
{
    clear();
}

SettingsTree::~SettingsTree()
{
    delete root_;
}

void SettingsTree::set_file_handler(SettingsFileHandler* handler,
        const string& name)
{
    file_handler_ = handler;
    set_file_name(name);
}

void SettingsTree::set_file_name(const std::string& name)
{
    if (file_handler_ && !name.empty())
        file_handler_->set_file_name(name);
}

void SettingsTree::set_defaults()
{
    set_defaults_(root_);
}

bool SettingsTree::is_modified() const
{
    return modified_;
}

void SettingsTree::begin_group(const string& name)
{
    group_.push_back(name);
}

void SettingsTree::end_group()
{
    group_.pop_back();
}

string SettingsTree::group_prefix() const
{
    string temp;
    if (group_.empty()) return temp;
    for (unsigned i = 0; i < group_.size(); ++i) temp += group_.at(i) + sep_;
    return temp;
}

bool SettingsTree::add_setting(const string& key,
                               const string& label,
                               const string& description,
                               const string& type_name,
                               const string& type_default,
                               const string& type_parameters,
                               bool required)
{
    SettingsKey k(group_prefix()+ key, sep_);
    SettingsNode* parent = root_;
    string sub_key;
    if (required && !type_default.empty()) {
        cerr << "ERROR: Invalid setting key '" << k << "'" << endl;
        cerr << "ERROR: Required settings cannot have a default." << endl;
        return false;
    }
    for (int i = 0; i < k.depth(); ++i) {
        sub_key += k[i];
        SettingsNode* child = parent->child(sub_key);
        if (child) {
            parent = child;
        }
        else {
            num_items_++;
            SettingsNode group_label(sub_key, sub_key, "");
            parent = parent->add_child(group_label);
        }
        sub_key += "/";
    }
    SettingsKey new_key(group_prefix()+ key, sep_);
    SettingsNode new_item(new_key, label, description, type_name,
                          type_default, type_parameters, required);
    if (!type_name.empty() && new_item.value().type() == SettingsValue::UNDEF)
        return false;
    current_node_ = parent->add_child(new_item);
    num_items_++;
    if (new_item.item_type() == SettingsItem::SETTING)
        num_settings_++;
    return true;
}


bool SettingsTree::begin_dependency_group(const string& logic)
{
    return current_node_ ? current_node_->begin_dependency_group(logic) : false;
}

void SettingsTree::end_dependency_group()
{
    if (current_node_)
        current_node_->end_dependency_group();
}

bool SettingsTree::add_dependency(const string& dependency_key,
                                  const string& value,
                                  const string& logic)
{
    return current_node_ ?
            current_node_->add_dependency(dependency_key, value, logic) : false;
}

bool SettingsTree::set_value(const string& key, const string& value, bool write)
{
    SettingsKey k(group_prefix() + key, sep_);
    SettingsNode* node = find_(root_, k, 0);
    if (node)
    {
        bool item_ok = node->set_value(value);
        if (item_ok)
        {
            bool write_ok = false;
            if (write && file_handler_)
                write_ok = file_handler_->write_all(this);
            modified_ = !write_ok;
        }
        return item_ok;
    }
    else
        return false;
}

const SettingsItem* SettingsTree::item(const string& key) const
{
    SettingsKey k(group_prefix() + key, sep_);
    const SettingsNode* node = find_(root_, k, 0);
    if (node) {
        return node;
    }
    else {
        cerr << "ERROR: failed to find node with key = '" << key << "'" << endl;
        return 0;
    }
}

const SettingsValue* SettingsTree::value(const string& key) const
{
    const SettingsItem* i = item(key);
    return i ? &i->value() : 0;
}

const SettingsValue* SettingsTree::operator[](const string& key) const
{
    SettingsKey k(group_prefix() + key, sep_);
    return value(k);
}

int SettingsTree::num_items() const
{
    return num_items_;
}

int SettingsTree::num_settings() const
{
    return num_settings_;
}

void SettingsTree::print() const
{
    cout << string(80, '-') << endl;
    print_(root_, 0);
    cout << string(80, '-') << endl;
}

void SettingsTree::clear()
{
    if (root_) delete root_;
    root_ = new SettingsNode;
    num_items_ = 0;
    num_settings_ = 0;
    modified_ = false;
}


bool SettingsTree::save(const string& file_name) const
{
    if (!file_handler_) return false;
    if (!file_name.empty()) file_handler_->set_file_name(file_name);
    bool ok = file_handler_->write_all(this);
    if (ok) modified_ = false;
    return ok;
}

bool SettingsTree::load(vector<pair<string, string> >& invalid,
        const string& file_name)
{
    if (!file_handler_) return false;
    if (!file_name.empty()) file_handler_->set_file_name(file_name);
    set_defaults();
    bool ok = file_handler_->read_all(this, invalid);
    if (ok) modified_ = false;
    return ok;
}

string SettingsTree::file_version() const
{
    if (!file_handler_) return string();
    return file_handler_->file_version();
}

bool SettingsTree::contains(const string& key) const
{
    SettingsKey k(group_prefix() + key, sep_);
    return find_(root_, k, 0) ? true : false;
}

bool SettingsTree::dependency_satisfied_(const SettingsDependency* dep) const
{
    SettingsKey key(dep->key()); // key pointed to.
    // Current value of the dependency.
    const SettingsNode* node = find_(root_, key, 0);
    if (!node) {
        cerr << "ERROR: Unable to find dependency key = '" << key << "'" << endl;
        return true;
    }
    const SettingsValue& current_value = find_(root_, key, 0)->value();
    // Construct the value required by the dependency.
    SettingsValue dep_value(current_value);
    dep_value.set_value(dep->value());

    switch (dep->logic())
    {
        case SettingsDependency::EQ:
            return (current_value == dep_value);
        case SettingsDependency::NE:
            return (current_value != dep_value);
        case SettingsDependency::GT:
            return (current_value > dep_value);
        case SettingsDependency::GE:
            return (current_value >= dep_value);
        case SettingsDependency::LT:
            return (current_value < dep_value);
        case SettingsDependency::LE:
            return (current_value <= dep_value);
        default:
            return false;
    }
}

bool SettingsTree::dependencies_satisfied_(const SettingsDependencyGroup* group) const
{
    SettingsDependencyGroup::GroupLogic logic = group->group_logic();
    bool group_ok = true;
    if (logic == SettingsDependencyGroup::OR) {
        group_ok = false;
    }
    // Loop over child groups and evaluate if satisfied.
    for (int i = 0; i < group->num_children(); ++i)
    {
        bool child_ok = dependencies_satisfied_(group->get_child(i));
        switch (logic) {
            case SettingsDependencyGroup::AND:
                group_ok &= child_ok;
                break;
            case SettingsDependencyGroup::OR:
                group_ok |= child_ok;
                break;
            default:
                return false;
        }
    }
    // Loop over dependencies and evaluate if satisfied.
    for (int i = 0; i < group->num_dependencies(); ++i)
    {
        bool dep_ok = dependency_satisfied_(group->get_dependency(i));
        switch (logic) {
            case SettingsDependencyGroup::AND:
                group_ok &= dep_ok;
                break;
            case SettingsDependencyGroup::OR:
                group_ok |= dep_ok;
                break;
            default:
                return false;
        }
    }
    return group_ok;
}

bool SettingsTree::parent_dependencies_satisfied_(const SettingsNode* node) const
{
    // If the parent has any dependencies not satisfied return false.
    const SettingsDependencyGroup* deps_ = node->dependency_tree();
    if (deps_) {
        if (!dependencies_satisfied_(deps_)) {
            return false;
        }
    }
    // Otherwise keep going up to the next parent.
    if (node->parent()) {
        return parent_dependencies_satisfied_(node->parent());
    }
    return true;
}

bool SettingsTree::dependencies_satisfied(const string& key) const
{
    SettingsKey k(group_prefix() + key, sep_);
    const SettingsNode* node = find_(root_, k, 0);
    if (node) {
        const SettingsNode* parent = node->parent();
        if (parent && !parent_dependencies_satisfied_(parent)) {
            return false;
        }
        const SettingsDependencyGroup* deps_ = node->dependency_tree();
        if (!deps_) return true;
        return dependencies_satisfied_(deps_);
    }
    else {
        cerr << "ERROR: failed to find node with key = '" << key << "'" << endl;
        return 0;
    }
}

// Returns true if this node or any child nodes are required, but not set
// and also have all their dependencies satisfied.
bool SettingsTree::is_critical(const std::string& key) const
{
    SettingsKey k(group_prefix() + key, sep_);
    const SettingsNode* node = find_(root_, key, 0);
    bool item_critical = false;
    if (node->item_type() == SettingsItem::SETTING &&
                    dependencies_satisfied(k))
    {
        if (node->is_required() && !node->value().is_set())
            item_critical = true;
    }
    bool critical_children = is_critical_(node);
    return (item_critical || critical_children);
}

bool SettingsTree::is_critical_(const SettingsNode* node) const
{
    bool critical = false;
    for (int i = 0; i < node->num_children(); ++i) {
        const SettingsNode* child = node->child(i);
        bool is_satisfied = dependencies_satisfied(child->key());
        critical |= (child->is_required() && !child->value().is_set()
                        && is_satisfied);
        // If a condition is found, no need to continue.
        if (critical) return true;
        critical |= is_critical_(child);
    }
    return critical;
}

const SettingsNode* SettingsTree::root_node() const
{
    return root_;
}

const SettingsNode* SettingsTree::find_(const SettingsNode* node,
                                        const SettingsKey& full_key,
                                        int depth) const
{
    for (int i = 0; i < node->num_children(); ++i)
    {
        const SettingsNode* child = node->child(i);
        if (oskar_settings_utility_string_to_upper(child->key().back()) ==
                        oskar_settings_utility_string_to_upper(full_key[depth]))
        {
            if (child->key() == full_key)
                return child;
            else
                return find_(child, full_key, depth + 1);
        }
    }
    return 0;
}

SettingsNode* SettingsTree::find_(SettingsNode* node,
                                  const SettingsKey& full_key,
                                  int depth)
{
    for (int i = 0; i < node->num_children(); ++i)
    {
        SettingsNode* child = node->child(i);
        if (oskar_settings_utility_string_to_upper(child->key().back()) ==
                        oskar_settings_utility_string_to_upper(full_key[depth]))
        {
            if (child->key() == full_key)
                return child;
            else
                return find_(child, full_key, depth + 1);
        }
    }
    return 0;
}

void SettingsTree::print_(const SettingsNode* node, int depth) const
{
    for (int i = 0; i < node->num_children(); ++i) {
        const SettingsNode* child = node->child(i);
        string key = child->key();
        cout << left;
        cout << string(depth * 2, ' ');
        cout << setw(80 - depth * 2) << key;
        cout << ". ";
        switch (child->item_type()) {
            case SettingsItem::INVALID:
                cout << "I";
                break;
            case SettingsItem::SETTING:
                cout << "S";
                break;
            case SettingsItem::LABEL:
                cout << "L";
                break;
        };
        cout << " . ";
        // cout << setw(20) << child->item().label();
        cout << setw(16) << child->value().type_name();
        cout << setw(10) << "deps:" << child->num_dependencies();
        cout << endl;
        print_(child, depth + 1);
    }
}

void SettingsTree::set_defaults_(SettingsNode* node)
{
    for (int i = 0; i < node->num_children(); ++i)
    {
        SettingsNode* child_node = node->child(i);
        child_node->set_value(child_node->value().get_default());
        set_defaults_(child_node);
    }
}

} // namespace oskar


/* C interface. */
struct oskar_Settings : public oskar::SettingsTree
{
    oskar_Settings() : oskar::SettingsTree() {}
};


oskar_Settings* oskar_settings_create()
{
    return new oskar_Settings();
}

void oskar_settings_free(oskar_Settings* s)
{
    delete s;
}

void oskar_settings_set_file_handler(oskar_Settings* s, void* handler)
{
    s->set_file_handler((oskar::SettingsFileHandler*) handler, string());
}

void oskar_settings_set_file_name(oskar_Settings* s, const char* name)
{
    s->set_file_name(string(name));
}

void oskar_settings_begin_group(oskar_Settings* s,
        const char* name)
{
    s->begin_group(string(name));
}

void oskar_settings_end_group(oskar_Settings* s)
{
    s->end_group();
}

int oskar_settings_add_setting(oskar_Settings* s,
        const char* key, const char* label, const char* description,
        const char* type_name, const char* type_default,
        const char* type_parameters, int required)
{
    return (int) s->add_setting(string(key), string(label),
            string(description), string(type_name), string(type_default),
            string(type_parameters), (bool) required);
}

int oskar_settings_begin_dependency_group(oskar_Settings* s,
        const char* logic)
{
    return (int) s->begin_dependency_group(string(logic));
}

void oskar_settings_end_dependency_group(oskar_Settings* s)
{
    s->end_dependency_group();
}

int oskar_settings_add_dependency(oskar_Settings* s,
        const char* dependency_key, const char* value, const char* condition)
{
    return (int) s->add_dependency(string(dependency_key),
            string(value), string(condition));
}

int oskar_settings_set_value(oskar_Settings* s,
        const char* key, const char* value)
{
    return (int) s->set_value(string(key), string(value));
}

const oskar_SettingsValue* oskar_settings_value(
        const oskar_Settings* s, const char* key)
{
    return (const oskar_SettingsValue*) s->value(string(key));
}

int oskar_settings_starts_with(const oskar_Settings* s, const char* key,
        const char* str, int* status)
{
    return oskar_settings_value_starts_with(
            oskar_settings_value(s, key), str, status);
}

char oskar_settings_first_letter(const oskar_Settings* s, const char* key,
        int* status)
{
    return oskar_settings_value_first_letter(oskar_settings_value(s, key),
            status);
}

char* oskar_settings_to_string(const oskar_Settings* s,
        const char* key, int* status)
{
    return oskar_settings_value_string(oskar_settings_value(s, key), status);
}

int oskar_settings_to_int(const oskar_Settings* s,
        const char* key, int* status)
{
    return oskar_settings_value_to_int(oskar_settings_value(s, key), status);
}

double oskar_settings_to_double(const oskar_Settings* s,
        const char* key, int* status)
{
    return oskar_settings_value_to_double(oskar_settings_value(s, key), status);
}

char** oskar_settings_to_string_list(const oskar_Settings* s,
        const char* key, int* num, int* status)
{
    return oskar_settings_value_to_string_list(
            oskar_settings_value(s, key), num, status);
}

int* oskar_settings_to_int_list(const oskar_Settings* s,
        const char* key, int* num, int* status)
{
    return oskar_settings_value_to_int_list(
            oskar_settings_value(s, key), num, status);
}

double* oskar_settings_to_double_list(const oskar_Settings* s,
        const char* key, int* num, int* status)
{
    return oskar_settings_value_to_double_list(
            oskar_settings_value(s, key), num, status);
}

int oskar_settings_num_items(const oskar_Settings* s)
{
    return s->num_items();
}

int oskar_settings_num_settings(const oskar_Settings* s)
{
    return s->num_settings();
}

void oskar_settings_print(const oskar_Settings* s)
{
    s->print();
}

void oskar_settings_clear(oskar_Settings* s)
{
    s->clear();
}

void oskar_settings_save(const oskar_Settings* s, const char* file_name,
        int* status)
{
    if (!status || *status) return;
    bool ok = s->save(string(file_name));
    if (!ok)
        *status = OSKAR_ERR_SETTINGS_SAVE;
}

void oskar_settings_load(oskar_Settings* s, const char* file_name,
        int* num_failed, char*** failed_keys, int* status)
{
    if (!status || *status) return;
    vector<pair<string, string> > failed;
    bool ok = s->load(failed, string(file_name));
    if (!ok)
        *status = OSKAR_ERR_SETTINGS_LOAD;
    *num_failed = failed.size();
    if (*num_failed > 0)
    {
        *failed_keys = (char**) calloc(*num_failed, sizeof(char*));
        for (int i = 0; i < *num_failed; ++i)
        {
            (*failed_keys)[i] = (char*) calloc(1, 1 + failed[i].first.size());
            strcpy((*failed_keys)[i], failed[i].first.c_str());
        }
    }
}

int oskar_settings_contains(const oskar_Settings* s, const char* key)
{
    return (int) s->contains(string(key));
}

int oskar_settings_dependencies_satisfied(const oskar_Settings* s,
        const char* key)
{
    return (int) s->dependencies_satisfied(string(key));
}


