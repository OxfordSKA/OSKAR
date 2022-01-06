/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_SettingsTree.h"
#include "settings/oskar_SettingsDependency.h"
#include "settings/oskar_SettingsDependencyGroup.h"
#include "settings/oskar_SettingsFileHandler.h"
#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_SettingsNode.h"
#include "settings/oskar_SettingsValue.h"
#include "settings/oskar_settings_utility_string.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace oskar;

namespace oskar {

struct SettingsTreePrivate
{
    SettingsNode* root_;
    SettingsFileHandler* file_handler_;
    SettingsItem* current_node_;
    char sep_;
    int num_items_, num_settings_;
    string group_prefix_;
    vector<string> failed_keys, failed_values, group_;
};

SettingsTree::SettingsTree(char key_separator)
{
    p = new SettingsTreePrivate;
    p->root_ = 0;
    p->file_handler_ = 0;
    p->current_node_ = 0;
    p->sep_ = key_separator;
    p->num_items_ = 0;
    p->num_settings_ = 0;
    modified_ = false;
    clear();
}

SettingsTree::~SettingsTree()
{
    delete p->root_;
    if (p->file_handler_) delete p->file_handler_;
    delete p;
}

bool SettingsTree::add_dependency(const char* dependency_key,
                                  const char* value,
                                  const char* logic)
{
    return p->current_node_ ?
            p->current_node_->add_dependency(dependency_key, value, logic) :
            false;
}

void SettingsTree::add_failed(const char* key, const char* value)
{
    p->failed_keys.push_back(string(key));
    p->failed_values.push_back(string(value));
}

bool SettingsTree::add_setting(const char* key,
                               const char* label,
                               const char* description,
                               const char* type_name,
                               const char* type_default,
                               const char* type_parameters,
                               bool required,
                               int priority)
{
    string current_key = p->group_prefix_ + key;
    SettingsKey k(current_key.c_str(), p->sep_);
    SettingsNode* parent = p->root_;
    string sub_key;
    string type_default_ = type_default ? string(type_default) : string();
    string type_name_ = type_name ? string(type_name) : string();
    if (required && !type_default_.empty())
    {
        cerr << "ERROR: Invalid setting key '" << k << "'" << endl;
        cerr << "ERROR: Required settings cannot have a default." << endl;
        return false;
    }
    for (int i = 0; i < k.depth(); ++i)
    {
        sub_key += k[i];
        SettingsNode* child = parent->child(sub_key.c_str());
        if (child)
        {
            parent = child;
        }
        else
        {
            p->num_items_++;
            parent = parent->add_child(
                    new SettingsNode(sub_key.c_str(), sub_key.c_str(), ""));
        }
        sub_key += p->sep_;
    }
    SettingsNode* new_item = new SettingsNode(k,
            label, description, type_name, type_default, type_parameters,
            required, priority);
    if (!type_name_.empty() &&
            new_item->settings_value().type() == SettingsValue::UNDEF)
    {
        delete new_item;
        return false;
    }
    p->current_node_ = parent->add_child(new_item);
    p->num_items_++;
    if (new_item->item_type() == SettingsItem::SETTING)
    {
        p->num_settings_++;
    }
    return true;
}

bool SettingsTree::begin_dependency_group(const char* logic)
{
    return p->current_node_ ?
            p->current_node_->begin_dependency_group(logic) : false;
}

void SettingsTree::begin_group(const char* name)
{
    if (!name) return;
    p->group_.push_back(name);
    p->group_prefix_.clear();
    for (unsigned i = 0; i < p->group_.size(); ++i)
    {
        p->group_prefix_ += p->group_[i] + p->sep_;
    }
}

void SettingsTree::clear()
{
    if (p->root_) delete p->root_;
    p->root_ = new SettingsNode;
    p->num_items_ = 0;
    p->num_settings_ = 0;
    modified_ = false;
    clear_group();
}

void SettingsTree::clear_group()
{
    p->group_.clear();
    p->group_prefix_.clear();
}

bool SettingsTree::contains(const char* key) const
{
    string current_key = p->group_prefix_ + key;
    SettingsKey k(current_key.c_str(), p->sep_);
    return find_node(p->root_, k, 0) ? true : false;
}

bool SettingsTree::dependencies_satisfied(const char* key) const
{
    string current_key = p->group_prefix_ + key;
    SettingsKey k(current_key.c_str(), p->sep_);
    const SettingsNode* node = find_node(p->root_, k, 0);
    if (node)
    {
        const SettingsNode* parent = node->parent();
        if (parent && !parent_dependencies_satisfied(parent))
        {
            return false;
        }
        const SettingsDependencyGroup* deps_ = node->dependency_tree();
        if (!deps_) return true;
        return dependencies_satisfied(deps_);
    }
    else
    {
        cerr << "ERROR: failed to find node with key = '" << key << "'" << endl;
        return 0;
    }
}

void SettingsTree::end_dependency_group()
{
    if (p->current_node_) p->current_node_->end_dependency_group();
}

void SettingsTree::end_group()
{
    p->group_.pop_back();
    p->group_prefix_.clear();
    for (unsigned i = 0; i < p->group_.size(); ++i)
    {
        p->group_prefix_ += p->group_[i] + p->sep_;
    }
}

const char* SettingsTree::failed_key(int i) const
{
    return i < (int) p->failed_keys.size() ? p->failed_keys[i].c_str() : 0;
}

const char* SettingsTree::failed_key_value(int i) const
{
    return i < (int) p->failed_values.size() ? p->failed_values[i].c_str() : 0;
}

const SettingsFileHandler* SettingsTree::file_handler() const
{
    return p->file_handler_;
}

SettingsFileHandler* SettingsTree::file_handler()
{
    return p->file_handler_;
}

const char* SettingsTree::file_name() const
{
    return p->file_handler_ ? p->file_handler_->file_name() : 0;
}

char SettingsTree::first_letter(const char* key, int* status) const
{
    if (*status) return 0;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return 0;}
    string s = v->to_string();
    return s.size() > 0 ? toupper(s[0]) : 0;
}

void SettingsTree::free(SettingsTree* h)
{
    if (h) delete h;
}

const char* SettingsTree::group_prefix() const
{
    return p->group_prefix_.c_str();
}

// Returns true if this node or any child nodes are required, but not set
// and also have all their dependencies satisfied.
bool SettingsTree::is_critical(const char* key) const
{
    string current_key = p->group_prefix_ + key;
    SettingsKey k(current_key.c_str(), p->sep_);
    const SettingsNode* node = find_node(p->root_, k, 0);
    bool item_critical = false;
    if (node->item_type() == SettingsItem::SETTING &&
                    dependencies_satisfied(k))
    {
        if (node->is_required() && !node->is_set())
        {
            item_critical = true;
        }
    }
    bool critical_children = is_node_critical(node);
    return (item_critical || critical_children);
}

bool SettingsTree::is_modified() const
{
    return modified_;
}

const SettingsItem* SettingsTree::item(const char* key) const
{
    string current_key = p->group_prefix_ + key;
    SettingsKey k(current_key.c_str(), p->sep_);
    const SettingsNode* node = find_node(p->root_, k, 0);
    if (node)
    {
        return node;
    }
    else
    {
        cerr << "ERROR: failed to find node with key = '" << key << "'" << endl;
        return 0;
    }
}

bool SettingsTree::load(const char* file_name)
{
    if (!p->file_handler_) return false;
    if (file_name) p->file_handler_->set_file_name(file_name);
    set_defaults();
    p->failed_keys.clear();
    p->failed_values.clear();
    bool ok = p->file_handler_->read_all(this);
    if (ok) modified_ = false;
    return ok;
}

int SettingsTree::num_failed_keys() const
{
    return (int) (p->failed_keys.size());
}

int SettingsTree::num_items() const
{
    return p->num_items_;
}

int SettingsTree::num_settings() const
{
    return p->num_settings_;
}

void SettingsTree::print() const
{
    cout << string(80, '-') << endl;
    print_from_node(p->root_, 0);
    cout << string(80, '-') << endl;
}

const SettingsNode* SettingsTree::root_node() const
{
    return p->root_;
}

bool SettingsTree::save(const char* file_name) const
{
    if (!p->file_handler_) return false;
    if (file_name) p->file_handler_->set_file_name(file_name);
    bool ok = p->file_handler_->write_all(this);
    if (ok) modified_ = false;
    return ok;
}

char SettingsTree::separator() const
{
    return p->sep_;
}

const SettingsValue* SettingsTree::settings_value(const char* key) const
{
    const SettingsItem* i = item(key);
    return i ? &i->settings_value() : 0;
}

bool SettingsTree::set_default(const char* key, bool write)
{
    string current_key = p->group_prefix_ + key;
    SettingsKey k(current_key.c_str(), p->sep_);
    SettingsNode* node = find_node(p->root_, k, 0);
    if (node)
    {
        bool item_ok = node->set_value(node->default_value());
        if (item_ok)
        {
            bool write_ok = false;
            if (write && p->file_handler_)
            {
                write_ok = p->file_handler_->write_all(this);
            }
            modified_ = !write_ok;
        }
        return item_ok;
    }
    return false;
}

void SettingsTree::set_defaults()
{
    set_defaults_from_node(p->root_);
    modified_ = false;
}

void SettingsTree::set_file_handler(SettingsFileHandler* handler,
        const char* name)
{
    if (p->file_handler_) delete p->file_handler_;
    p->file_handler_ = handler;
    set_file_name(name);
}

void SettingsTree::set_file_name(const char* name)
{
    if (p->file_handler_ && name && strlen(name) > 0)
    {
        p->file_handler_->set_file_name(name);
    }
}

bool SettingsTree::set_value(const char* key, const char* value, bool write)
{
    string current_key = p->group_prefix_ + key;
    SettingsKey k(current_key.c_str(), p->sep_);
    SettingsNode* node = find_node(p->root_, k, 0);
    if (node)
    {
        bool item_ok = node->set_value(value);
        if (item_ok)
        {
            bool write_ok = false;
            if (write && p->file_handler_)
            {
                write_ok = p->file_handler_->write_all(this);
            }
            modified_ = !write_ok;
        }
        return item_ok;
    }
    return false;
}

bool SettingsTree::set_values(int num_strings, const char* const* list,
        bool write)
{
    if (num_strings % 2 != 0) return false;
    const int num_items = num_strings / 2;
    bool ok = true;
    for (int i = 0; i < num_items || (num_items == 0); ++i)
    {
        const char* key = list[2 * i];
        const char* value = list[2 * i + 1];
        if (!key || strlen(key) == 0) break;
        bool item_ok = set_value(key, value, write);
        if (!item_ok)
        {
            add_failed(key, value);
        }
        ok &= item_ok;
    }
    return ok;
}

bool SettingsTree::starts_with(const char* key, const char* str,
        int* status) const
{
    if (*status) return false;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return false;}
    string s = v->to_string();
    return oskar_settings_utility_string_starts_with(s, str, false);
}

double SettingsTree::to_double(const char* key, int* status) const
{
    double t = 0.0;
    if (*status) return t;
    bool ok = false;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return t;}
    t = v->to_double(ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_DOUBLE_CONVERSION_FAIL;
    return t;
}

const double* SettingsTree::to_double_list(const char* key, int* size,
        int* status) const
{
    const double* t = 0;
    *size = 0;
    if (*status) return t;
    bool ok = false;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return t;}
    t = v->to_double_list(size, ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_DOUBLE_LIST_CONVERSION_FAIL;
    return t;
}

int SettingsTree::to_int(const char* key, int* status) const
{
    int t = 0;
    if (*status) return t;
    bool ok = false;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return t;}
    t = v->to_int(ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_INT_CONVERSION_FAIL;
    return t;
}

const int* SettingsTree::to_int_list(const char* key, int* size,
        int* status) const
{
    const int* t = 0;
    *size = 0;
    if (*status) return t;
    bool ok = false;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return t;}
    t = v->to_int_list(size, ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_INT_LIST_CONVERSION_FAIL;
    return t;
}

const char* SettingsTree::to_string(const char* key, int* status) const
{
    if (*status) return 0;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return 0;}
    return v->to_string();
}

const char* const* SettingsTree::to_string_list(const char* key, int* size,
        int* status) const
{
    const char* const* t = 0;
    *size = 0;
    if (*status) return t;
    bool ok = false;
    const SettingsValue* v = settings_value(key);
    if (!v) {*status = OSKAR_ERR_SETTINGS_NO_VALUE; return t;}
    t = v->to_string_list(size, ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_STRING_LIST_CONVERSION_FAIL;
    return t;
}

const char* SettingsTree::operator[](const char* key) const
{
    const SettingsValue* v = settings_value(key);
    return v ? v->get_value() : "";
}


/* Private methods ***********************************************************/

bool SettingsTree::dependencies_satisfied(const SettingsDependencyGroup* group) const
{
    SettingsDependencyGroup::GroupLogic logic = group->group_logic();
    bool group_ok = (logic == SettingsDependencyGroup::OR) ? false : true;
    // Loop over child groups and evaluate if satisfied.
    for (int i = 0; i < group->num_children(); ++i)
    {
        bool child_ok = dependencies_satisfied(group->get_child(i));
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
        bool dep_ok = dependency_satisfied(group->get_dependency(i));
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

bool SettingsTree::dependency_satisfied(const SettingsDependency* dep) const
{
    SettingsKey key(dep->key()); // key pointed to.
    // Current value of the dependency.
    const SettingsNode* node = find_node(p->root_, key, 0);
    if (!node)
    {
        cerr << "ERROR: Unable to find dependency key = '" << key << "'" << endl;
        return true;
    }
    const SettingsValue& current_value =
            find_node(p->root_, key, 0)->settings_value();

    // Construct the value required by the dependency.
    SettingsValue dep_value(current_value);
    dep_value.set_value(dep->value());
    switch (dep->logic())
    {
        case SettingsDependency::EQ: return (current_value == dep_value);
        case SettingsDependency::NE: return (current_value != dep_value);
        case SettingsDependency::GT: return (current_value > dep_value);
        case SettingsDependency::GE: return (current_value >= dep_value);
        case SettingsDependency::LT: return (current_value < dep_value);
        case SettingsDependency::LE: return (current_value <= dep_value);
        default: return false;
    }
}

const SettingsNode* SettingsTree::find_node(const SettingsNode* node,
        const SettingsKey& full_key, int depth) const
{
    string current_key = string(full_key[depth]);
    for (int i = 0; i < node->num_children(); ++i)
    {
        const SettingsNode* child = node->child(i);
        string leaf = string(child->settings_key().back());
        if (oskar_settings_utility_string_to_upper(leaf) ==
                oskar_settings_utility_string_to_upper(current_key))
        {
            if (child->settings_key() == full_key)
            {
                return child;
            }
            else
            {
                return find_node(child, full_key, depth + 1);
            }
        }
    }
    return 0;
}

SettingsNode* SettingsTree::find_node(SettingsNode* node,
        const SettingsKey& full_key, int depth)
{
    string current_key = string(full_key[depth]);
    for (int i = 0; i < node->num_children(); ++i)
    {
        SettingsNode* child = node->child(i);
        string leaf = string(child->settings_key().back());
        if (oskar_settings_utility_string_to_upper(leaf) ==
                oskar_settings_utility_string_to_upper(current_key))
        {
            if (child->settings_key() == full_key)
            {
                return child;
            }
            else
            {
                return find_node(child, full_key, depth + 1);
            }
        }
    }
    return 0;
}

bool SettingsTree::is_node_critical(const SettingsNode* node) const
{
    bool critical = false;
    for (int i = 0; i < node->num_children(); ++i)
    {
        const SettingsNode* child = node->child(i);
        bool is_satisfied = dependencies_satisfied(child->key());
        critical |= (child->is_required() && !child->is_set() && is_satisfied);
        // If a condition is found, no need to continue.
        if (critical) return true;
        critical |= is_node_critical(child);
    }
    return critical;
}

bool SettingsTree::parent_dependencies_satisfied(const SettingsNode* node) const
{
    // If the parent has any dependencies not satisfied return false.
    const SettingsDependencyGroup* deps_ = node->dependency_tree();
    if (deps_)
    {
        if (!dependencies_satisfied(deps_))
        {
            return false;
        }
    }
    // Otherwise keep going up to the next parent.
    if (node->parent())
    {
        return parent_dependencies_satisfied(node->parent());
    }
    return true;
}

void SettingsTree::print_from_node(const SettingsNode* node, int depth) const
{
    for (int i = 0; i < node->num_children(); ++i)
    {
        const SettingsNode* child = node->child(i);
        const char* key = child->key();
        cout << left;
        cout << string(depth * 2, ' ');
        cout << setw(80 - depth * 2) << key;
        cout << ". ";
        switch (child->item_type())
        {
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
        cout << setw(16) << child->settings_value().type_name();
        cout << setw(10) << "deps:" << child->num_dependencies();
        cout << endl;
        print_from_node(child, depth + 1);
    }
}

void SettingsTree::set_defaults_from_node(SettingsNode* node)
{
    for (int i = 0; i < node->num_children(); ++i)
    {
        SettingsNode* child_node = node->child(i);
        child_node->set_value(child_node->default_value());
        set_defaults_from_node(child_node);
    }
}

} // namespace oskar
