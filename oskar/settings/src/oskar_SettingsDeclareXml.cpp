/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_SettingsDeclareXml.h"
#include "settings/oskar_SettingsDependency.h"
#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_SettingsTree.h"
#include "settings/oskar_settings_utility_string.h"
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>

using namespace std;
using namespace oskar;

// Begin anonymous namespace for file-local helper functions.
namespace {

// File-local typedefs.
typedef rapidxml::xml_document<> doc_t;
typedef rapidxml::xml_node<> node;
typedef rapidxml::xml_attribute<> attr;

map<string, string> get_attributes(node* n)
{
    map<string, string> attrib;
    for (attr* a = n->first_attribute(); a; a = a->next_attribute())
    {
        attrib[string(a->name())] = string(a->value());
    }
    return attrib;
}

node* get_child_node(node* parent, const vector<string>& possible_names)
{
    for (size_t i = 0; i < possible_names.size(); ++i)
    {
        node* n = parent->first_node(possible_names[i].c_str());
        if (n) return n;
    }
    return 0;
}

node* get_node(const doc_t& doc, const string& search_key,
        string parent_key, node* parent = 0)
{
    if (!parent) parent = doc.first_node("root");
    for (node* s = parent->first_node(); s; s = s->next_sibling())
    {
        string key;
        string name = s->name();
        if (name != "s" && name != "setting" && name != "group") continue;

        // Get node attributes (the key must be defined).
        map<string, string> a = get_attributes(s);
        if (key.empty()) key = a["key"];
        if (key.empty()) key = a["k"];
        if (key.empty()) continue;

        // Check if keys match.
        string current_key = parent_key + key;
        if (search_key == current_key) return s;

        // Otherwise keep searching.
        if (s->first_node())
        {
            node* child = get_node(doc, search_key, current_key + '/', s);
            if (child) return child;
        }
    }
    return 0;
}

string get_description(node* s)
{
    // Obtain a pointer to the description node.
    vector<string> names;
    names.push_back("description");
    names.push_back("desc");
    node* n = get_child_node(s, names);
    if (!n) return "";

    // Convert the node to a string.
    stringstream ss;
    ss << *n;
    string desc = ss.str();

    //cout << "[" << desc << "]" << endl;

    // Remove the opening description tag and excess whitespace.
    oskar_settings_utility_string_replace(desc, "<description>", " ");
    oskar_settings_utility_string_replace(desc, "<desc>", " ");
    oskar_settings_utility_string_replace(desc, "</description>", " ");
    oskar_settings_utility_string_replace(desc, "</desc>", " ");
    desc = oskar_settings_utility_string_trim(desc);
    desc = oskar_settings_utility_string_trim(desc, " \n");
    desc = oskar_settings_utility_string_reduce(desc);
    desc = oskar_settings_utility_string_reduce(desc, " ", " \n");
    oskar_settings_utility_string_replace(desc, "&amp;", "&");

    return desc;
}

string get_label(node* s)
{
    // Obtain a pointer to the label node.
    vector<string> names;
    names.push_back("label");
    names.push_back("l");
    node* n = get_child_node(s, names);
    if (!n) return "";

    // Convert the node to a string.
    stringstream ss;
    ss << *n;
    string label = ss.str();

    // Remove the opening label tag and excess whitespace.
    oskar_settings_utility_string_replace(label, "<label>", " ");
    oskar_settings_utility_string_replace(label, "<l>", " ");
    oskar_settings_utility_string_replace(label, "</label>", " ");
    oskar_settings_utility_string_replace(label, "</l>", " ");
    label = oskar_settings_utility_string_trim(label);
    label = oskar_settings_utility_string_trim(label, " \n");
    label = oskar_settings_utility_string_reduce(label);
    label = oskar_settings_utility_string_reduce(label, " ", " \n");

    return label;
}

bool add_setting_deps(node* s, oskar::SettingsTree* settings)
{
    for (node* n = s->first_node(); n; n = n->next_sibling())
    {
        // Only proceed if this is a recognised node.
        string name = n->name();
        bool dependency_node = (name == "depends" || name == "d");
        bool logic_node = (name == "logic" || name == "l");
        if (!dependency_node && !logic_node) continue;

        // Get attributes.
        string g, k, c, v;
        map<string, string> a = get_attributes(n);
        if (g.empty()) g = a["group"];
        if (g.empty()) g = a["g"];
        if (k.empty()) k = a["key"];
        if (k.empty()) k = a["k"];
        if (c.empty()) c = a["condition"];
        if (c.empty()) c = a["c"];
        if (v.empty()) v = a["value"];
        if (v.empty()) v = a["v"];

        // Dependency node (has key, condition, value defined).
        if (dependency_node)
        {
            if (!settings->add_dependency(k.c_str(), v.c_str(), c.c_str()))
            {
                return false;
            }
        }

        // Logic node (will only have group defined, if that).
        if (logic_node)
        {
            // Process any child nodes.
            if (!settings->begin_dependency_group(g.c_str())) return false;
            if (!add_setting_deps(n, settings)) return false;
            settings->end_dependency_group();
        }
    }
    return true;
}


bool declare_setting(node* s, const string& key_root,
        const string& key_leaf, bool required, int priority,
        oskar::SettingsTree* settings)
{
    // Get a pointer to the type node.
    string key, type_name, default_value, options;
    vector<string> names;
    names.push_back(string("type"));
    names.push_back(string("t"));
    node* type_node = get_child_node(s, names);
    if (type_node)
    {
        // Get attributes.
        map<string, string> a = get_attributes(type_node);
        if (type_name.empty()) type_name = a["name"];
        if (type_name.empty()) type_name = a["n"];
        if (type_name.empty()) return false;
        if (default_value.empty()) default_value = a["default"];
        if (default_value.empty()) default_value = a["d"];

        // Get any type parameters.
        options = type_node->value();
        if (!options.empty()) {
            options = oskar_settings_utility_string_trim(options);
        }
    }

    // Add the setting.
    key = key_root + key_leaf;
    if (!settings->add_setting(key.c_str(), get_label(s).c_str(),
            get_description(s).c_str(), type_name.c_str(),
            default_value.c_str(), options.c_str(), required, priority))
    {
        return false;
    }

    // Add any dependencies.
    if (!add_setting_deps(s, settings)) return false;

    return true;
}

bool iterate_settings(oskar::SettingsTree* settings, const doc_t& doc,
        node* n = 0, const string key_root = "")
{
    if (!n) n = doc.first_node("root");
    for (node* s = n->first_node(); s; s = s->next_sibling())
    {
        // Get the node name and check its type.
        string name = s->name();
        if (name == "s" || name == "setting")
        {
            string key, req, pri;
            bool required = false, ok = false;
            int priority = 0;

            // Get setting attributes (the key must be defined).
            map<string, string> a = get_attributes(s);
            if (key.empty()) key = a["key"];
            if (key.empty()) key = a["k"];
            if (key.empty()) continue;
            if (req.empty()) req = a["required"];
            if (req.empty()) req = a["req"];
            if (req.empty()) req = a["r"];
            if (pri.empty()) pri = a["priority"];
            if (pri.empty()) pri = a["p"];
            if (!req.empty())
            {
                for (size_t i = 0; i < req.length(); ++i)
                {
                    req[i] = toupper(req[i]);
                }
                required = (req == "TRUE" || req == "YES");
            }
            if (!pri.empty())
            {
                priority = oskar_settings_utility_string_to_int(pri, &ok);
            }

            // Declare the setting.
            if (!declare_setting(s, key_root, key,
                    required, priority, settings))
            {
                cerr << "ERROR: Problem reading setting: " << key << endl;
                return false;
            }

            // Read any child settings.
            if (s->first_node())
            {
                if (!iterate_settings(settings, doc, s, key_root + key + "/"))
                {
                    return false;
                }
            }
        }
        else if (name == "import")
        {
            string group;

            // Get setting attributes (the group must be defined).
            map<string, string> a = get_attributes(s);
            if (group.empty()) group = a["group"];
            if (group.empty()) continue;

            node* import_group = get_node(doc, group, "");
            if (import_group)
            {
                if (!iterate_settings(settings, doc, import_group, key_root))
                {
                    return false;
                }
            }
            else
            {
                cerr << "ERROR: Could not find group: " << group << endl;
                return false;
            }
        }
    }
    return true;
}

} // End anonymous namespace.

bool oskar_settings_declare_xml(SettingsTree* settings, const char* xml)
{
    if (!settings) return false;
    doc_t doc;
    string xml_copy(xml);
    doc.parse<0>(&xml_copy[0]);

    settings->clear();
    return iterate_settings(settings, doc);
}
