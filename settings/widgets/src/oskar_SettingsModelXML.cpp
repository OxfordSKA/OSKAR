/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <oskar_SettingsModelXML.h>
#include "oskar_xml_all.h"
#include <oskar_SettingsItem.h>

#include <rapidxml_print.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using std::stringstream;
using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

///////////////////////////////////////////////////////////////////////////////

// Begin anonymous namespace for file-local helper functions.
namespace {

// File-local typedefs.
typedef rapidxml::xml_document<> doc_t;
typedef rapidxml::xml_node<> node;
typedef rapidxml::xml_attribute<> attr;

int key_index(const string& key, const vector<string>& keys)
{
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == key) return i;
    return -1;
}

string str_replace(string& s, string toReplace, string replaceWith)
{
    // FIXME clean up this logic!
    size_t p = 0;
    while (p != string::npos)
    {
        p = s.find(toReplace);
        if (p != string::npos)
            s.replace(p, toReplace.length(), replaceWith);
    }
    return s;
}

string str_trim(const string& str, const string& whitespace = " \t")
{
    const size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == string::npos) return "";
    const size_t strEnd = str.find_last_not_of(whitespace);
    const size_t strRange = strEnd - strBegin + 1;
    return str.substr(strBegin, strRange);
}

string str_reduce(const string& str, const string& fill = " ",
        const string& whitespace = " \t")
{
    // Trim first
    string result = str_trim(str, whitespace);

    // Replace sub ranges
    size_t beginSpace = result.find_first_of(whitespace);
    while (beginSpace != string::npos)
    {
        const size_t endSpace = result.find_first_not_of(whitespace, beginSpace);
        const size_t range = endSpace-beginSpace;
        result.replace(beginSpace, range, fill);
        const size_t newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }

    return result;
}

vector<string> str_get_options(const string& s)
{
    vector<string> opts;
    stringstream sin(s);
    string line;
    while (std::getline(sin, line, '"')) {
        stringstream ss(line);
        while (std::getline(ss, line, ',')) {
            line = str_trim(line, " \t");
            if (!line.empty()) opts.push_back(line);
        }
        if (std::getline(sin, line, '"')) {
            if (!line.empty()) opts.push_back(line);
        }
    }
    return opts;
}

string str_to_upper(const string& s)
{
    string s_(s);
    for (size_t i = 0; i < s_.length(); ++i) s_[i] = toupper(s_[i]);
    return s_;
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

attr* get_attribute(node* n, const vector<string>& possible_names)
{
    for (size_t i = 0; i < possible_names.size(); ++i)
    {
        attr* a = n->first_attribute(possible_names[i].c_str());
        if (a) return a;
    }
    return 0;
}

oskar_SettingsItem::type_id get_oskar_SettingsItem_type(const string& name,
        const vector<string>& params)
{
    string name_(name);
    for (size_t i = 0; i < name_.length(); ++i) name_[i] = toupper(name_[i]);
    oskar_SettingsItem::type_id tid = oskar_SettingsItem::UNDEF;

    if      (name_ == "BOOL")           tid = oskar_SettingsItem::BOOL;
    else if (name_ == "INT")            tid = oskar_SettingsItem::INT;
    else if (name_ == "UINT")           tid = oskar_SettingsItem::INT_UNSIGNED;
    else if (name_ == "INTPOSITIVE")    tid = oskar_SettingsItem::INT_POSITIVE;
    else if (name_ == "INTLIST")        tid = oskar_SettingsItem::INT_CSV_LIST;
    else if (name_ == "INTLISTEXT")     tid = oskar_SettingsItem::INT_CSV_LIST;
    else if (name_ == "RANDOMSEED")     tid = oskar_SettingsItem::RANDOM_SEED;
    else if (name_ == "INPUTFILELIST")  tid = oskar_SettingsItem::INPUT_FILE_LIST;
    else if (name_ == "DOUBLE")         tid = oskar_SettingsItem::DOUBLE;
    else if (name_ == "DOUBLERANGE")    tid = oskar_SettingsItem::DOUBLE;
    else if (name_ == "DOUBLELIST")     tid = oskar_SettingsItem::DOUBLE_CSV_LIST;
    else if (name_ == "UNSIGNEDDOUBLE") tid = oskar_SettingsItem::DOUBLE;
    else if (name_ == "OPTIONLIST")     tid = oskar_SettingsItem::OPTIONS;
    else if (name_ == "INPUTFILE")      tid = oskar_SettingsItem::INPUT_FILE_NAME;
    else if (name_ == "OUTPUTFILE")     tid = oskar_SettingsItem::OUTPUT_FILE_NAME;
    else if (name_ == "INPUTDIRECTORY") tid = oskar_SettingsItem::TELESCOPE_DIR_NAME;
    else if (name_ == "DATETIME")       tid = oskar_SettingsItem::DATE_TIME;
    else if (name_ == "TIME")           tid = oskar_SettingsItem::TIME;
    else if (name_ == "DOUBLERANGEEXT")
    {
        if (params.size() == 3)
        {
            string special_value = params[2];
            for (size_t i = 0; i < special_value.length(); ++i)
                special_value[i] = toupper(special_value[i]);
            if (special_value == "MAX") tid = oskar_SettingsItem::DOUBLE_MAX;
            else if (special_value == "MIN") tid = oskar_SettingsItem::DOUBLE_MIN;
            else tid = oskar_SettingsItem::DOUBLE;
        }
        else tid = oskar_SettingsItem::DOUBLE;
    }
    // Note IntRangeExtended currently only maps to AXIS_RANGE
    else if (name_ == "INTRANGEEXT")
    {
        if (params.size() == 3)
        {
            string special_value = params[2];
            for (size_t i = 0; i < special_value.length(); ++i)
                special_value[i] = toupper(special_value[i]);
            if (special_value == "MAX") tid = oskar_SettingsItem::AXIS_RANGE;
        }
    }
//    else {
//        tid = oskar_SettingsItem::UNDEF;
//    }

    return tid;
}

string get_key(node* n)
{
    vector<string> names;
    names.push_back("key");
    names.push_back("k");
    attr* a = get_attribute(n, names);
    if (a) return a->value();
    return "";
}

bool is_required(node* n)
{
    vector<string> names;
    names.push_back("required");
    names.push_back("req");
    names.push_back("r");
    attr* a = get_attribute(n, names);
    if (a) {
        string req(a->value());
        //std::transform(req.begin(), req.end(), req.begin(), std::toupper);
        for (size_t i = 0; i < req.length(); ++i) req[i] = toupper(req[i]);
        return req == "TRUE" || req == "YES";
    }
    return false;
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
    str_replace(desc, "<description>", " ");
    str_replace(desc, "<desc>", " ");
    str_replace(desc, "</description>", " ");
    str_replace(desc, "</desc>", " ");
    desc = str_trim(desc);
    desc = str_trim(desc, " \n");
    desc = str_reduce(desc);
    desc = str_reduce(desc, " ", " \n");
    str_replace(desc, "&amp;", "&");

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
    str_replace(label, "<label>", " ");
    str_replace(label, "<l>", " ");
    str_replace(label, "</label>", " ");
    str_replace(label, "</l>", " ");
    label = str_trim(label);
    label = str_trim(label, " \n");
    label = str_reduce(label);
    label = str_reduce(label, " ", " \n");

    return label;
}

string get_type(node* s, oskar_SettingsItem::type_id& id,
        string& defaultValue, vector<string>& options)
{
    id = oskar_SettingsItem::UNDEF;

    // Obtain a pointer to the type node.
    vector<string> names;
    names.push_back("type");
    names.push_back("t");
    node* n = get_child_node(s, names);
    if (!n) return "";

    // Obtain a pointer to the name attribute
    names.clear();
    names.push_back("name");
    names.push_back("n");
    attr* name = get_attribute(n, names);

    // Types have to have a name to be valid.
    if (!name) return "";
    const string type_name(name->value());

    // Obtain a pointer to the 'default' attribute.
    names.clear();
    names.push_back("default");
    names.push_back("d");
    attr* def = get_attribute(n, names);
    if (def) defaultValue = string(def->value());

    // Obtain any type parameters.
    string param(n->value());
    if (!param.empty()) {
        param = str_trim(param);
        options = str_get_options(param);
    }

    // Convert type name to a mata_type_id.
    id = get_oskar_SettingsItem_type(type_name, options);

    return type_name;
}

} // End anonymous namespace.

///////////////////////////////////////////////////////////////////////////////

namespace oskar {

// Private helper class.
class SettingsModelXML_private
{
public:
    enum Status {
        AllOk = 0,
        MissingKey,
        InvalidNode
    };

    SettingsModelXML_private(SettingsModelXML* model) {this->model = model;}
    //string get_key_type_(const string& key);
    //vector<string> get_option_list_values_(const string& key);
    void index_settings(node* n, string& key_root);
    void iterate_settings(node* n, int& depth, string& key_root);
    SettingsModelXML_private::Status declare_setting(node* s,
            int depth, const string& key_root);
    bool get_dependency(node* n, string& key, string& value) const;

    void setLabel(const string& key, const string& label);
    void declare(const string& key, const string& label,
            const oskar_SettingsItem::type_id& type,
            const string& defaultValue, bool required);
    void declareOptionList(const string& key, const string& label,
            const vector<string>& options,
            const string& defaultValue, bool required);
    void setTooltip(const string& key, const string& description);
    void setDependency(const string& key, const string& depends_key,
            const string& depends_value);

    SettingsModelXML* model;
    vector<string> keys;
    vector<string> types;
    vector<vector<string> > type_params;
};

bool SettingsModelXML_private::get_dependency(node* s, string& key,
        string& value) const
{
    // Obtain a pointer to the depends node.
    vector<string> names;
    names.push_back("d");
    names.push_back("deps");
    names.push_back("depends");
    node* n = get_child_node(s, names);
    if (!n) return false;

    // Obtain a pointer to the 'key' attribute.
    names.clear();
    names.push_back("key");
    names.push_back("k");
    attr* aKey = get_attribute(n, names);

    // Obtain a pointer to the 'value' attribute.
    names.clear();
    names.push_back("value");
    names.push_back("v");
    attr* aValue = get_attribute(n, names);

    if (!aKey || !aValue) return false;

    key   = aKey->value();
    value = aValue->value();

    int iKey = key_index(key, keys);
    if (iKey == -1) {
        string skey = get_key(s);
        cerr << "ERROR: [" << skey << "] dependency key '";
        cerr << key << "' doesn't exist!" << endl;
        return false;
    }
    string key_type = str_to_upper(types[iKey]);

    // For option lists minimal match to the option value.
    if (key_type == "OPTIONLIST") {
//        cout << "OptionList dependency!" << endl;
//        cout << " -- xml declared value: " << value << endl;
        int optionIndex = -1;
        vector<string> options = type_params[iKey];
        for (size_t i = 0; i < options.size(); ++i) {
            if (options[i].find(value) == 0) {
                optionIndex = i;
                break;
            }
        }
        if (optionIndex == -1) {
            string skey = get_key(s);
            cerr << "ERROR: [" << skey << "] with dependency key '";
            cerr << key << "' has an invalid value!" << endl;
        }
        else {
            value = options[optionIndex];
        }
//        cout << " -- matched value: " << value << endl;
    }

    return true;
}

SettingsModelXML_private::Status SettingsModelXML_private::declare_setting(
        node* s, int depth, const string& key_root)
{
    if (!s || string(s->name()) != "s") return InvalidNode;

    // Key
    string key = get_key(s);
    string fullKey = key_root + key;
    if (key.empty()) return MissingKey;

    // Type
    string defaultValue;
    vector<string> options;
    oskar_SettingsItem::type_id id;
    string type = get_type(s, id, defaultValue, options);

    // Label, description & dependency
    string label = get_label(s);
    string desc = get_description(s);
    string dKey, dValue;
    bool has_dependency = get_dependency(s, dKey, dValue);

    // Group -----------------------------------------------------------------
    if (depth == 0 || type.empty()) {
        setLabel(fullKey, label);
#if 0
        else
            cout << "WARNING: Missing description for setting: " << fullKey << endl;
#endif
    }

    // Setting ---------------------------------------------------------------
    else {
        if (id == oskar_SettingsItem::UNDEF)
        {
            cerr << "ERROR: Unknown oskar_SettingsItem type_id for XML type '";
            cout << type << "'";
            cerr << " [key = " << fullKey << "]" << endl;
        }

        bool required = is_required(s);
        if (id == oskar_SettingsItem::OPTIONS)
            declareOptionList(fullKey, label, options, defaultValue, required);
        else
            declare(fullKey, label, id, defaultValue, required);
        if (desc.empty())
            cout << "WARNING: Missing description for setting: " << fullKey << endl;
    }

    // Common properties ------------------------------------------------------
    if (has_dependency)
        setDependency(fullKey, dKey, dValue);
    if (!desc.empty())
        setTooltip(fullKey, desc);

    return AllOk;
}

void SettingsModelXML_private::index_settings(node* n, string& key_root)
{
    vector<string> type_node_names;
    type_node_names.push_back("type");
    type_node_names.push_back("t");
    vector<string> type_attr_names;
    type_attr_names.push_back("name");
    type_attr_names.push_back("n");

    for (node* s = n->first_node("s"); s; s = s->next_sibling())
    {
        string key  = get_key(s);
        if (key.empty()) continue;

        // Add the key to the keys list.
        keys.push_back(key_root + key);

        // Obtain the type name (if defined) and add it to the type list.
        string type = "GROUP";
        node* type_node = get_child_node(s, type_node_names);
        if (type_node) {
            attr* name = get_attribute(type_node, type_attr_names);
            if (name) type = string(name->value());
        }
        types.push_back(type);

        // Obtain any type parameters.
        vector<string> params;
        if (type_node) {
            string type_node_text(type_node->value());
            if (!type_node_text.empty()) {
                type_node_text = str_trim(type_node_text);
                params = str_get_options(type_node_text);
            }
        }
        type_params.push_back(params);

        if (s->first_node("s")) {
            string new_key_root = key_root + key + "/";
            index_settings(s, new_key_root);
        }
    }
}

void SettingsModelXML_private::iterate_settings(node* n, int& depth,
        string& key_root)
{
    for (node* s = n->first_node("s"); s; s = s->next_sibling())
    {
        string key = get_key(s);
        if (key.empty()) continue;

        Status status = declare_setting(s, depth, key_root);
        if (status)
            cout << "ERROR: Problem reading setting with key: " << key << endl;

        // Read any child settings.
        if (s->first_node("s")) {
            string key_root_ = key_root + key + "/";
            int depth_ = depth + 1;
            iterate_settings(s, depth_, key_root_);
        }
    }
}

void SettingsModelXML_private::setLabel(const string& key, const string& label)
{
    model->setLabel(QString::fromStdString(key), QString::fromStdString(label));
}

void SettingsModelXML_private::declare(const string& key, const string& label,
        const oskar_SettingsItem::type_id& type,
        const string& defaultValue, bool required)
{
    model->declare(QString::fromStdString(key), QString::fromStdString(label),
            static_cast<int>(type), QString::fromStdString(defaultValue),
            required);
}

void SettingsModelXML_private::declareOptionList(const string& key,
        const string& label, const vector<string>& options,
        const string& defaultValue, bool required)
{
    QStringList options_;
    int defaultIndex_ = 0;
    for (size_t i = 0; i < options.size(); ++i)
        options_ << QString::fromStdString(options[i]);
    for (size_t i = 0; i < options.size(); ++i) {
        if (options[i].find(defaultValue) == 0) {
            defaultIndex_ = i;
            break;
        }
    }
    model->declare(QString::fromStdString(key), QString::fromStdString(label),
            options_, defaultIndex_, required);
}

void SettingsModelXML_private::setTooltip(const string& key,
        const string& description)
{
    model->setTooltip(QString::fromStdString(key),
            QString::fromStdString(description));
}

void SettingsModelXML_private::setDependency(const string& key,
        const string& depends_key, const string& depends_value)
{
    model->setDependency(QString::fromStdString(key),
            QString::fromStdString(depends_key),
            QString::fromStdString(depends_value));
}

///////////////////////////////////////////////////////////////////////////////

SettingsModelXML::SettingsModelXML(QObject* parent)
: oskar_SettingsModel(parent)
{
    p = new SettingsModelXML_private(this);

    string xml(OSKAR_XML_STR);
    vector<char> xml_(xml.begin(), xml.end());
    xml_.push_back(0);

    int depth = 0;
    doc_t doc;
    doc.parse<0>(&xml_[0]);

    node* root_node = doc.first_node("root");
    string temp;
    p->index_settings(root_node, temp);
//    cout << "total number of keys = " << keys_.size() << endl;
//    for (size_t i = 0; i < keys_.size(); ++i) {
//        cout << i << " -- " << keys_[i];
//        cout << " -- " << types_[i];
//        cout << " -- " << type_params_[i].size();
//        cout << endl;
//    }

    string version = "";
    attr* ver = root_node->first_attribute("version");
    if (ver) version = string(ver->value());

    string key_root = "";
    p->iterate_settings(root_node, depth, key_root);
}

SettingsModelXML::~SettingsModelXML()
{
    delete p;
}

} // namespace oskar
