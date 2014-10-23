/*
 * Copyright (c) 2014, The University of Oxford
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

#include <sstream>
#include <iostream>

using namespace std;

namespace oskar {

SettingsModelXML::SettingsModelXML(QObject* parent)
: oskar_SettingsModel(parent)
{
    typedef rapidxml::xml_document<> doc_t;
    typedef rapidxml::xml_node<> node_t;
    typedef rapidxml::xml_attribute<> attr_t;

    std::string xml(OSKAR_XML_STR);
    std::vector<char> xml_(xml.begin(), xml.end());
    xml_.push_back(0);

    int depth = 0;
    doc_t doc;
    doc.parse<0>(&xml_[0]);

    node_t* root_node = doc.first_node("root");
    std::string temp;
    index_settings_(root_node, temp);
//    cout << "total number of keys = " << keys_.size() << endl;
//    for (size_t i = 0; i < keys_.size(); ++i) {
//        cout << i << " -- " << keys_[i];
//        cout << " -- " << types_[i];
//        cout << " -- " << type_params_[i].size();
//        cout << endl;
//    }

    std::string version = "";
    attr_t* ver = root_node->first_attribute("version");
    if (ver) version = std::string(ver->value());

    std::string key_root = "";
    iterate_settings_(root_node, depth, key_root);
}


SettingsModelXML::~SettingsModelXML()
{
}

void SettingsModelXML::index_settings_(rapidxml::xml_node<>* n, std::string& key_root)
{
    typedef rapidxml::xml_node<> node;
    typedef rapidxml::xml_attribute<> attr;
    std::vector<std::string> type_node_names;
    type_node_names.push_back("type");
    type_node_names.push_back("t");
    std::vector<std::string> type_attr_names;
    type_attr_names.push_back("name");
    type_attr_names.push_back("n");

    for (node* s = n->first_node("s"); s; s = s->next_sibling())
    {
        std::string key  = get_key_(s);
        if (key.empty()) continue;

        // Add the key to the keys list.
        keys_.push_back(key_root + key);

        // Obtain the type name (if defined) and add it to the type list.
        std::string type = "GROUP";
        node* type_node = get_child_node_(s, type_node_names);
        if (type_node) {
            attr* name = get_attribute_(type_node, type_attr_names);
            if (name) type = std::string(name->value());
        }
        types_.push_back(type);

        // Obtain any type parameters.
        std::vector<std::string> params;
        if (type_node) {
            std::string type_node_text(type_node->value());
            if (!type_node_text.empty()) {
                type_node_text = str_trim_(type_node_text);
                params = str_get_options_(type_node_text);
            }
        }
        type_params_.push_back(params);

        if (s->first_node("s")) {
            std::string new_key_root = key_root + key + "/";
            index_settings_(s, new_key_root);
        }
    }
}

int SettingsModelXML::key_index_(const std::string& key) const
{
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == key) return i;
    return -1;
}

std::string SettingsModelXML::str_replace_(std::string& s,
        std::string toReplace, std::string replaceWith) const
{
    // FIXME clean up this logic!
    size_t p = 0;
    while (p != std::string::npos) {
        p = s.find(toReplace);
        if (p != std::string::npos)
            s.replace(p, toReplace.length(), replaceWith);
    }
    return s;
}

std::string SettingsModelXML::str_trim_(const std::string& str,
        const std::string& whitespace) const
{
    const size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos) return "";
    const size_t strEnd = str.find_last_not_of(whitespace);
    const size_t strRange = strEnd - strBegin + 1;
    return str.substr(strBegin, strRange);
}

std::string SettingsModelXML::str_reduce_(const std::string& str,
                   const std::string& fill, const std::string& whitespace) const
{
    // Trim first
    std::string result = str_trim_(str, whitespace);

    // Replace sub ranges
    size_t beginSpace = result.find_first_of(whitespace);
    while (beginSpace != std::string::npos)
    {
        const size_t endSpace = result.find_first_not_of(whitespace, beginSpace);
        const size_t range = endSpace-beginSpace;
        result.replace(beginSpace, range, fill);
        const size_t newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }

    return result;
}

std::vector<std::string> SettingsModelXML::str_get_options_(
        const std::string& s) const
{
    std::vector<std::string> opts;
    std::stringstream sin(s);
    std::string line;
    while (std::getline(sin, line, '"')) {
        std::stringstream ss(line);
        while (std::getline(ss, line, ',')) {
            line = str_trim_(line, " \t");
            if (!line.empty()) opts.push_back(line);
        }
        if (std::getline(sin, line, '"')) {
            if (!line.empty()) opts.push_back(line);
        }
    }
    return opts;
}

std::string SettingsModelXML::str_to_upper_(const std::string& s) const
{
    std::string s_(s);
    for (size_t i = 0; i < s_.length(); ++i) s_[i] = toupper(s_[i]);
    return s_;
}

rapidxml::xml_node<>* SettingsModelXML::get_child_node_(
        rapidxml::xml_node<>* parent,
        const std::vector<std::string>& possible_names) const
{
    typedef rapidxml::xml_node<> node;
    for (size_t i = 0; i < possible_names.size(); ++i)
    {
        node* n = parent->first_node(possible_names[i].c_str());
        if (n) return n;
    }
    return 0;
}

rapidxml::xml_attribute<>* SettingsModelXML::get_attribute_(
        rapidxml::xml_node<>* n,
        const std::vector<std::string>& possible_names) const
{
    typedef rapidxml::xml_attribute<> attr;
    for (size_t i = 0; i < possible_names.size(); ++i)
    {
        attr* a = n->first_attribute(possible_names[i].c_str());
        if (a) return a;
    }
    return 0;
}


oskar_SettingsItem::type_id SettingsModelXML::get_oskar_SettingsItem_type_(
        const std::string& name, const std::vector<std::string>& params) const
{
    std::string name_(name);
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
    else if (name_ == "DOUBLERANGEEXT") {
        if (params.size() == 3) {
            std::string special_value = params[2];
            for (size_t i = 0; i < special_value.length(); ++i)
                special_value[i] = toupper(special_value[i]);
            if (special_value == "MAX") tid = oskar_SettingsItem::DOUBLE_MAX;
            else if (special_value == "MIN") tid = oskar_SettingsItem::DOUBLE_MIN;
            else tid = oskar_SettingsItem::DOUBLE;
        }
        else tid = oskar_SettingsItem::DOUBLE;
    }
    // Note IntRangeExtended currently only maps to AXIS_RANGE
    else if (name_ == "INTRANGEEXT") {
        if (params.size() == 3) {
            std::string special_value = params[2];
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

std::string SettingsModelXML::get_key_(rapidxml::xml_node<>* n) const
{
    typedef rapidxml::xml_attribute<> attr;
    std::vector<std::string> names;
    names.push_back("key");
    names.push_back("k");
    attr* a = get_attribute_(n, names);
    if (a) return a->value();
    return "";
}

bool SettingsModelXML::is_required_(rapidxml::xml_node<>* n) const
{
    typedef rapidxml::xml_attribute<> attr;
    std::vector<std::string> names;
    names.push_back("required");
    names.push_back("req");
    names.push_back("r");
    attr* a = get_attribute_(n, names);
    if (a) {
        std::string req(a->value());
        //std::transform(req.begin(), req.end(), req.begin(), std::toupper);
        for (size_t i = 0; i < req.length(); ++i) req[i] = toupper(req[i]);
        return req == "TRUE" || req == "YES";
    }
    return false;
}

std::string SettingsModelXML::get_description_(rapidxml::xml_node<>* s) const
{
    typedef rapidxml::xml_node<> node;

    // Obtain a pointer to the description node.
    std::vector<std::string> names;
    names.push_back("description");
    names.push_back("desc");
    node* n = get_child_node_(s, names);
    if (!n) return "";

    // Convert the node to a string.
    stringstream ss;
    ss << *n;
    string desc = ss.str();

    //cout << "[" << desc << "]" << endl;

    // Remove the opening description tag and excess whitespace.
    str_replace_(desc, "<description>", " ");
    str_replace_(desc, "<desc>", " ");
    str_replace_(desc, "</description>", " ");
    str_replace_(desc, "</desc>", " ");
    desc = str_trim_(desc);
    desc = str_trim_(desc, " \n");
    desc = str_reduce_(desc);
    desc = str_reduce_(desc, " ", " \n");
    str_replace_(desc, "&amp;", "&");

    return desc;
}

std::string SettingsModelXML::get_label_(rapidxml::xml_node<>* s) const
{
    typedef rapidxml::xml_node<> node;
    // Obtain a pointer to the label node.
    std::vector<std::string> names;
    names.push_back("label");
    names.push_back("l");
    node* n = get_child_node_(s, names);
    if (!n) return "";

    // Convert the node to a string.
    stringstream ss;
    ss << *n;
    string label = ss.str();

    // Remove the opening label tag and excess whitespace.
    str_replace_(label, "<label>", " ");
    str_replace_(label, "<l>", " ");
    str_replace_(label, "</label>", " ");
    str_replace_(label, "</l>", " ");
    label = str_trim_(label);
    label = str_trim_(label, " \n");
    label = str_reduce_(label);
    label = str_reduce_(label, " ", " \n");

    return label;
}

std::string SettingsModelXML::get_type_(rapidxml::xml_node<>* s,
        oskar_SettingsItem::type_id& id,
        std::string& defaultValue,
        std::vector<std::string>& options) const
{
    typedef rapidxml::xml_node<> node;
    typedef rapidxml::xml_attribute<> attr;

    // Obtain a pointer to the type node.
    std::vector<std::string> names;
    names.push_back("type");
    names.push_back("t");
    node* n = get_child_node_(s, names);
    if (!n) return "";

    // Obtain a pointer to the name attribute
    names.clear();
    names.push_back("name");
    names.push_back("n");
    attr* name = get_attribute_(n, names);

    // Types have to have a name to be valid.
    if (!name) return "";
    const std::string type_name(name->value());

    // Obtain a pointer to the 'default' attribute.
    names.clear();
    names.push_back("default");
    names.push_back("d");
    attr* def = get_attribute_(n, names);
    if (def) defaultValue = std::string(def->value());

    // Obtain any type parameters.
    std::string param(n->value());
    if (!param.empty()) {
        param = str_trim_(param);
        options = str_get_options_(param);
    }

    // Convert type name to a mata_type_id.
    id = get_oskar_SettingsItem_type_(type_name, options);

    return type_name;
}

bool SettingsModelXML::get_dependency_(rapidxml::xml_node<>* s,
        std::string& key, std::string& value) const
{
    typedef rapidxml::xml_node<> node;
    typedef rapidxml::xml_attribute<> attr;

    // Obtain a pointer to the depends node.
    std::vector<std::string> names;
    names.push_back("d");
    names.push_back("deps");
    names.push_back("depends");
    node* n = get_child_node_(s, names);
    if (!n) return false;

    // Obtain a pointer to the 'key' attribute.
    names.clear();
    names.push_back("key");
    names.push_back("k");
    attr* aKey = get_attribute_(n, names);

    // Obtain a pointer to the 'value' attribute.
    names.clear();
    names.push_back("value");
    names.push_back("v");
    attr* aValue = get_attribute_(n, names);

    if (!aKey || !aValue) return false;

    key   = aKey->value();
    value = aValue->value();

    int iKey = key_index_(key);
    if (iKey == -1) {
        std::string skey = get_key_(s);
        cerr << "ERROR: [" << skey << "] dependency key '";
        cerr << key << "' doesn't exist!" << endl;
    }
    std::string key_type = str_to_upper_(types_[iKey]);

    // For option lists minimal match to the option value.
    if (key_type == "OPTIONLIST") {
//        cout << "OptionList dependency!" << endl;
//        cout << " -- xml declared value: " << value << endl;
        int optionIndex = -1;
        std::vector<std::string> options = type_params_[iKey];
        for (size_t i = 0; i < options.size(); ++i) {
            if (options[i].find(value) == 0) {
                optionIndex = i;
                break;
            }
        }
        if (optionIndex == -1) {
            std::string skey = get_key_(s);
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

void SettingsModelXML::setLabel_(const std::string& key, const std::string& label)
{
    setLabel(QString::fromStdString(key), QString::fromStdString(label));
}

void SettingsModelXML::declare_(const std::string& key, const std::string& label,
        const oskar_SettingsItem::type_id& type,
        const std::string& defaultValue, bool required)
{
    declare(QString::fromStdString(key), QString::fromStdString(label),
            static_cast<int>(type), QString::fromStdString(defaultValue),
            required);
}

void SettingsModelXML::declareOptionList_(const std::string& key, const std::string& label,
        const std::vector<std::string>& options,
        const std::string& defaultValue, bool required)
{
    QStringList options_;
    int defaultIndex_ = 0;
    for (size_t i = 0; i < options.size(); ++i) {
        options_ << QString::fromStdString(options[i]);
    }
    for (size_t i = 0; i < options.size(); ++i) {
        if (options[i].find(defaultValue) == 0) {
            defaultIndex_ = i;
            break;
        }
    }
    declare(QString::fromStdString(key), QString::fromStdString(label),
            options_, defaultIndex_, required);
}

void SettingsModelXML::setTooltip_(const std::string& key,
        const std::string& description)
{
    setTooltip(QString::fromStdString(key), QString::fromStdString(description));
}

void SettingsModelXML::setDependency_(const std::string& key,
        const std::string& depends_key, const std::string& depends_value)
{
    setDependency(QString::fromStdString(key),
            QString::fromStdString(depends_key),
            QString::fromStdString(depends_value));
}

SettingsModelXML::Status SettingsModelXML::decalare_setting_(
        rapidxml::xml_node<>* s, int depth, const std::string& key_root)
{
    if (!s || std::string(s->name()) != "s") return InvalidNode;

    // Key
    std::string key = get_key_(s);
    std::string fullKey = key_root + key;
    if (key.empty()) return MissingKey;

    // Type
    std::string defaultValue;
    std::vector<std::string> options;
    oskar_SettingsItem::type_id id;
    std::string type = get_type_(s, id, defaultValue, options);
    if (id == oskar_SettingsItem::UNDEF) {
        cerr << "ERROR: Unknown oskar_SettingsItem type_id for xml type '";
        cout << type << "'";
        cerr << " [key = " << fullKey << "]" << endl;
    }

    // Label, description & dependency
    std::string label = get_label_(s);
    std::string desc = get_description_(s);
    std::string dKey, dValue;
    bool has_dependency = get_dependency_(s, dKey, dValue);

    // Group -----------------------------------------------------------------
    if (depth == 0 || type.empty()) {
        setLabel_(fullKey, label);
#if 0
        else
            cout << "WARNING: Missing description for setting: " << fullKey << endl;
#endif
    }

    // Setting ---------------------------------------------------------------
    else {
        bool required = is_required_(s);
        if (id == oskar_SettingsItem::OPTIONS)
            declareOptionList_(fullKey, label, options, defaultValue, required);
        else
            declare_(fullKey, label, id, defaultValue, required);
        if (desc.empty())
            cout << "WARNING: Missing description for setting: " << fullKey << endl;
    }

    // Common properties ------------------------------------------------------
    if (has_dependency)
        setDependency_(fullKey, dKey, dValue);
    if (!desc.empty())
        setTooltip_(fullKey, desc);

    return AllOk;
}


void SettingsModelXML::iterate_settings_(rapidxml::xml_node<>* n,
        int& depth, std::string& key_root)
{
    typedef rapidxml::xml_node<> node;

    for (node* s = n->first_node("s"); s; s = s->next_sibling())
    {
        std::string key = get_key_(s);
        if (key.empty()) continue;

        Status status = decalare_setting_(s, depth, key_root);
        if (status) {
            cout << "ERROR: Problem reading setting with key: " << key << endl;
        }

        // Read any child settings.
        if (s->first_node("s")) {
            std::string key_root_ = key_root + key + "/";
            int depth_ = depth + 1;
            iterate_settings_(s, depth_, key_root_);
        }
    }
}

} // namespace oskar

