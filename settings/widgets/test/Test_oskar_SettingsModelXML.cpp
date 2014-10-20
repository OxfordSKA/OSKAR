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

#include <oskar_global.h>
#include <gtest/gtest.h>
#include <cstdio>

#include <rapidxml-1.13/rapidxml.hpp>
#include <rapidxml-1.13/rapidxml_print.hpp>

#include "oskar_xml_all.h"
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <locale>

#include <oskar_SettingsItem.h>

using namespace std;

enum status_t
{
    AllOk = 0,
    MissingKey,
    InvalidNode
};

enum meta_type_id
{
    None = 0,
};

std::string str_replace(std::string& s, std::string toReplace, std::string replaceWith)
{
    size_t p = s.find(toReplace);
    if (p != std::string::npos)
        return s.replace(p, toReplace.length(), replaceWith);
    else return s;
}

std::string str_trim(const std::string& str, const std::string& whitespace = " \t")
{
    const size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos) return "";
    const size_t strEnd = str.find_last_not_of(whitespace);
    const size_t strRange = strEnd - strBegin + 1;
    return str.substr(strBegin, strRange);
}

std::string str_reduce(const std::string& str,
                   const std::string& fill = " ",
                   const std::string& whitespace = " \t")
{
    // trim first
    std::string result = str_trim(str, whitespace);

    // replace sub ranges
    size_t beginSpace = result.find_first_of(whitespace);
    while (beginSpace != std::string::npos)
    {
        const size_t endSpace = result.find_first_not_of(whitespace, beginSpace);
        const size_t range = endSpace - beginSpace;

        result.replace(beginSpace, range, fill);

        const size_t newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }
    return result;
}

std::vector<std::string> str_get_options(const std::string& s)
{
    std::vector<std::string> opts;
    std::stringstream sin(s);
    std::string line;

    while (std::getline(sin, line, '"')) {
        std::stringstream ss(line);
        while (std::getline(ss, line, ',')) {
            if (!line.empty()) opts.push_back(line);
        }
        if (std::getline(sin, line, '"')) {
            if (!line.empty()) opts.push_back(str_trim(line, " \n"));
        }
    }
    return opts;
}

rapidxml::xml_node<>* get_child_node(rapidxml::xml_node<>* parent,
        const std::vector<std::string>& possible_names)
{
    typedef rapidxml::xml_node<> node;
    for (size_t i = 0; i < possible_names.size(); ++i)
    {
        node* n = parent->first_node(possible_names[i].c_str());
        if (n) return n;
    }
    return NULL;
}

rapidxml::xml_attribute<>* get_attribute(rapidxml::xml_node<>* n,
        const std::vector<std::string>& possible_names)
{
    typedef rapidxml::xml_attribute<> attr;
    for (size_t i = 0; i < possible_names.size(); ++i)
    {
        attr* a = n->first_attribute(possible_names[i].c_str());
        if (a) return a;
    }
    return NULL;
}

std::string get_key(rapidxml::xml_node<>* n)
{
    typedef rapidxml::xml_attribute<> attr;
    std::vector<std::string> names;
    names.push_back("key");
    names.push_back("k");
    attr* a = get_attribute(n, names);
    if (a) return a->value();
    return "";
}

bool is_required(rapidxml::xml_node<>* n)
{
    typedef rapidxml::xml_attribute<> attr;
    std::vector<std::string> names;
    names.push_back("required");
    names.push_back("req");
    names.push_back("r");
    attr* a = get_attribute(n, names);
    if (a) {
        std::string req(a->value());
        //std::transform(req.begin(), req.end(), req.begin(), std::toupper);
        for (size_t i = 0; i < req.length(); ++i) req[i] = toupper(req[i]);
        return req == "TRUE" || req == "YES";
    }
    return false;
}

std::string get_description(rapidxml::xml_node<>* s)
{
    typedef rapidxml::xml_node<> node;
    // Obtain a pointer to the description node.
    std::vector<std::string> names;
    names.push_back("description");
    names.push_back("desc");
    node* n = get_child_node(s, names);
    if (!n) return "";

    // Convert the node to a string.
    stringstream ss;
    ss << *n;
    string desc = ss.str();

    // Remove the opening description tag and excess whitespace.
    str_replace(desc, "<description>", " ");
    str_replace(desc, "<desc>", " ");
    str_replace(desc, "</description>", " ");
    str_replace(desc, "</desc>", " ");
    desc = str_trim(desc);
    desc = str_trim(desc, " \n");
    desc = str_reduce(desc);
    desc = str_reduce(desc, " ", " \n");

    return desc;
}

std::string get_label(rapidxml::xml_node<>* s)
{
    typedef rapidxml::xml_node<> node;
    // Obtain a pointer to the label node.
    std::vector<std::string> names;
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

//meta_type_id type_name_to_type_id(const std::string& name)
//{
//    std::string name_(name);
//    for (size_t i = 0; i < name_.length(); ++i) name_[i] = toupper(name_[i]);
//
//    if (name_ == "BOOL")
//
//
//    cout << name_ << endl;
//
//    return None;
//}

oskar_SettingsItem::type_id get_oskar_SettingsItem_type(
        const std::string& name,
        const std::vector<std::string>& params)
{
    std::string name_(name);
    for (size_t i = 0; i < name_.length(); ++i) name_[i] = toupper(name_[i]);
    oskar_SettingsItem::type_id tid = oskar_SettingsItem::UNDEF;

    if      (name_ == "BOOL")           tid = oskar_SettingsItem::BOOL;
    else if (name_ == "INT")            tid = oskar_SettingsItem::INT;
    else if (name_ == "INTPOSITIVE")    tid = oskar_SettingsItem::INT_POSITIVE;
    else if (name_ == "INTLISTEXT")     tid = oskar_SettingsItem::INT_CSV_LIST;

    else if (name_ == "INTPUTFILELIST") tid = oskar_SettingsItem::INPUT_FILE_LIST;


    else if (name_ == "DOUBLE")         tid = oskar_SettingsItem::DOUBLE;
    else if (name_ == "DOUBLERANGE")    tid = oskar_SettingsItem::DOUBLE;
    else if (name_ == "OPTIONLIST")     tid = oskar_SettingsItem::OPTIONS;
    else if (name_ == "INPUTFILE")      tid = oskar_SettingsItem::INPUT_FILE_NAME;

    else if (name_ == "DOUBLERANGEEXT") {
        if (params.size() == 3) {
            std::string special_value = params[2];
            for (size_t i = 0; i < special_value.length(); ++i)
                special_value[i] = toupper(special_value[i]);
            if (special_value == "MAX")
                tid = oskar_SettingsItem::DOUBLE_MAX;
            else if (special_value == "MIN")
                tid = oskar_SettingsItem::DOUBLE_MIN;
            else
                tid = oskar_SettingsItem::DOUBLE;
        }
        else
            tid = oskar_SettingsItem::DOUBLE;
    }


    return tid;
}


std::string get_type(rapidxml::xml_node<>* s)
{
    typedef rapidxml::xml_node<> node;
    typedef rapidxml::xml_attribute<> attr;

    // Obtain a pointer to the type node.
    std::vector<std::string> names;
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
    const std::string type_name(name->value());

    // Obtain a pointer to the 'default' attribute
    names.clear();
    names.push_back("default");
    names.push_back("d");
    attr* def = get_attribute(n, names);

    // Obtain any type parameters
    std::string param(n->value());
    std::vector<std::string> opts;
    if (!param.empty()) {
        param = str_trim(param);
        opts = str_get_options(param);
    }

    // Convert type name to a mata_type_id
    oskar_SettingsItem::type_id id = get_oskar_SettingsItem_type(type_name, opts);

    return type_name;
}

static status_t declare_setting_(rapidxml::xml_node<>* s, int depth, const std::string& key_root)
{
    typedef rapidxml::xml_node<> node;
    typedef rapidxml::xml_attribute<> attr;

    if (!s || std::string(s->name()) != "s") return InvalidNode;

    // key
    std::string key = get_key(s);
    if (key.empty()) return MissingKey;

    // required
    bool required = is_required(s);

    // label
    std::string label = get_label(s);

    // description
    std::string desc = get_description(s);

    // type
    std::string type = get_type(s);

    // dependencies
    node* deps = s->first_node("depends");

    // Group -----------------------------------------------------------------
    // Groups are at level 0 or have a key but no type.
    if (depth == 0 && type.empty()) {
        cout << endl;
        cout << key;
        cout << " [GROUP]";
        cout << endl;
    }

    // Setting ---------------------------------------------------------------
    else {
        //cout << key_root << key << endl;
        cout << string(depth*4,' ');
        cout << key;
        cout << " [" << type << "]";
        //if (required) cout << " [REQUIRED]" << endl;
        //else cout << endl;
        //    cout << "Desc  : [" << desc  << "]" << endl;
        //cout << "Label : [" << label << "]" << endl;
        cout << endl;
    }

    return AllOk;
}

static void parse_settings_(rapidxml::xml_node<>* n, int& depth, std::string& key_root)
{
    typedef rapidxml::xml_node<> node;

    for (node* s = n->first_node("s"); s; s = s->next_sibling())
    {
        std::string key = get_key(s);
        if (key.empty()) continue;

        status_t status = declare_setting_(s, depth, key_root);
        if (status) {
            cout << "ERROR: Problem reading setting with key: " << key << endl;
        }

        // Read any child settings.
        if (s->first_node("s")) {
            std::string key_root_ = key_root + key + "/";
            int depth_ = depth + 1;
            parse_settings_(s, depth_, key_root_);
        }
    }
}

TEST(oskar_SettingsModelXML, test1)
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

    std::string version = "";
    attr_t* ver = root_node->first_attribute("version");
    if (ver) version = std::string(ver->value());

    ASSERT_STREQ(OSKAR_VERSION_STR, version.c_str());

    std::string key_root = "";
    parse_settings_(root_node, depth, key_root);

}


