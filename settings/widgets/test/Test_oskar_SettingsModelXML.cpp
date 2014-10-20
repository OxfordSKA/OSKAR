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

using namespace std;

enum status_t
{
    AllOk = 0,
    MissingKey,
    InvalidNode
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
    typedef rapidxml::xml_node<> node;
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

static status_t declare_setting_(rapidxml::xml_node<>* s, int depth)
{
    typedef rapidxml::xml_node<> node;
    typedef rapidxml::xml_attribute<> attr;

    if (!s || std::string(s->name()) != "s") return InvalidNode;

    // key
    std::string key = get_key(s);
    if (key.empty()) return MissingKey;
    cout << "Key  : " << key << endl;

    // required
    attr* required = s->first_attribute("required");

    // label
    node* label = s->first_node("label");

    // description TODO also handle "desc"
    std::string desc = get_description(s);
    cout << "Desc : [" << desc << "]" <<endl << endl;

    // type
    node* type = s->first_node("type");

    // dependencies
    node* deps = s->first_node("depends");

    // Group -----------------------------------------------------------------
    // Groups are at level 0 or have a key but no type.


    // Setting ---------------------------------------------------------------

    return AllOk;
}

static void parse_settings_(rapidxml::xml_node<>* n, int& depth, std::string& key_root)
{
    typedef rapidxml::xml_node<> node;
    typedef rapidxml::xml_attribute<> attr;

    for (node* s = n->first_node("s"); s; s = s->next_sibling())
    {
        std::string key = get_key(s);
        if (key.empty()) continue;

        status_t status = declare_setting_(s, depth);
        if (status) {
            cout << "ERROR: Problem reading setting with key: " << key << endl;
        }

        // Read any child settings.
//        if (s->first_node("s")) {
//            std::string key_root_ = key_root + key + "/";
//            int depth_ = depth + 1;
//            parse_settings_(s, depth_, key_root_);
//        }
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


