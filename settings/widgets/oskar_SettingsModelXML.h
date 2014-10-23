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

#ifndef OSKAR_SETTINGS_MODEL_XML_H_
#define OSKAR_SETTINGS_MODEL_XML_H_

/**
 * @file oskar_SettingsModelXML.h
 */

#include <oskar_global.h>
#include <oskar_SettingsItem.h>
#include <oskar_SettingsModel.h>

#include <rapidxml.hpp>

#include <string>
#include <vector>

namespace oskar {

class SettingsModelXML : public oskar_SettingsModel
{
public:
    enum Status {
        AllOk = 0,
        MissingKey,
        InvalidNode
    };

public:
    SettingsModelXML(QObject* parent = 0);
    ~SettingsModelXML();

    //static std::vector<std::pair<std::string, std::string> > get_defaults();


private:
    void index_settings_(rapidxml::xml_node<>* node, std::string& key_root);
    int key_index_(const std::string& key) const;
    std::string get_key_type_(const std::string& key);
    std::vector<std::string> get_option_list_values_(const std::string& key);

    void setLabel_(const std::string& key, const std::string& label);
    void declare_(const std::string& key, const std::string& label,
            const oskar_SettingsItem::type_id& type,
            const std::string& defaultValue, bool required);
    void declareOptionList_(const std::string& key, const std::string& label,
            const std::vector<std::string>& options,
            const std::string& defaultValue, bool required);
    void setTooltip_(const std::string& key, const std::string& description);
    void setDependency_(const std::string& key, const std::string& depends_key,
            const std::string& depends_value);

    SettingsModelXML::Status decalare_setting_(rapidxml::xml_node<>* s,
            int depth, const std::string& key_root);

    void iterate_settings_(rapidxml::xml_node<>* n,
            int& depth, std::string& key_root);

    std::string get_key_(rapidxml::xml_node<>* n) const;
    std::string get_label_(rapidxml::xml_node<>* n) const;
    std::string get_description_(rapidxml::xml_node<>* n) const;
    bool is_required_(rapidxml::xml_node<>* n) const;
    std::string get_type_(rapidxml::xml_node<>* n,
            oskar_SettingsItem::type_id& id,
            std::string& defaultValue, std::vector<std::string>& options) const;
    bool get_dependency_(rapidxml::xml_node<>* n, std::string& key,
            std::string& value) const;

    oskar_SettingsItem::type_id get_oskar_SettingsItem_type_(
            const std::string& name,
            const std::vector<std::string>& params) const;

    rapidxml::xml_node<>* get_child_node_(rapidxml::xml_node<>* parent,
            const std::vector<std::string>& possible_names) const;
    rapidxml::xml_attribute<>* get_attribute_(rapidxml::xml_node<>* n,
            const std::vector<std::string>& possible_names) const;

    std::vector<std::string> str_get_options_(const std::string& s) const;
    std::string str_reduce_(const std::string& str,
            const std::string& fill = " ",
            const std::string& whitespace = " \t") const;
    std::string str_trim_(const std::string& str,
            const std::string& whitespace = " \t") const;
    std::string str_replace_(std::string& s, std::string toReplace,
            std::string replaceWith) const;

    std::string str_to_upper_(const std::string& s) const;

private:
    std::vector<std::string> keys_;
    std::vector<std::string> types_;
    std::vector<std::vector<std::string> > type_params_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_MODEL_XML_H_ */
