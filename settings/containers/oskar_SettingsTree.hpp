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

#ifndef OSKAR_SETTINGS_TREE_HPP_
#define OSKAR_SETTINGS_TREE_HPP_

#include <oskar_SettingsValue.hpp>
#include <oskar_SettingsNode.hpp>

#ifdef __cplusplus

#include <utility>

namespace oskar {

class SettingsFileHandler;
class SettingsDependencyGroup;

/**
 * @class SettingsTree
 *
 * @brief Tree of settings for use with OSKAR applications.
 *
 * @details
 * High level representation of a tree of OSKAR application settings in memory.
 *
 * TODO: rename from tree to Settings? Yes please!
 */

class SettingsTree
{
 public:
    SettingsTree(char key_separator = '/');
    ~SettingsTree();

    void set_file_handler(SettingsFileHandler*,
            const std::string& name = std::string());
    void set_file_name(const std::string& name);
    void set_defaults();

    void begin_group(const std::string& name);
    void end_group();
    std::string group_prefix() const;

    bool add_setting(const std::string& key,
                     const std::string& label = std::string(),
                     const std::string& description = std::string(),
                     const std::string& type_name = std::string(),
                     const std::string& type_default = std::string(),
                     const std::string& type_parameters = std::string(),
                     bool required = false);

    bool begin_dependency_group(const std::string& logic = "AND");
    void end_dependency_group();

    bool add_dependency(const std::string& dependency_key,
                        const std::string& value,
                        const std::string& condition = "EQ");

    /* TODO(BM) set value for other intrinsic types?! (needed for GUI) */
    bool set_value(const std::string& key, const std::string& value,
            bool write = true);
    const SettingsItem* item(const std::string& key) const;
    const SettingsValue* value(const std::string& key) const;
    const SettingsValue* operator[](const std::string& key) const;

    int num_items() const;
    int num_settings() const;
    void print() const;
    void clear();
    bool save(const std::string& file_name = std::string()) const;
    bool load(std::vector<std::pair<std::string, std::string> >& failed,
            const std::string& file_name = std::string());
    std::string file_version() const;

    bool contains(const std::string &key) const;
    bool dependencies_satisfied(const std::string& key) const;
    bool is_critical(const std::string& key) const;
    bool is_modified() const;

    const SettingsNode* root_node() const;

 private:
    void print_(const SettingsNode* node, int depth = 0) const;
    const SettingsNode* find_(const SettingsNode* node,
                              const SettingsKey& full_key, int depth) const;
    SettingsNode* find_(SettingsNode* node, const SettingsKey& full_key,
                        int depth);
    bool is_critical_(const SettingsNode* node) const;
    std::string group_prefix_() const;
    bool dependencies_satisfied_(const SettingsDependencyGroup* group) const;
    bool dependency_satisfied_(const SettingsDependency* dep) const;
    bool parent_dependencies_satisfied_(const SettingsNode*) const;
    void set_defaults_(SettingsNode* node);

 private:
    SettingsNode* root_;
    SettingsFileHandler* file_handler_;
    SettingsItem* current_node_;
    char sep_;
    int num_items_;
    int num_settings_;
    mutable bool modified_;
    std::vector<std::string> group_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* C interface. */
struct oskar_Settings;
#ifndef OSKAR_SETTINGS_TYPEDEF_
#define OSKAR_SETTINGS_TYPEDEF_
typedef struct oskar_Settings oskar_Settings;
#endif /* OSKAR_SETTINGS_TYPEDEF_ */

oskar_Settings* oskar_settings_create();
void oskar_settings_free(oskar_Settings* s);

void oskar_settings_set_file_handler(oskar_Settings* s, void* handler);
void oskar_settings_set_file_name(oskar_Settings* s, const char* name);
void oskar_settings_begin_group(oskar_Settings* s, const char* name);
void oskar_settings_end_group(oskar_Settings* s);

int oskar_settings_add_setting(oskar_Settings* s,
        const char* key, const char* label, const char* description,
        const char* type_name, const char* type_default,
        const char* type_parameters, int required);

int oskar_settings_begin_dependency_group(oskar_Settings* s,
        const char* logic);
void oskar_settings_end_dependency_group(oskar_Settings* s);

int oskar_settings_add_dependency(oskar_Settings* s,
        const char* dependency_key, const char* value, const char* condition);

int oskar_settings_set_value(oskar_Settings* s,
        const char* key, const char* value);
const oskar_SettingsValue* oskar_settings_value(
        const oskar_Settings* s, const char* key);
int oskar_settings_starts_with(const oskar_Settings* s, const char* key,
        const char* str, int* status);
char oskar_settings_first_letter(const oskar_Settings* s, const char* key,
        int* status);
char* oskar_settings_to_string(const oskar_Settings* s,
        const char* key, int* status);
int oskar_settings_to_int(const oskar_Settings* s,
        const char* key, int* status);
double oskar_settings_to_double(const oskar_Settings* s,
        const char* key, int* status);
char** oskar_settings_to_string_list(const oskar_Settings* s,
        const char* key, int* num, int* status);
int* oskar_settings_to_int_list(const oskar_Settings* s,
        const char* key, int* num, int* status);
double* oskar_settings_to_double_list(const oskar_Settings* s,
        const char* key, int* num, int* status);

int oskar_settings_num_items(const oskar_Settings* s);
int oskar_settings_num_settings(const oskar_Settings* s);
void oskar_settings_print(const oskar_Settings* s);
void oskar_settings_clear(oskar_Settings* s);
void oskar_settings_save(const oskar_Settings* s, const char* file_name,
        int* status);
void oskar_settings_load(oskar_Settings* s, const char* file_name,
        int* num_failed, char*** failed_keys, int* status);

int oskar_settings_contains(const oskar_Settings* s, const char* key);
int oskar_settings_dependencies_satisfied(const oskar_Settings* s,
        const char* key);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SETTINGS_TREE_HPP_ */
