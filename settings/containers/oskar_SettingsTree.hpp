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
 * Each node in the tree consists of a SettingsItem which is
 *
 * By attaching a file handler with the method @fn set_file_handler settings
 * can be read from a designated settings file.
 *
 * Settings within the tree are addressed with a key which takes the form of a
 * delimited string using the separator specified in the constructor.
 *
 * Example:
 * @code{.cpp}
 * @endcode
 */

class SettingsTree
{
 public:
    /*! Constructor */
    SettingsTree(char key_separator = '/');

    /*! Destructor */
    ~SettingsTree();

    /*! Attach a settings file handler. */
    /*!
     * Assign a settings file handler which will give the SettingsTree object
     * the ability to read and write settings files.
     * Note that ownership of the handler is not transferred to this object.
     *
     * @param handler SettingsFileHandler object to attach.
     * @param name    Path of the settings file to use.
     */
    void set_file_handler(SettingsFileHandler* handler,
                          const std::string& name = std::string());

    /*! Set or update the file name path of the associated settings file. */
    void set_file_name(const std::string& name);

    /*! Set all of the settings in the tree to their default value. */
    void set_defaults();

    /*! A new entry to the end of the current settings key group prefix. */
    /*!
     * Convenience method for working with settings keys. When working with
     * settings through accessor methods which take a key as the first
     * argument, the specified key will be prefixed with the set of groups
     * registered though this function.
     *
     * @param name Name of the group to add.
     */
    void begin_group(const std::string& name);

    /*! Removes the last level from the current settings key group prefix. */
    /*!
     * Convenience method for working with settings keys to be used in
     * conjunction with @fn begin_group.
     */
    void end_group();

    /*! Returns the current group prefix. */
    /*!
     * Returns the current group prefix defined by the @fn being_group and
     * @fn end_group functions.
     *
     * @return The group prefix string.
     */
    std::string group_prefix() const;

    /*! Add a setting to the Tree, specified by a key and optional meta-data. */
    /*!
     * Adds a setting to the tree at the position defined by the specified
     * @p key modified by the current group prefix.
     *
     * @param key          The settings key.
     * @param label        Brief label given to the setting.
     * @param description  A more detailed description of the setting.
     * @param type_name    String name of the settings type. This should be the
     *                     name of type handled by the SettingsValue class.
     * @param type_default String containing the default value for the type.
     * @param type_parameters String containing any initialisation parameters
     *                        for the type.
     * @param required     If true, mark the setting as required.
     *
     * @return True if successfully added, otherwise false.
     */
    bool add_setting(const std::string& key,
                     const std::string& label = std::string(),
                     const std::string& description = std::string(),
                     const std::string& type_name = std::string(),
                     const std::string& type_default = std::string(),
                     const std::string& type_parameters = std::string(),
                     bool required = false);

    /*! Begin a logical group of dependencies for the current setting. */
    /*!
     * @param logic Combination logic of dependencies in the group.
     * @return True if successful, otherwise false.
     */
    bool begin_dependency_group(const std::string& logic = "AND");

    /*! Closes the current logical group of dependencies. */
    void end_dependency_group();

    /*! Adds a dependency to the current setting. */
    /*!
     * Adds a dependency to the current setting based on the value of
     * another setting specified by the @p dependency_key. Settings are
     * enabled or disabled based on their dependencies. A dependency is
     * satisfied if the value of the setting at the key specified by
     * @p dependency_key evaluated with the specified @p condition found to be
     * True. A setting can have multiple dependencies by calling this function
     * multiple times and the combination of multiple dependencies can be
     * controlled though the functions @p begin_dependency_group and
     * @p end_dependency_group.
     *
     * Allowed conditions are: EQ, NE, GT, GE, LT, LE.
     *
     * @param dependency_key Full settings key which the current key is
     *                       dependent on.
     * @param value          The required value for the dependency to be
     *                       satisfied.
     * @param condition      The condition logic to be satisfied.
     * @return True if successfully added, otherwise false.
     */
    bool add_dependency(const std::string& dependency_key,
                        const std::string& value,
                        const std::string& condition = "EQ");

    /*! Set the value of the setting at the specified @p key. */
    /*!
     * Sets the value of the setting at the specified key. The current group
     * prefix, if defined will be appended to the key.
     *
     * @param key Settings key.
     * @param value Value of the setting.
     * @param write If true and a SettingsFileHandler has been set, update
     *              the settings file on setting the value.
     * @return True is successful, otherwise false.
     *
     * TODO-BM: implement alternative versions of this function with
     * typed value arguments?
     */
    bool set_value(const std::string& key, const std::string& value,
                   bool write = true);

    /*! Return a pointer to SettingsItem object at the specified @p key. */
    /*!
     * Method to give access to the item at the specified @p key. Note the
     * settings key given to this function will be modified by the
     * current group prefix.
     *
     * @param key The settings key.
     * @return SettingsItem pointer.
     */
    const SettingsItem* item(const std::string& key) const;

    /*! Return a pointer to SettingsValue object at the specified @p key. */
    /*!
     * Method to give access to the value of the item at the specified @p key.
     * Note the settings key given to this function will be modified by the
     * current group prefix.
     *
     * @param key The settings key.
     * @return SettingsValue pointer.
     */
    const SettingsValue* value(const std::string& key) const;

    /*! Operator overload of the @p value method. */
    /*!
     * Operator to give access to the value of the item at the specified @p key.
     * Note the settings key given to this function will be modified by the
     * current group prefix.
     *
     * @param key The settings key.
     * @return SettingsValue pointer.
     */
    const SettingsValue* operator[](const std::string& key) const;

    /*! Returns the total number of settings items in the tree. */
    int num_items() const;

    /*! Returns the number of settings items in the tree which have a value. */
    int num_settings() const;

    /*! Utility method which prints a formatted summary of the settings tree. */
    void print() const;

    /*! Clears the tree, removing all items. */
    void clear();

    /*! Saves the settings tree to a settings file. */
    /*!
     * Saves the settings tree to a settings file. This function requires
     * that a SettingsFileHandler has already been specified via the
     * @fn set_file_handler method. If the @p file_name is specified
     * it will be used, however if left empty (the default argument) the
     * existing setting file will be used if it has previously been specified.
     *
     * @param file_name File name of the settings file.
     * @return True if successful, otherwise false.
     */
    bool save(const std::string& file_name = std::string()) const;

    /*! Loads settings values from a settings file. */
    /*!
     * Loads values from a settings file into the settings tree. This function
     * requires that a SettingsFileHandler has already been specified via the
     * @fn set_file_handler method. If the @p file_name is specified it will
     * be used, however if left empty (the default argument) the existing
     * settings file will be used if previously specified.
     *
     * The function also returns a vector of the key, value pairs of settings
     * that have been identified as failed to load by the file handler.
     *
     * @param invalid   Vector of key, value pairs of settings that have
     *                  not been recognised as valid settings.
     * @param file_name Settings file path name from which to load.
     * @return True if successful, otherwise false.
     */
    bool load(std::vector<std::pair<std::string, std::string> >& invalid,
            const std::string& file_name = std::string());

    /*! Return the (application) version string in the settings file. */
    std::string file_version() const;

    /*! Returns true if the a setting with the specified key exists. */
    bool contains(const std::string &key) const;

    /*! Returns true if dependencies of the specified key are satisfied. */
    bool dependencies_satisfied(const std::string& key) const;

    /*! Returns true for required settings with satisfied dependencies. */
    bool is_critical(const std::string& key) const;

    /*! Returns true if the setting has been changed from its default value. */
    bool is_modified() const;

    /*! Returns the root node of the settings tree. */
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
