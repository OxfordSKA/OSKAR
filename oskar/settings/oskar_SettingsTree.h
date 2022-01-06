/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_TREE_H_
#define OSKAR_SETTINGS_TREE_H_

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus

namespace oskar {

/*
 * Forward declarations only - don't include any other headers in this file!
 *
 * We shouldn't use std::string in interfaces because of ABI
 * incompatibilities with different versions of the C++ "standard" library.
 *
 * (This caused various "unresolved symbol" errors in the Python
 * interface, for example.)
 *
 * For methods which we expect will be called from elsewhere,
 * use const char* instead, as this does work.
 */

class SettingsDependency;
class SettingsDependencyGroup;
class SettingsFileHandler;
class SettingsItem;
class SettingsKey;
class SettingsNode;
class SettingsValue;
struct SettingsTreePrivate;

/**
 * @brief Settings container for use with OSKAR applications.
 *
 * @details
 * High level representation of a tree of OSKAR application settings in memory.
 * Each node in the tree consists of a SettingsNode, which is a
 * specialisation of a SettingsItem.
 *
 * By attaching a file handler with the method @fn set_file_handler settings
 * can be read from a settings file.
 *
 * Settings within the tree are addressed with a key which takes the form of a
 * delimited string using the separator specified in the constructor.
 */
class OSKAR_SETTINGS_EXPORT SettingsTree
{
 public:
    /*! Constructor */
    SettingsTree(char key_separator = '/');

    /*! Destructor */
    ~SettingsTree();

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
     * @param condition      The condition logic to be satisfied. Default "EQ".
     * @return True if successfully added, otherwise false.
     */
    bool add_dependency(const char* dependency_key,
                        const char* value,
                        const char* condition = "EQ");

    /*! Records a setting key-value pair that failed to validate on load. */
    /*!
     * When loading a file, if a key is unknown, it can be recorded by calling
     * this method. This is used by the file handler classes.
     *
     * @param key          The unknown settings key.
     * @param value        The unknown settings value.
     */
    void add_failed(const char* key, const char* value);

    /*! Add a setting to the tree, specified by a key and optional meta-data. */
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
     * @param priority     Priority level (importance) of the setting.
     *
     * @return True if successfully added, otherwise false.
     */
    bool add_setting(const char* key,
                     const char* label = 0,
                     const char* description = 0,
                     const char* type_name = 0,
                     const char* type_default = 0,
                     const char* type_parameters = 0,
                     bool required = false,
                     int priority = 0);

    /*! Begin a logical group of dependencies for the current setting. */
    /*!
     * @param logic Combination logic of dependencies in the group
     *              ("AND" or "OR"). Default "AND".
     * @return True if successful, otherwise false.
     */
    bool begin_dependency_group(const char* logic = "AND");

    /*! Add a new entry to the end of the current settings key group prefix. */
    /*!
     * Convenience method for working with settings keys. When working with
     * settings through accessor methods which take a key as the first
     * argument, the specified key will be prefixed with the set of groups
     * registered though this function.
     *
     * @param name Name of the group to add.
     */
    void begin_group(const char* name);

    /*! Clears the tree, removing all items. */
    void clear();

    /*! Clears the current settings key group prefix. */
    /*!
     * Convenience method to clear the current settings group, if set using
     * @fn begin_group.
     */
    void clear_group();

    /*! Returns true if a setting with the specified key exists in the tree. */
    bool contains(const char* key) const;

    /*! Returns true if dependencies of the specified key are satisfied. */
    bool dependencies_satisfied(const char* key) const;

    /*! Closes the current logical group of dependencies. */
    void end_dependency_group();

    /*! Removes the last level from the current settings key group prefix. */
    /*!
     * Convenience method for working with settings keys to be used in
     * conjunction with @fn begin_group.
     */
    void end_group();

    /*! Return the i-th key that failed to load. */
    const char* failed_key(int i) const;

    /*! Return the value of the i-th key that failed to load. */
    const char* failed_key_value(int i) const;

    /*! Return a pointer to the file handler, if one has been set. */
    const SettingsFileHandler* file_handler() const;

    /*! Return a pointer to the file handler, if one has been set. */
    SettingsFileHandler* file_handler();

    /*! Return the file name path of the associated settings file. */
    const char* file_name() const;

    /*! Return the first letter of the setting with the given key.
     *
     * @param key          The settings key.
     * @param status       Status return code.
     */
    char first_letter(const char* key, int* status) const;

    /*! Deletes the settings tree. */
    static void free(SettingsTree* h);

    /*! Returns the current group prefix. */
    /*!
     * Returns the current group prefix defined by the @fn begin_group and
     * @fn end_group functions.
     *
     * @return The group prefix string.
     */
    const char* group_prefix() const;

    /*! Returns true for required settings with satisfied dependencies. */
    bool is_critical(const char* key) const;

    /*! Returns true if the setting has been changed from its default value. */
    bool is_modified() const;

    /*! Return a pointer to SettingsItem object at the specified @p key. */
    /*!
     * Method to give access to the item at the specified @p key.
     * Note the settings key given to this function will be modified by the
     * current group prefix.
     *
     * @param key The settings key.
     * @return SettingsItem pointer.
     */
    const SettingsItem* item(const char* key) const;

    /*! Loads settings values from a settings file. */
    /*!
     * Loads values from a settings file into the settings tree. This function
     * requires that a SettingsFileHandler has already been specified via the
     * @fn set_file_handler method. If the @p file_name is specified it will
     * be used, however if left empty (the default argument) the existing
     * settings file will be used if previously specified.
     *
     * @param file_name Settings file path name from which to load.
     * @return True if successful, otherwise false.
     */
    bool load(const char* file_name = 0);

    /*! Returns the number of keys that failed to load. */
    int num_failed_keys() const;

    /*! Returns the total number of settings items in the tree. */
    int num_items() const;

    /*! Returns the number of settings items in the tree which have a value. */
    int num_settings() const;

    /*! Utility method which prints a formatted summary of the settings tree. */
    void print() const;

    /*! Returns the root node of the settings tree. */
    const SettingsNode* root_node() const;

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
    bool save(const char* file_name = 0) const;

    /*! Returns the key separator character. */
    char separator() const;

    /*! Return a pointer to SettingsValue object at the specified @p key. */
    /*!
     * Method to give access to the value of the item at the specified @p key.
     * Note the settings key given to this function will be modified by the
     * current group prefix.
     *
     * @param key The settings key.
     * @return SettingsValue pointer.
     */
    const SettingsValue* settings_value(const char* key) const;

    /*! Set the setting at the specified @p key to its default value. */
    /*!
     * Sets the setting at the specified key to its default value.
     * The current group prefix will be prepended to the key.
     *
     * @param key Settings key.
     * @param write If true and a SettingsFileHandler has been set, update
     *              the settings file on setting the value.
     * @return True is successful, otherwise false.
     */
    bool set_default(const char* key, bool write = true);

    /*! Set all of the settings in the tree to their default value. */
    void set_defaults();

    /*! Attach a settings file handler. */
    /*!
     * Assign a settings file handler which will give the SettingsTree object
     * the ability to read and write settings files.
     * Note that ownership of the handler *IS* transferred to this object.
     *
     * @param handler SettingsFileHandler object to attach.
     * @param name    Path of the settings file to use.
     */
    void set_file_handler(SettingsFileHandler* handler, const char* name = 0);

    /*! Set or update the file name path of the associated settings file. */
    void set_file_name(const char* name);

    /*! Set the value of the setting at the specified @p key. */
    /*!
     * Sets the value of the setting at the specified key. The current group
     * prefix will be prepended to the key.
     *
     * @param key Settings key.
     * @param value Value of the setting.
     * @param write If true and a SettingsFileHandler has been set, update
     *              the settings file on setting the value.
     * @return True is successful, otherwise false.
     */
    bool set_value(const char* key, const char* value, bool write = true);

    /*! Sets multiple settings from a string list. */
    /*!
     * Sets all settings in the given list. Settings keys and values must
     * be interleaved, and the @p num_strings parameter must be the total
     * size of the list (all the keys and all the values). If the number
     * of strings is not divisible by 2, nothing will be set and the method
     * will return false.
     *
     * @param num_strings  The total number of strings in the list.
     * @param list         The string list.
     * @param write        If true and a SettingsFileHandler has been set,
     *                     update the settings file on setting the value.
     * @return True is successful, otherwise false.
     */
    bool set_values(int num_strings, const char* const* list,
            bool write = true);

    bool starts_with(const char* key, const char* str, int* status) const;
    double to_double(const char* key, int* status) const;
    const double* to_double_list(const char* key, int* size, int* status) const;
    int to_int(const char* key, int* status) const;
    const int* to_int_list(const char* key, int* size, int* status) const;
    const char* to_string(const char* key, int* status) const;
    const char* const* to_string_list(const char* key, int* size,
            int* status) const;

    /*! Returns the value of a specified key. */
    /*!
     * Operator to give access to the string value of the item at the
     * specified @p key.
     * Note the settings key given to this function will be modified by the
     * current group prefix.
     *
     * @param key The settings key.
     * @return The settings value as a string.
     */
    const char* operator[](const char* key) const;

 private:
    SettingsTree(const SettingsTree&);
    bool dependencies_satisfied(const SettingsDependencyGroup* group) const;
    bool dependency_satisfied(const SettingsDependency* dep) const;
    const SettingsNode* find_node(const SettingsNode* node,
            const SettingsKey& full_key, int depth) const;
    SettingsNode* find_node(SettingsNode* node,
            const SettingsKey& full_key, int depth);
    bool is_node_critical(const SettingsNode* node) const;
    bool parent_dependencies_satisfied(const SettingsNode*) const;
    void print_from_node(const SettingsNode* node, int depth = 0) const;
    void set_defaults_from_node(SettingsNode* node);

 private:
    mutable bool modified_;
    SettingsTreePrivate* p;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* include guard */
