/*
 * Copyright (c) 2015-2022, The University of Oxford
 * All rights reserved.
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

#ifndef OSKAR_SETTINGS_NODE_HPP_
#define OSKAR_SETTINGS_NODE_HPP_

/**
 * @file oskar_SettingsNode.h
 */

#include <settings/oskar_SettingsItem.h>

#ifdef __cplusplus

namespace oskar {

struct SettingsNodePrivate;

/*!
 * @class SettingsNode
 *
 * @brief A node within the settings tree, inherits @class SettingsItem.
 *
 * @details
 * Settings tree node class which specialises a settings item for use within
 * a tree structure.
 */
class OSKAR_SETTINGS_EXPORT SettingsNode : public SettingsItem
{
 public:
    /*! Default constructor */
    SettingsNode();

    /*! Constructor */
    SettingsNode(const char* key,
                 const char* label = 0,
                 const char* description = 0,
                 const char* type_name = 0,
                 const char* type_default = 0,
                 const char* type_parameters = 0,
                 bool is_required = false,
                 int priority = 0);

    /*! Destructor */
    virtual ~SettingsNode();

    /*! Return the number of child nodes */
    int num_children() const;

    /*! Add a child node */
    SettingsNode* add_child(SettingsNode* node);

    /*! Return a pointer to the child node with index @p i */
    SettingsNode* child(int i);

    /*! Return a const pointer to the child node with index @p i */
    const SettingsNode* child(int i) const;

    /*! Return a pointer to the child node with key @p key */
    SettingsNode* child(const char* key);

    /*! Return a const pointer to the child node with key @p key */
    const SettingsNode* child(const char* key) const;

    /*! Return a pointer to the node's parent */
    const SettingsNode* parent() const;

    /*! Return the child index of the node */
    int child_number() const;

    /*! Return true, if the item or one its children has a non-default value. */
    bool value_or_child_set() const;

    /*! Set the value field of the node. */
    bool set_value(const char* value);

 private:
    /* Disable copy constructor. */
    SettingsNode(const SettingsNode&);

    void update_value_set_counter(bool increment_counter);
    void update_priority(int priority);

    SettingsNodePrivate* p;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_NODE_HPP_ */
