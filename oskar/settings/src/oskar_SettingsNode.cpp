/*
 * Copyright (c) 2015-2017, The University of Oxford
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

#include "settings/oskar_SettingsNode.h"
#include <cstring>
#include <vector>

using namespace std;

namespace oskar {

struct SettingsNodePrivate
{
    SettingsNode* parent;
    vector<SettingsNode*> children;
    /* Counter used to determine if the item or its children have been set.
     * Incremented by 1 for each item or child with a value not at default. */
    int value_set_counter;
};

SettingsNode::SettingsNode() : SettingsItem()
{
    p = new SettingsNodePrivate;
    p->parent = 0;
    p->value_set_counter = 0;
}

SettingsNode::SettingsNode(const char* key,
             const char* label,
             const char* description,
             const char* type_name,
             const char* type_default,
             const char* type_parameters,
             bool is_required,
             int priority)
: SettingsItem(key, label, description, type_name, type_default,
        type_parameters, is_required, priority)
{
    p = new SettingsNodePrivate;
    p->parent = 0;
    p->value_set_counter = 0;
}

SettingsNode::~SettingsNode()
{
    for (unsigned i = 0; i < p->children.size(); ++i)
        delete p->children[i];
    delete p;
}

bool SettingsNode::value_or_child_set() const
{
    return p->value_set_counter > 0;
}

int SettingsNode::num_children() const
{
    return (int) p->children.size();
}

SettingsNode* SettingsNode::add_child(SettingsNode* node)
{
    if (node->item_type() != SettingsItem::INVALID)
    {
        node->p->parent = this;
        p->children.push_back(node);
        update_priority(node->priority());
        return p->children.back();
    }
    return 0;
}

SettingsNode* SettingsNode::child(int i)
{
    return p->children[i];
}

const SettingsNode* SettingsNode::child(int i) const
{
    return p->children[i];
}

SettingsNode* SettingsNode::child(const char* key)
{
    for (unsigned i = 0; i < p->children.size(); ++i)
        if (!strcmp(p->children[i]->key(), key))
            return p->children[i];
    return 0;
}

const SettingsNode* SettingsNode::child(const char* key) const
{
    for (unsigned i = 0; i < p->children.size(); ++i)
        if (!strcmp(p->children[i]->key(), key))
            return p->children[i];
    return 0;
}

const SettingsNode* SettingsNode::parent() const
{
    return p->parent;
}

int SettingsNode::child_number() const
{
    if (p->parent)
    {
        for (unsigned i = 0; i < p->children.size(); ++i)
            if (p->children[i] == this) return i;
    }
    return 0;
}

bool SettingsNode::set_value(const char* v)
{
    bool was_set = is_set();
    bool ok = SettingsItem::set_value(v);
    bool now_set = is_set();
    // Only update the counter if the set state has changed.
    // ie. changed from default to non default.
    if (was_set != now_set)
        update_value_set_counter_(now_set);
    return ok;
}

void SettingsNode::update_value_set_counter_(bool increment_counter)
{
    if (increment_counter)
        ++(p->value_set_counter);
    else
        --(p->value_set_counter);

    // Recursively set the counter for parents.
    if (p->parent)
        p->parent->update_value_set_counter_(increment_counter);
}

void SettingsNode::update_priority(int priority)
{
    set_priority(SettingsItem::priority() + priority);
    if (p->parent)
        p->parent->update_priority(priority);
}

} // namespace oskar
