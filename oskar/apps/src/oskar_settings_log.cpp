/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "apps/oskar_settings_log.h"

static void oskar_settings_log_private(const oskar::SettingsTree* s,
        oskar_Log* log, const oskar::SettingsNode* node, int depth)
{
    if (!s->dependencies_satisfied(node->key())) return;

    if (node->priority() > 0 || node->value_or_child_set() ||
            node->is_required())
    {
        const std::string& label = node->label();
        const std::string& value = node->value().to_string();
        if (value.size() == 0)
            oskar_log_message(log, 'M', depth, label.c_str());
        else if (value.size() > 35)
            oskar_log_message(log, 'M', depth,
                    "%s: %s", label.c_str(), value.c_str());
        else
            oskar_log_value(log, 'M', depth,
                    label.c_str(), "%s", value.c_str());

        for (int i = 0; i < node->num_children(); ++i)
            oskar_settings_log_private(s, log, node->child(i), depth + 1);
    }
}

void oskar_settings_log(const oskar::SettingsTree* s, oskar_Log* log,
        const std::vector<std::pair<std::string, std::string> >& failed_keys)
{
    const oskar::SettingsNode* node = s->root_node();
    for (int i = 0; i < node->num_children(); ++i)
        oskar_settings_log_private(s, log, node->child(i), 0);
    for (size_t i = 0; i < failed_keys.size(); ++i)
        oskar_log_warning(log, "Ignoring '%s'='%s'",
                failed_keys[i].first.c_str(), failed_keys[i].second.c_str());
}
