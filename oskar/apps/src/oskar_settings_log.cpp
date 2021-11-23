/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_settings_log.h"
#include "log/oskar_log.h"
#include "settings/oskar_SettingsNode.h"
#include <cstring>

using namespace std;

static void oskar_settings_log_private(const oskar::SettingsTree* s,
        oskar_Log* log, const oskar::SettingsNode* node, int depth)
{
    if (!s->dependencies_satisfied(node->key())) return;

    if (node->priority() > 0 || node->value_or_child_set() ||
            node->is_required())
    {
        const char* label = node->label();
        const char* value = node->value();
        const int len = (int) strlen(value);
        if (len == 0)
        {
            oskar_log_message(log, 'M', depth, label);
        }
        else if (len > 35)
        {
            oskar_log_message(log, 'M', depth, "%s: %s", label, value);
        }
        else
        {
            oskar_log_value(log, 'M', depth, label, "%s", value);
        }

        for (int i = 0; i < node->num_children(); ++i)
        {
            oskar_settings_log_private(s, log, node->child(i), depth + 1);
        }
    }
}

void oskar_settings_log(const oskar::SettingsTree* s, oskar_Log* log)
{
    oskar_log_section(log, 'M', "Loaded settings file '%s'", s->file_name());
    const oskar::SettingsNode* node = s->root_node();
    for (int i = 0; i < node->num_children(); ++i)
    {
        oskar_settings_log_private(s, log, node->child(i), 0);
    }
    for (int i = 0; i < s->num_failed_keys(); ++i)
    {
        oskar_log_warning(log, "Ignoring '%s'='%s'",
                s->failed_key(i), s->failed_key_value(i));
    }
}
