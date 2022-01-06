/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_DECLARE_XML_HPP_
#define OSKAR_SETTINGS_DECLARE_XML_HPP_

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus

namespace oskar {

class SettingsTree;

}

/*! Populates a settings tree from the specified @p xml string */
OSKAR_SETTINGS_EXPORT
bool oskar_settings_declare_xml(
        oskar::SettingsTree* settings, const char* xml);

#endif /* __cplusplus */

#endif /* include guard */
