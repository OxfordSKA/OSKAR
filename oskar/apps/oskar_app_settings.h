/*
 * Copyright (c) 2017, The University of Oxford
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

#ifndef OSKAR_APP_SETTINGS_H_
#define OSKAR_APP_SETTINGS_H_

/**
 * @file oskar_app_settings.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C"
#endif
OSKAR_APPS_EXPORT
const char* oskar_app_settings(const char* app);

#ifdef __cplusplus
#include <settings/oskar_SettingsTree.h>

/**
 * @brief
 * Returns a SettingsTree for the given application and settings file.
 *
 * @details
 * Returns a SettingsTree for the given application and its settings file.
 * NULL is returned on failure.
 *
 * The settings file is optional. If the string is empty, the default
 * settings tree for the application is returned.
 *
 * @param[in] app             Name of the application.
 * @param[in] settings_file   Path of the settings file to load.
 *
 * @return A handle to the settings tree.
 */
OSKAR_APPS_EXPORT
oskar::SettingsTree* oskar_app_settings_tree(const char* app,
        const char* settings_file);

#endif /* __cplusplus */

#endif /* OSKAR_APP_SETTINGS_H_ */
