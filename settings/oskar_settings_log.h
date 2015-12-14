/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_LOG_SETTINGS_H_
#define OSKAR_LOG_SETTINGS_H_

/**
 * @file oskar_log_settings.h
 */

#include <oskar_global.h>
#include <oskar_Settings.h>
#include <oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes simulator settings to a log.
 *
 * @details
 * This function writes simulator settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_simulator(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes sky settings to a log.
 *
 * @details
 * This function writes sky settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_sky(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes observation settings to a log.
 *
 * @details
 * This function writes all settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_observation(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes telescope settings to a log.
 *
 * @details
 * This function writes telescope settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_telescope(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes interferometer settings to a log.
 *
 * @details
 * This function writes interferometer settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_interferometer(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes beam pattern settings to a log.
 *
 * @details
 * This function writes beam pattern settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_beam_pattern(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes image settings to a log.
 *
 * @details
 * This function writes image settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_image(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes ionosphere settings to a log.
 *
 * @details
 * This function writes ionosphere settings to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_ionosphere(oskar_Log* log, const oskar_Settings* s);

/**
 * @brief
 * Writes element fitting parameters to a log.
 *
 * @details
 * This function writes element fitting parameters to a log.
 *
 * @param[out,in] log  Pointer to a log structure.
 * @param[in] s        Pointer to a populated settings structure.
 */
void oskar_log_settings_element_fit(oskar_Log* log, const oskar_Settings* s);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_LOG_SETTINGS_H_ */
