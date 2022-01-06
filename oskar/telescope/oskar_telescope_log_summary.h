/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOG_SUMMARY_H_
#define OSKAR_TELESCOPE_LOG_SUMMARY_H_

/**
 * @file oskar_telescope_log_summary.h
 */

#include <oskar_global.h>
#include <log/oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes a summary of the telescope model to the log.
 *
 * @details
 * Writes a summary of the telescope model to the log.
 *
 * @param[in] telescope  Telescope model structure to summarise.
 * @param[in] status     Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_log_summary(const oskar_Telescope* telescope,
        oskar_Log* log, const int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
