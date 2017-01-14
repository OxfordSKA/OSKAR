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

#include "apps/oskar_OptionParser.h"
#include "apps/oskar_settings_log.h"
#include "imager/oskar_imager.h"
#include "log/oskar_log.h"
#include "settings/oskar_SettingsTree.h"
#include "settings/oskar_SettingsDeclareXml.h"
#include "settings/oskar_SettingsFileHandlerIni.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include "apps/xml/oskar_imager_xml_all.h"

#include <cstdio>
#include <cstdlib>

using namespace oskar;
using std::vector;
using std::string;
using std::pair;

static const char settings_def[] = oskar_imager_XML_STR;

int main(int argc, char** argv)
{
    int i, e = 0, end;
    vector<pair<string, string> > failed_keys;
    oskar_Log* log = 0;

    OptionParser opt("oskar_imager", oskar_version_string(), settings_def);
    opt.add_settings_options();
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* settings_file = opt.get_arg(0);

    // Declare settings.
    SettingsTree s;
    settings_declare_xml(&s, settings_def);
    SettingsFileHandlerIni handler("oskar_imager", oskar_version_string());
    s.set_file_handler(&handler);

    // Create the log if necessary.
    if (!opt.is_set("--get") && !opt.is_set("--set"))
    {
        int priority = opt.is_set("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
        log = oskar_log_create(OSKAR_LOG_MESSAGE, priority);
        oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);
        oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    }

    // Load the settings file.
    if (!s.load(settings_file, failed_keys))
    {
        oskar_log_error(log, "Failed to read settings file.");
        if (log) oskar_log_free(log);
        return EXIT_FAILURE;
    }

    // Get/set setting if necessary.
    if (opt.is_set("--get"))
    {
        printf("%s\n", s.to_string(opt.get_arg(1), &e).c_str());
        return !e ? 0 : EXIT_FAILURE;
    }
    else if (opt.is_set("--set"))
    {
        const char* key = opt.get_arg(1);
        const char* val = opt.get_arg(2);
        bool ok = val ? s.set_value(key, val) : s.set_default(key);
        if (!ok) oskar_log_error(log, "Failed to set '%s'='%s'", key, val);
        return ok ? 0 : EXIT_FAILURE;
    }

    // Write settings to log.
    oskar_log_set_keep_file(log, 0);
    oskar_settings_log(&s, log, failed_keys);

    // Create imager and set values from settings.
    s.begin_group("image");
    int prec = s.to_int("double_precision", &e) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Imager* h = oskar_imager_create(prec, &e);
    oskar_imager_set_log(h, log);
    if (!s.starts_with("cuda_device_ids", "all", &e))
    {
        vector<int> ids = s.to_int_list("cuda_device_ids", &e);
        if (ids.size() > 0) oskar_imager_set_gpus(h, ids.size(), &ids[0], &e);
    }
    vector<string> files = s.to_string_list("input_vis_data", &e);
    int num_files = files.size();
    char** input_files = (char**) calloc(num_files, sizeof(char*));
    for (i = 0; i < num_files; ++i)
    {
        input_files[i] = (char*) calloc(1 + files[i].length(), sizeof(char));
        strcpy(input_files[i], files[i].c_str());
    }
    oskar_imager_set_input_files(h, num_files, input_files, &e);
    for (i = 0; i < num_files; ++i)
        free(input_files[i]);
    free(input_files);
    oskar_imager_set_scale_norm_with_num_input_files(h,
            s.to_int("scale_norm_with_num_input_files", &e));
    oskar_imager_set_ms_column(h, s.to_string("ms_column", &e).c_str(), &e);
    oskar_imager_set_output_root(h, s.to_string("root_path", &e).c_str(), &e);
    oskar_imager_set_image_type(h, s.to_string("image_type", &e).c_str(), &e);
    if (s.to_int("specify_cellsize", &e))
        oskar_imager_set_cellsize(h, s.to_double("cellsize_arcsec", &e));
    else
        oskar_imager_set_fov(h, s.to_double("fov_deg", &e));
    oskar_imager_set_size(h, s.to_int("size", &e), &e);
    oskar_imager_set_channel_snapshots(h, s.to_int("channel_snapshots", &e));
    oskar_imager_set_channel_start(h, s.to_int("channel_start", &e));
    end = s.starts_with("channel_end", "max", &e) ? -1 :
            s.to_int("channel_end", &e);
    oskar_imager_set_channel_end(h, end);
    oskar_imager_set_time_snapshots(h, s.to_int("time_snapshots", &e));
    oskar_imager_set_time_start(h, s.to_int("time_start", &e));
    end = s.starts_with("time_end", "max", &e) ? -1 : s.to_int("time_end", &e);
    oskar_imager_set_time_end(h, end);
    oskar_imager_set_uv_filter_min(h, s.to_double("uv_filter_min", &e));
    double uv_max = s.starts_with("uv_filter_max", "max", &e) ? -1.0 :
            s.to_double("uv_filter_max", &e);
    oskar_imager_set_uv_filter_max(h, uv_max);
    oskar_imager_set_algorithm(h, s.to_string("algorithm", &e).c_str(), &e);
    oskar_imager_set_weighting(h, s.to_string("weighting", &e).c_str(), &e);
    if (s.starts_with("algorithm", "FFT", &e) ||
            s.starts_with("algorithm", "fft", &e))
    {
        oskar_imager_set_grid_kernel(h,
                s.to_string("fft/kernel_type", &e).c_str(),
                s.to_int("fft/support", &e),
                s.to_int("fft/oversample", &e), &e);
    }
    if (!s.starts_with("wproj/num_w_planes", "auto", &e))
        oskar_imager_set_num_w_planes(h, s.to_int("wproj/num_w_planes", &e));
    oskar_imager_set_fft_on_gpu(h, s.to_int("fft/use_gpu", &e));
    oskar_imager_set_generate_w_kernels_on_gpu(h,
            s.to_int("wproj/generate_w_kernels_on_gpu", &e));
    if (s.first_letter("direction", &e) == 'R')
    {
        oskar_imager_set_direction(h,
                s.to_double("direction/ra_deg", &e),
                s.to_double("direction/dec_deg", &e));
    }

    // Make the images.
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
    if (!e)
    {
        oskar_log_section(log, 'M', "Starting imager...");
        oskar_timer_resume(tmr);
        oskar_imager_run(h, 0, 0, 0, 0, &e);
    }

    // Check for errors.
    if (!e)
        oskar_log_message(log, 'M', 0, "Imaging completed in %.3f sec.",
                oskar_timer_elapsed(tmr));
    else
        oskar_log_error(log, "Run failed with code %i: %s.", e,
                oskar_get_error_string(e));

    // Free memory.
    oskar_timer_free(tmr);
    oskar_imager_free(h, &e);
    oskar_log_free(log);

    return e;
}
