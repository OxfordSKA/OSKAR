/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_option_parser.h"
#include "binary/oskar_binary.h"
#include "binary/private_binary.h"
#include "log/oskar_log.h"
#include "mem/oskar_binary_read_mem.h"
#include "utility/oskar_get_binary_tag_string.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static void print_log(const char* filename, oskar_Log* log, int* status)
{
    int tag_not_present = 0;
    oskar_Binary* h = oskar_binary_create(filename, 'r', status);
    if (*status)
    {
        oskar_binary_free(h);
        return;
    }
    oskar_Mem* temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    oskar_binary_read_mem(h, temp, OSKAR_TAG_GROUP_RUN,
            OSKAR_TAG_RUN_LOG, 0, &tag_not_present);
    oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, status);
    oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0; /* Null-terminate. */
    if (tag_not_present)
    {
        oskar_log_error(log, "Run log not found");
    }
    else
    {
        printf("%s\n", oskar_mem_char(temp));
    }
    oskar_mem_free(temp, status);
    oskar_binary_free(h);
}

static void print_settings(const char* filename, oskar_Log* log, int* status)
{
    int tag_not_present = 0;
    oskar_Binary* h = oskar_binary_create(filename, 'r', status);
    if (*status)
    {
        oskar_binary_free(h);
        return;
    }
    oskar_Mem* temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    oskar_binary_read_mem(h, temp, OSKAR_TAG_GROUP_SETTINGS,
            OSKAR_TAG_SETTINGS, 0, &tag_not_present);
    oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, status);
    oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0; /* Null-terminate. */
    if (tag_not_present)
    {
        oskar_log_error(log, "Settings data not found");
    }
    else
    {
        printf("%s\n", oskar_mem_char(temp));
    }
    oskar_mem_free(temp, status);
    oskar_binary_free(h);
}

static void scan_file(const char* filename, oskar_Log* log, int* status)
{
    int extended_tags = 0, depth = -4, i = 0;
    const char p = 'M';
    oskar_Mem* temp = 0;
    oskar_Binary* h = oskar_binary_create(filename, 'r', status);
    if (*status)
    {
        oskar_binary_free(h);
        return;
    }

    /* Log file header data. */
    const int num_chunks = oskar_binary_num_tags(h);
    oskar_log_section(log, p, "File header in '%s'", filename);
    oskar_log_message(log, p, 0, "File contains %d chunks.", num_chunks);

    /* Iterate all tags in index. */
    oskar_log_section(log, p, "Standard tags:");
    oskar_log_message(log, p, -1, "[%3s] %-23s %5s.%-3s : %-10s (%s)",
            "ID", "TYPE", "GROUP", "TAG", "INDEX", "BYTES");
    oskar_log_message(log, p, depth, "CONTENTS");
    oskar_log_line(log, p, '-');
    for (i = 0; i < num_chunks; ++i)
    {
        size_t num_items = 0;
        if (h->extended[i])
        {
            extended_tags++;
            continue;
        }
        const char group   = (char) (h->id_group[i]);
        const char tag     = (char) (h->id_tag[i]);
        const char type    = (char) (h->data_type[i]);
        const int idx      = h->user_index[i];
        const size_t bytes = h->payload_size_bytes[i];

        /* Display tag data. */
        oskar_log_message(log, p, -1,
                "[%3d] %-23s %5d.%-3d : %-10d (%ld bytes)",
                i, oskar_mem_data_type_string(type), group, tag, idx, bytes);

        /* Display more info if available. */
        const char* label = oskar_get_binary_tag_string(group, tag);
        temp = oskar_mem_create(type, OSKAR_CPU, 0, status);
        if (bytes <= 512 || type == OSKAR_CHAR)
        {
            oskar_binary_read_mem(h, temp, group, tag, idx, status);
        }
        const int precision = oskar_type_precision((int)type);
        num_items = oskar_mem_length(temp);
        if (oskar_type_is_complex(type)) num_items *= 2;
        if (oskar_type_is_matrix(type)) num_items *= 4;
        switch (precision)
        {
        case OSKAR_CHAR:
        {
            size_t c = 0;
            char* data = oskar_mem_char(temp);
            const int max_string_length = 40;
            const char* fmt = "%s: %.*s";
            for (c = 0; c < bytes && c < oskar_mem_length(temp); ++c)
            {
                if (data[c] < 32 && data[c] != 0) data[c] = ' ';
            }
            if (bytes > max_string_length) fmt = "%s: %.*s ...";
            oskar_log_message(log, p, depth, fmt, label,
                    max_string_length, oskar_mem_char(temp));
            break;
        }
        case OSKAR_INT:
        {
            const int* data = oskar_mem_int_const(temp, status);
            switch (num_items)
            {
            case 0:
                oskar_log_message(log, p, depth, "%s", label);
                break;
            case 1:
                oskar_log_message(log, p, depth, "%s: %d", label, data[0]);
                break;
            case 2:
                oskar_log_message(log, p, depth, "%s: [%d, %d]",
                        label, data[0], data[1]);
                break;
            case 3:
                oskar_log_message(log, p, depth, "%s: [%d, %d, %d]",
                        label, data[0], data[1], data[2]);
                break;
            case 4:
                oskar_log_message(log, p, depth, "%s: [%d, %d, %d, %d]",
                        label, data[0], data[1], data[2], data[3]);
                break;
            case 5:
                oskar_log_message(log, p, depth, "%s: [%d, %d, %d, %d, %d]",
                        label, data[0], data[1], data[2], data[3], data[4]);
                break;
            case 6:
                oskar_log_message(log, p, depth, "%s: [%d, %d, %d, %d, %d, %d]",
                        label, data[0], data[1], data[2], data[3], data[4],
                        data[5]);
                break;
            default:
                oskar_log_message(log, p, depth,
                        "%s: [%d, %d, %d, %d, %d, %d ...]",
                        label, data[0], data[1], data[2], data[3], data[4],
                        data[5]);
            }
            break;
        }
        case OSKAR_SINGLE:
        {
            const float* data = oskar_mem_float_const(temp, status);
            switch (num_items)
            {
            case 1:
                if ((fabs(data[0]) < 1e3 && fabs(data[0]) > 1e-3) ||
                        data[0] == 0.0)
                {
                    oskar_log_message(log, p, depth, "%s: %.3f",
                            label, data[0]);
                }
                else
                {
                    oskar_log_message(log, p, depth, "%s: %.3e",
                            label, data[0]);
                }
                break;
            case 2:
                oskar_log_message(log, p, depth, "%s: [%.3e, %.3e]",
                        label, data[0], data[1]);
                break;
            default:
                if (num_items > 2)
                {
                    oskar_log_message(log, p, depth,
                            "%s: [%.3e, %.3e ...]", label, data[0], data[1]);
                }
                else
                {
                    oskar_log_message(log, p, depth, "%s", label);
                }
            }
            break;
        }
        case OSKAR_DOUBLE:
        {
            const double* data = oskar_mem_double_const(temp, status);
            switch (num_items)
            {
            case 1:
                if ((fabs(data[0]) < 1e3 && fabs(data[0]) > 1e-3) ||
                        data[0] == 0.0)
                {
                    oskar_log_message(log, p, depth, "%s: %.3f",
                            label, data[0]);
                }
                else
                {
                    oskar_log_message(log, p, depth, "%s: %.3e",
                            label, data[0]);
                }
                break;
            case 2:
                oskar_log_message(log, p, depth, "%s: [%.3e, %.3e]",
                        label, data[0], data[1]);
                break;
            default:
                if (num_items > 2)
                {
                    oskar_log_message(log, p, depth,
                            "%s: [%.3e, %.3e ...]", label, data[0], data[1]);
                }
                else
                {
                    oskar_log_message(log, p, depth, "%s", label);
                }
            }
            break;
        }
        default:
            break;
        }
        oskar_mem_free(temp, status);
    }

    /* Iterate extended tags in index. */
    if (extended_tags)
    {
        oskar_log_section(log, p, "Extended tags:");
        oskar_log_message(log, p, -1, "[%3s] %-23s (%s)",
                "ID", "TYPE", "BYTES");
        oskar_log_message(log, p, depth, "%s.%s : %s", "GROUP", "TAG", "INDEX");
        oskar_log_line(log, p, '-');
        for (i = 0; i < num_chunks; ++i)
        {
            if (!h->extended[i]) continue;
            const char* group  = h->name_group[i];
            const char* tag    = h->name_tag[i];
            const char type    = (char) (h->data_type[i]);
            const int idx      = h->user_index[i];
            const size_t bytes = h->payload_size_bytes[i];

            /* Display tag data. */
            oskar_log_message(log, p, -1, "[%3d] %-23s (%d bytes)",
                    i, oskar_mem_data_type_string(type), bytes);
            oskar_log_message(log, p, depth, "%s.%s : %d", group, tag, idx);
        }
    }
    oskar_binary_free(h);
}


int main(int argc, char** argv)
{
    oskar::OptionParser opt("oskar_binary_file_query", oskar_version_string());
    opt.set_description("List a summary of the contents of an OSKAR binary file.");
    opt.add_required("binary file", "Path to an OSKAR binary file.");
    opt.add_flag("-l", "Display the log.", false, "--log");
    opt.add_flag("-s", "Display the settings file.", false, "--opts");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* filename = opt.get_arg();

    int error = 0;
    bool display_log = opt.is_set("-l") ? true : false;
    bool display_settings = opt.is_set("-s") ? true : false;

    oskar_Log* log = 0;
    oskar_log_set_file_priority(log, OSKAR_LOG_NONE);
    oskar_log_set_term_priority(log, OSKAR_LOG_STATUS);

    if (display_log)
    {
        print_log(filename, log, &error);
    }
    else if (display_settings)
    {
        print_settings(filename, log, &error);
    }
    else
    {
        scan_file(filename, log, &error);
    }

    if (error)
    {
        oskar_log_error(log, oskar_get_error_string(error));
    }

    return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
