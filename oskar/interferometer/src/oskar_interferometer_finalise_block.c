/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"

#include "convert/oskar_convert_ecef_to_uvw.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_get_memory_usage.h"

#ifdef __cplusplus
extern "C" {
#endif

static unsigned int disp_width(unsigned int v);

oskar_VisBlock* oskar_interferometer_finalise_block(oskar_Interferometer* h,
        int block_index, int* status)
{
    int i, i_active;
    oskar_VisBlock *b0 = 0, *b = 0;
    if (*status) return 0;

    /* The visibilities must be copied back
     * at the end of the block simulation. */

    /* Combine all vis blocks into the first one. */
    i_active = (block_index + 1) % 2;
    b0 = h->d[0].vis_block_cpu[!i_active];
    if (!h->coords_only)
    {
        oskar_Mem *xc0 = 0, *ac0 = 0;
        xc0 = oskar_vis_block_cross_correlations(b0);
        ac0 = oskar_vis_block_auto_correlations(b0);
        for (i = 1; i < h->num_devices; ++i)
        {
            b = h->d[i].vis_block_cpu[!i_active];
            if (oskar_vis_block_has_cross_correlations(b))
                oskar_mem_add(xc0, xc0, oskar_vis_block_cross_correlations(b),
                        0, 0, 0, oskar_mem_length(xc0), status);
            if (oskar_vis_block_has_auto_correlations(b))
                oskar_mem_add(ac0, ac0, oskar_vis_block_auto_correlations(b),
                        0, 0, 0, oskar_mem_length(ac0), status);
        }
    }

    /* Calculate (u,v,w) coordinates for the block. */
    if (oskar_vis_block_has_cross_correlations(b0))
    {
        const oskar_Mem *x, *y, *z;
        x = oskar_telescope_station_measured_offset_ecef_metres_const(h->tel, 0);
        y = oskar_telescope_station_measured_offset_ecef_metres_const(h->tel, 1);
        z = oskar_telescope_station_measured_offset_ecef_metres_const(h->tel, 2);
        oskar_convert_ecef_to_uvw(
                oskar_telescope_num_stations(h->tel), x, y, z,
                oskar_telescope_phase_centre_ra_rad(h->tel),
                oskar_telescope_phase_centre_dec_rad(h->tel),
                oskar_vis_block_num_times(b0),
                oskar_vis_header_time_start_mjd_utc(h->header),
                oskar_vis_header_time_inc_sec(h->header) / 86400.0,
                oskar_vis_block_start_time_index(b0), h->ignore_w_components,
                oskar_vis_block_station_uvw_metres(b0, 0),
                oskar_vis_block_station_uvw_metres(b0, 1),
                oskar_vis_block_station_uvw_metres(b0, 2),
                oskar_vis_block_baseline_uvw_metres(b0, 0),
                oskar_vis_block_baseline_uvw_metres(b0, 1),
                oskar_vis_block_baseline_uvw_metres(b0, 2), status);
    }

    /* Add uncorrelated system noise to the combined visibilities. */
    if (!h->coords_only && oskar_telescope_noise_enabled(h->tel))
        oskar_vis_block_add_system_noise(b0, h->header, h->tel,
                block_index, h->temp, status);

    /* Print status message. */
    if (!*status)
    {
        const int num_blocks = oskar_interferometer_num_vis_blocks(h);
        oskar_log_message(h->log, 'S', 0, "Block %*i/%i (%3.0f%%) "
                "complete. Simulation time elapsed: %.3f s",
                disp_width(num_blocks), block_index + 1, num_blocks,
                100.0 * (block_index + 1) / (double)num_blocks,
                oskar_timer_elapsed(h->tmr_sim));
    }

    /* Return a pointer to the block. */
    return b0;
}

static unsigned int disp_width(unsigned int v)
{
    return (v >= 100000u) ? 6 : (v >= 10000u) ? 5 : (v >= 1000u) ? 4 :
            (v >= 100u) ? 3 : (v >= 10u) ? 2u : 1u;
    /* return v == 1u ? 1u : (unsigned)log10(v)+1 */
}

#ifdef __cplusplus
}
#endif
