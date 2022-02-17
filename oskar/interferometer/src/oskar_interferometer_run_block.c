/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"

#include "convert/oskar_convert_apparent_ra_dec_to_enu_directions.h"
#include "convert/oskar_convert_ecef_to_station_uvw.h"
#include "convert/oskar_convert_mjd_to_gast_fast.h"
#include "correlate/oskar_auto_correlate.h"
#include "correlate/oskar_cross_correlate.h"
#include "interferometer/oskar_evaluate_jones_E.h"
#include "interferometer/oskar_evaluate_jones_K.h"
#include "interferometer/oskar_evaluate_jones_R.h"
#include "interferometer/oskar_evaluate_jones_Z.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

static void sim_baselines(oskar_Interferometer* h, DeviceData* d,
        oskar_Sky* sky, int channel_index_block, int time_index_block,
        int channel_index_sim, int time_index_sim, int* status);
static unsigned int disp_width(unsigned int v);

void oskar_interferometer_run_block(oskar_Interferometer* h, int block_index,
        int device_id, int* status)
{
    int chan_index_start = 0, chan_index_end = 0;
    int time_index_start = 0, time_index_end = 0;
    DeviceData* d = 0;
    if (*status) return;

    /* Check that initialisation has happened. We can't initialise here,
     * as we're already multi-threaded at this point. */
    if (!h->header)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        oskar_log_error(h->log, "Simulator not initalised. "
                "Call oskar_interferometer_check_init() first.");
        return;
    }

    /* Set the GPU to use. (Supposed to be a very low-overhead call.) */
    if (device_id >= 0 && device_id < h->num_gpus)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[device_id], status);
    }

    /* Clear the visibility block. */
    d = &(h->d[device_id]);
    oskar_timer_resume(d->tmr_compute);
    oskar_vis_block_clear(d->vis_block, status);

    /* Set the visibility block meta-data. */
    const int total_chunks = h->num_sky_chunks;
    const int total_chans = h->num_channels;
    const int total_times = h->num_time_steps;
    const int num_blocks_chan = (total_chans + h->max_channels_per_block - 1) /
            h->max_channels_per_block;
    const int i_block_chan = block_index % num_blocks_chan;
    const int i_block_time = block_index / num_blocks_chan;
    const double obs_start_mjd = h->time_start_mjd_utc;
    const double dt_dump_days = h->time_inc_sec / 86400.0;
    chan_index_start = i_block_chan * h->max_channels_per_block;
    chan_index_end = chan_index_start + h->max_channels_per_block - 1;
    time_index_start = i_block_time * h->max_times_per_block;
    time_index_end = time_index_start + h->max_times_per_block - 1;
    if (time_index_end >= total_times)
    {
        time_index_end = total_times - 1;
    }
    if (chan_index_end >= total_chans)
    {
        chan_index_end = total_chans - 1;
    }
    const int num_times_block = 1 + time_index_end - time_index_start;
    const int num_chans_block = 1 + chan_index_end - chan_index_start;

    /* Set the size of the block. */
    oskar_vis_block_resize(d->vis_block, num_times_block, num_chans_block,
            oskar_vis_block_num_stations(d->vis_block), status);
    oskar_vis_block_set_start_time_index(d->vis_block, time_index_start);
    oskar_vis_block_set_start_channel_index(d->vis_block, chan_index_start);

    /* Go though all possible work units in the block. A work unit is defined
     * as the simulation for one time and one sky chunk. */
    while (!h->coords_only)
    {
        oskar_Sky* sky = 0;
        int i_channel = 0;

        oskar_mutex_lock(h->mutex);
        const int i_work_unit = (h->work_unit_index)++;
        oskar_mutex_unlock(h->mutex);
        if ((i_work_unit >= num_times_block * total_chunks) || *status) break;

        /* Convert slice index to chunk/time index. */
        const int i_chunk      = i_work_unit / num_times_block;
        const int i_time       = i_work_unit - i_chunk * num_times_block;
        const int sim_time_idx = time_index_start + i_time;

        /* Copy sky chunk to device only if different from the previous one. */
        if (i_chunk != d->previous_chunk_index)
        {
            oskar_timer_resume(d->tmr_copy);
            oskar_sky_copy(d->chunk, h->sky_chunks[i_chunk], status);
            oskar_timer_pause(d->tmr_copy);
        }
        sky = h->apply_horizon_clip ? d->chunk_clip : d->chunk;

        /* Apply horizon clip if required. */
        if (h->apply_horizon_clip)
        {
            double gast = 0.0, mjd = 0.0;
            mjd = obs_start_mjd + dt_dump_days * (sim_time_idx + 0.5);
            gast = oskar_convert_mjd_to_gast_fast(mjd);
            oskar_timer_resume(d->tmr_clip);
            oskar_sky_horizon_clip(d->chunk_clip, d->chunk, d->tel, gast,
                    d->station_work, status);
            oskar_timer_pause(d->tmr_clip);
        }

        /* Simulate all baselines for all channels for this time and chunk. */
        for (i_channel = 0; i_channel < num_chans_block; ++i_channel)
        {
            if (*status) break;
            const int sim_chan_idx = chan_index_start + i_channel;
            oskar_mutex_lock(h->mutex);
            oskar_log_message(h->log, 'S', 1, "Time %*i/%i, "
                    "Chunk %*i/%i, Channel %*i/%i [Device %i, %i sources]",
                    disp_width(total_times), sim_time_idx + 1, total_times,
                    disp_width(total_chunks), i_chunk + 1, total_chunks,
                    disp_width(total_chans), sim_chan_idx + 1, total_chans,
                    device_id, oskar_sky_num_sources(sky));
            oskar_mutex_unlock(h->mutex);
            sim_baselines(h, d, sky, i_channel, i_time,
                    sim_chan_idx, sim_time_idx, status);
        }
        d->previous_chunk_index = i_chunk;
    }

    /* Copy the visibility block to host memory. */
    const int i_active = block_index % 2; /* Index of the active buffer. */
    oskar_timer_resume(d->tmr_copy);
    oskar_vis_block_copy(d->vis_block_cpu[i_active], d->vis_block, status);
    oskar_timer_pause(d->tmr_copy);
    oskar_timer_pause(d->tmr_compute);
}


static void sim_baselines(oskar_Interferometer* h, DeviceData* d,
        oskar_Sky* sky, int channel_index_block, int time_index_block,
        int channel_index_sim, int time_index_sim, int* status)
{
    /* Get dimensions. */
    const int num_baselines   = oskar_telescope_num_baselines(d->tel);
    const int num_stations    = oskar_telescope_num_stations(d->tel);
    const int num_src         = oskar_sky_num_sources(sky);
    const int num_times_block = oskar_vis_block_num_times(d->vis_block);
    const int num_chans_block = oskar_vis_block_num_channels(d->vis_block);

    /* Return if there are no sources in the chunk,
     * or if block indices requested are outside the block dimensions. */
    if (num_src == 0 ||
            time_index_block >= num_times_block ||
            channel_index_block >= num_chans_block)
    {
        return;
    }

    /* Get the time and frequency of the visibility slice being simulated. */
    const double dt_dump_days = h->time_inc_sec / 86400.0;
    const double t_start = h->time_start_mjd_utc;
    const double t_dump = t_start + dt_dump_days * (time_index_sim + 0.5);
    const double gast_rad = oskar_convert_mjd_to_gast_fast(t_dump);
    const double freq = h->freq_start_hz + channel_index_sim * h->freq_inc_hz;

    /* Scale source fluxes with spectral index and rotation measure. */
    oskar_sky_scale_flux_with_frequency(sky, freq, status);
    const oskar_Mem* const src_flux[] = {
            oskar_sky_I_const(sky),
            oskar_sky_Q_const(sky),
            oskar_sky_U_const(sky),
            oskar_sky_V_const(sky)
    };

    /* Get true station (u,v,w) coordinates. */
    oskar_telescope_uvw(d->tel,
            1, /* Use true coordinates. */
            0, /* Do not ignore w-components. */
            1, /* Single time sample. */
            t_start, dt_dump_days, time_index_sim,
            d->uvw[0], d->uvw[1], d->uvw[2], 0, 0, 0, status);
    const oskar_Mem* const uvw[] = { d->uvw[0], d->uvw[1], d->uvw[2] };

    /* Get source direction cosines. */
    const oskar_Mem* lmn[3];
    if (oskar_telescope_phase_centre_coord_type(d->tel) == OSKAR_COORDS_AZEL)
    {
        /* Calculate ENU source direction cosines for array centre. */
        const double lst_rad = gast_rad + oskar_telescope_lon_rad(d->tel);
        oskar_convert_apparent_ra_dec_to_enu_directions(num_src,
                oskar_sky_ra_rad_const(sky), oskar_sky_dec_rad_const(sky),
                lst_rad, oskar_telescope_lat_rad(d->tel),
                0, d->lmn[0], d->lmn[1], d->lmn[2], status);

        /* Reference direction cosine scratch arrays. */
        lmn[0] = d->lmn[0];
        lmn[1] = d->lmn[1];
        lmn[2] = d->lmn[2];
    }
    else
    {
        /* Reference source direction cosines from sky model. */
        lmn[0] = oskar_sky_l_const(sky);
        lmn[1] = oskar_sky_m_const(sky);
        lmn[2] = oskar_sky_n_const(sky);
    }

    /* Set dimensions of Jones matrices. */
    if (d->R)
    {
        oskar_jones_set_size(d->R, num_stations, num_src, status);
    }
    oskar_jones_set_size(d->J, num_stations, num_src, status);
    oskar_jones_set_size(d->E, num_stations, num_src, status);
    oskar_jones_set_size(d->K, num_stations, num_src, status);

    /* Evaluate station beam (Jones E: may be matrix). */
    const oskar_Mem* const source_coords[] = {
            oskar_sky_l_const(sky),
            oskar_sky_m_const(sky),
            oskar_sky_n_const(sky)
    };
    oskar_timer_resume(d->tmr_E);
    oskar_evaluate_jones_E(d->E, OSKAR_COORDS_REL_DIR, num_src, source_coords,
            oskar_sky_reference_ra_rad(sky), oskar_sky_reference_dec_rad(sky),
            d->tel, time_index_sim, gast_rad, freq, d->station_work, status);
    oskar_timer_pause(d->tmr_E);

    /* Evaluate parallactic angle (Jones R: matrix), and join with Jones E.
     * TODO Move this into station beam evaluation instead. */
    if (d->R)
    {
        oskar_timer_resume(d->tmr_E);
        oskar_evaluate_jones_R(d->R, num_src,
                oskar_sky_ra_rad_const(sky),
                oskar_sky_dec_rad_const(sky),
                d->tel, gast_rad, status);
        oskar_timer_pause(d->tmr_E);
        oskar_timer_resume(d->tmr_join);
        oskar_jones_join(d->R, d->E, d->R, status);
        oskar_timer_pause(d->tmr_join);
    }

    /* Evaluate interferometer phase (Jones K: scalar). */
    oskar_timer_resume(d->tmr_K);
    oskar_evaluate_jones_K(d->K, num_src,
            lmn[0], lmn[1], lmn[2], uvw[0], uvw[1], uvw[2],
            freq, src_flux[0], h->source_min_jy, h->source_max_jy,
            h->ignore_w_components, status);
    oskar_timer_pause(d->tmr_K);

    /* Multiply Jones matrix chain to get a single block. */
    oskar_timer_resume(d->tmr_join);
    oskar_jones_join(d->J, d->K, d->R ? d->R : d->E, status);
    oskar_timer_pause(d->tmr_join);

    /* Check whether gain model exists.
     * If so, evaluate gains and apply them. */
    if (oskar_gains_defined(oskar_telescope_gains(d->tel)))
    {
        oskar_gains_evaluate(oskar_telescope_gains(d->tel),
                time_index_sim, freq, d->gains, 0, status);
        oskar_jones_apply_station_gains(d->J, d->gains, status);
    }

    /* Calculate output offset. */
    const int offset = num_chans_block * time_index_block + channel_index_block;
    oskar_timer_resume(d->tmr_correlate);

    /* Auto-correlate for this time and channel. */
    if (oskar_vis_block_has_auto_correlations(d->vis_block))
    {
        oskar_auto_correlate(num_src, d->J, src_flux, num_stations * offset,
                oskar_vis_block_auto_correlations(d->vis_block), status);
    }

    /* Cross-correlate for this time and channel. */
    if (oskar_vis_block_has_cross_correlations(d->vis_block))
    {
        const int source_type = oskar_sky_use_extended(sky);
        const oskar_Mem* const src_extended[] = {
            oskar_sky_gaussian_a_const(sky),
            oskar_sky_gaussian_b_const(sky),
            oskar_sky_gaussian_c_const(sky)
        };
        oskar_cross_correlate(
                source_type, num_src, d->J,
                src_flux, lmn, src_extended,
                d->tel, uvw,
                gast_rad, freq, num_baselines * offset,
                oskar_vis_block_cross_correlations(d->vis_block), status);
    }
    oskar_timer_pause(d->tmr_correlate);
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
