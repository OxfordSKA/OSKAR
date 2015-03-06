/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <cuda_runtime_api.h>

#include <oskar_sim_beam_pattern_new.h>

#include <oskar_beam_pattern_generate_coordinates.h>
#include <oskar_cmath.h>
#include <oskar_convert_mjd_to_gast_fast.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_evaluate_average_cross_power_beam.h>
#include <oskar_evaluate_station_beam.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_file_exists.h>
#include <oskar_jones.h>
#include <oskar_log.h>
#include <oskar_set_up_telescope.h>
#include <oskar_settings_free.h>
#include <oskar_settings_load.h>
#include <oskar_settings_log.h>
#include <oskar_station_work.h>
#include <oskar_telescope.h>
#include <oskar_timer.h>

#include <fits/oskar_fits_write_axis_header.h>
#include <fitsio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated per GPU. */
struct DeviceData
{
    oskar_Jones* jones;
    oskar_Telescope* tel;
    oskar_StationWork* work;
    oskar_Mem *x, *y, *z, *beam_data;
};
typedef struct DeviceData DeviceData;

/* Memory allocated once, on the host. */
struct HostData
{
    /* Input data (settings, pixel positions, telescope model). */
    oskar_Mem *x, *y, *z;
    oskar_Telescope* tel;
    oskar_Settings s;

    /* Metadata. */
    int coord_type, max_chunk_size, num_times, num_channels, num_chunks, type;
    int width, height, num_pixels, num_stations, station_id;
    int time_average, average_cross_power_beam; /* Flags. */
    double lon0, lat0, phase_centre_deg[2], fov_deg[2];
    double start_mjd_utc, delta_t, start_freq_hz, delta_f;

    /* Output data (each slice 1 GB for 4096 pixels in double precision). */
    oskar_Mem *a, *b, *c, *d; /* Holds (accumulated) split polarisations. */
    oskar_Mem* slice_cpu[2]; /* For copy back & write. */

    /* FITS file handles. */
    int num_fits_handles;
    int* fits_data_type;
    fitsfile** handle_fits;

    /* ASCII file handles. */
    int num_ascii_handles;
    int* ascii_data_type;
    FILE** handle_ascii;

    /* Timers. */
    oskar_Timer* tmr_sim;
    oskar_Timer* tmr_load;
};
typedef struct HostData HostData;

enum
{
    TOTAL_INTENSITY,
    VOLTAGE,
    PHASE
};

#define RAD2DEG (180.0 / M_PI)

static void sim_slice(int gpu_id, DeviceData* d, const HostData* h,
        int i_slice, oskar_Mem* slice_cpu, int* next_chunk_index,
        oskar_Log* log, int* status);
static void write_slice(HostData* h, int i_slice, oskar_Mem* image_slice,
        int* status);
static void set_up_host_data(HostData* h, oskar_Log* log, int *status);
static void set_up_device_data(DeviceData* d, const oskar_Settings* s,
        const oskar_Telescope* tel, int* status);
static void free_device_data(int num_gpus, int* cuda_device_ids,
        DeviceData* d, int* status);
static void free_host_data(HostData* h, int* status);
static void slice_to_time_and_channel_index(int slice, int num_times,
        int* i_channel, int* i_time);
static double fov_to_cellsize(double fov_deg, int num_pixels);
static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, int num_channels,
        double centre_deg[2], double fov_deg[2], double start_time_mjd,
        double delta_time_sec, double start_freq_hz, double delta_freq_hz,
        int* status);
static void new_fits(HostData* h, int file_type, const char* name, int* status);
static void new_ascii(HostData* h, int file_type, const char* name);
static unsigned int disp_width(unsigned int value);

void oskar_sim_beam_pattern_new(const char* settings_file,
        oskar_Log* log, int* status)
{
    int i, num_gpus = 0, num_gpus_avail = 0, num_threads = 1;
    int num_slices = 0, next_chunk_index = 0;
    DeviceData* d = 0;
    HostData* h = 0;
    oskar_Settings* s = 0;

    /* Create the host data structure (initialised with all bits zero). */
    h = (HostData*) calloc(1, sizeof(HostData));
    s = &h->s;

    /* Start the load timer. */
    h->tmr_sim   = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_load  = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_start(h->tmr_load);

    /* Load the settings file. */
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    oskar_settings_load(s, log, settings_file, status);
    if (*status)
    {
        free_host_data(h, status);
        return;
    }

    /* Log the relevant settings. (TODO fix/automate these functions) */
    oskar_log_set_keep_file(log, s->sim.keep_log_file);
    oskar_log_set_file_priority(log, s->sim.write_status_to_log_file ?
            OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    oskar_log_settings_simulator(log, s);
    oskar_log_settings_observation(log, s);
    oskar_log_settings_telescope(log, s);
    oskar_log_settings_beam_pattern(log, s);

    /* Get the number of requested GPUs.
     * If OpenMP is not available, this can only be 1. */
    num_gpus = s->sim.num_cuda_devices;
#ifdef _OPENMP
    num_threads = num_gpus + 1;
    omp_set_num_threads(num_threads);
#else
    num_gpus = 1;
    oskar_log_warning(log, "OpenMP not available: Ignoring CUDA device list.");
#endif

    /* Find out how many GPUs are in the system. */
    *status = (int) cudaGetDeviceCount(&num_gpus_avail);
    if (*status)
    {
        free_host_data(h, status);
        return;
    }
    if (num_gpus_avail < num_gpus)
    {
        oskar_log_error(log, "More CUDA devices were requested than found.");
        free_host_data(h, status);
        *status = OSKAR_ERR_CUDA_DEVICES;
        return;
    }

    /* Set up host data and check for errors. */
    set_up_host_data(h, log, status);
    if (*status)
    {
        free_host_data(h, status);
        return;
    }

    /* Initialise each of the requested GPUs and set up per-GPU memory. */
    d = (DeviceData*) calloc(num_gpus, sizeof(DeviceData));
    for (i = 0; i < num_gpus; ++i)
    {
        *status = (int) cudaSetDevice(s->sim.cuda_device_ids[i]);
        if (*status)
        {
            free_device_data(num_gpus, s->sim.cuda_device_ids, d, status);
            free_host_data(h, status);
            return;
        }
        set_up_device_data(&d[i], s, h->tel, status);
        cudaDeviceSynchronize();
    }

    /* Work out how many chunks and image slices have to be processed. */
    h->num_chunks = (h->num_pixels + h->max_chunk_size - 1) / h->max_chunk_size;
    num_slices = h->num_channels * h->num_times;

    /* Record memory usage. */
    oskar_log_section(log, 'M', "Initial memory usage");
    for (i = 0; i < num_gpus; ++i)
        oskar_cuda_mem_log(log, 0, s->sim.cuda_device_ids[i]);

    /* Start simulation timer and stop the load timer. */
    oskar_timer_pause(h->tmr_load);
    oskar_log_section(log, 'M', "Starting simulation...");
    oskar_timer_start(h->tmr_sim);

    /*-----------------------------------------------------------------------
     *-- START OF MULTITHREADED SIMULATION CODE -----------------------------
     *-----------------------------------------------------------------------*/
    /* Loop over image slices, running simulation and file writing one
     * slice at a time. Simulation and file output are overlapped by using
     * double buffering, and a dedicated thread is used for file output.
     *
     * Thread 0 is used for file writes.
     * Threads 1 to n (mapped to GPUs) execute the simulation.
     *
     * Note that no write is launched on the first loop counter (as no
     * data are ready yet) and no simulation is performed for the last loop
     * counter (which corresponds to the last slice + 1) as this iteration
     * simply writes the last slice.
     */
#pragma omp parallel shared(next_chunk_index)
    {
        int b, i_active, thread_id = 0, gpu_id = 0;

        /* Get host thread ID, and set CUDA device used by this thread. */
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        gpu_id = thread_id - 1;
#endif
        if (gpu_id >= 0)
            cudaSetDevice(s->sim.cuda_device_ids[gpu_id]);

        /* Loop over simulation slices (+1, for the last write). */
        for (b = 0; b < num_slices + 1; ++b)
        {
            i_active = b % 2; /* Index of the active buffer. */
            if ((thread_id > 0 || num_threads == 1) && b < num_slices)
                sim_slice(gpu_id, &d[gpu_id], h, b, h->slice_cpu[i_active],
                        &next_chunk_index, log, status);
            if (thread_id == 0 && b > 0)
                write_slice(h, b - 1, h->slice_cpu[!i_active], status);

            /* Barrier1: Reset chunk index. */
#pragma omp barrier
            if (thread_id == 0) next_chunk_index = 0;

            /* Barrier2: Check sim and write are done before next slice. */
#pragma omp barrier
            if (thread_id == 0 && b < num_slices && !*status)
                oskar_log_message(log, 'S', 0, "Slice %*i/%i complete. "
                        "Simulation time elapsed: %.3f s",
                        disp_width(num_slices), b+1, num_slices,
                        oskar_timer_elapsed(h->tmr_sim));
        }
    }
    /*-----------------------------------------------------------------------
     *-- END OF MULTITHREADED SIMULATION CODE -------------------------------
     *-----------------------------------------------------------------------*/

    /* Record memory usage. */
    oskar_log_section(log, 'M', "Final memory usage");
    for (i = 0; i < num_gpus; ++i)
        oskar_cuda_mem_log(log, 0, s->sim.cuda_device_ids[i]);

    /* Record time taken. */
    oskar_log_section(log, 'M', "Simulation completed in %.3f sec.",
            oskar_timer_elapsed(h->tmr_sim));

    /* Free device and host memory (and close output files). */
    free_device_data(num_gpus, s->sim.cuda_device_ids, d, status);
    free_host_data(h, status);
}


static void sim_slice(int gpu_id, DeviceData* d, const HostData* h,
        int i_slice, oskar_Mem* slice_cpu, int* next_chunk_index,
        oskar_Log* log, int* status)
{
    int i_time, i_channel;
    double dt_dump, mjd, gast, freq_hz;
    const oskar_Station* st = 0;
    const oskar_Settings* s = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Convert slice index to time and channel index. */
    slice_to_time_and_channel_index(i_slice, h->num_times, &i_channel, &i_time);

    /* Get time and frequency values. */
    s = &h->s;
    dt_dump = s->obs.dt_dump_days;
    mjd = h->start_mjd_utc + dt_dump * (i_time + 0.5);
    gast = oskar_convert_mjd_to_gast_fast(mjd);
    freq_hz = s->obs.start_frequency_hz + i_channel * s->obs.frequency_inc_hz;
    st = oskar_telescope_station_const(d->tel, h->station_id);

    /* Loop until all chunks are done. */
    while (1)
    {
        int i_chunk = 0, chunk_size = 0;
        #pragma omp critical (ChunkIndexUpdate)
        {
            i_chunk = (*next_chunk_index)++;
        }
        if ((i_chunk >= h->num_chunks) || *status) break;

        /* Copy pixel chunk coordinate data to GPU. */
        chunk_size = h->max_chunk_size;
        if ((i_chunk + 1) * h->max_chunk_size > h->num_pixels)
            chunk_size = h->num_pixels - i_chunk * h->max_chunk_size;
        oskar_mem_copy_contents(d->x, h->x, 0,
                i_chunk * h->max_chunk_size, chunk_size, status);
        oskar_mem_copy_contents(d->y, h->y, 0,
                i_chunk * h->max_chunk_size, chunk_size, status);
        oskar_mem_copy_contents(d->z, h->z, 0,
                i_chunk * h->max_chunk_size, chunk_size, status);

        /* Run simulation for this pixel chunk. */
        if (s->beam_pattern.average_cross_power_beam)
        {
            oskar_evaluate_jones_E(d->jones, chunk_size,
                    d->x, d->y, d->z, h->coord_type, h->lon0, h->lat0,
                    d->tel, gast, freq_hz, d->work, i_time, status);
            oskar_evaluate_average_cross_power_beam(chunk_size,
                    h->num_stations, d->jones, d->beam_data, status);
        }
        else
        {
            oskar_evaluate_station_beam(d->beam_data, chunk_size,
                    d->x, d->y, d->z, h->coord_type, h->lon0, h->lat0,
                    st, d->work, i_time, freq_hz, gast, status);
        }

        /* Copy the pixel chunk into host memory buffer for slice. */
        oskar_mem_copy_contents(slice_cpu, d->beam_data,
                i_chunk * h->max_chunk_size, 0, chunk_size, status);

        oskar_log_message(log, 'S', 1, "Channel %*i/%i, "
                "Time %*i/%i, Chunk %*i/%i [GPU %i]",
                disp_width(h->num_channels), i_channel+1, h->num_channels,
                disp_width(h->num_times), i_time+1, h->num_times,
                disp_width(h->num_chunks), i_chunk+1, h->num_chunks, gpu_id);
    }
}


static void write_slice(HostData* h, int i_slice, oskar_Mem* image_slice,
        int* status)
{
    /* Convert slice index to time and channel index. */
    int i, i_time, i_channel, num_pixels;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Split up and accumulate polarisation data. */
    num_pixels = h->num_pixels;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
    {
        if (h->type == OSKAR_SINGLE)
        {
            const float4c* in;
            float2 *a, *b, *c, *d;
            in = oskar_mem_float4c_const(image_slice, status);
            a  = oskar_mem_float2(h->a, status);
            b  = oskar_mem_float2(h->b, status);
            c  = oskar_mem_float2(h->c, status);
            d  = oskar_mem_float2(h->d, status);
            for (i = 0; i < num_pixels; ++i)
            {
                a[i].x += in[i].a.x;
                a[i].y += in[i].a.y;
                b[i].x += in[i].b.x;
                b[i].y += in[i].b.y;
                c[i].x += in[i].c.x;
                c[i].y += in[i].c.y;
                d[i].x += in[i].d.x;
                d[i].y += in[i].d.y;
            }
        }
        else if (h->type == OSKAR_DOUBLE)
        {
            const double4c* in;
            double2 *a, *b, *c, *d;
            in = oskar_mem_double4c_const(image_slice, status);
            a  = oskar_mem_double2(h->a, status);
            b  = oskar_mem_double2(h->b, status);
            c  = oskar_mem_double2(h->c, status);
            d  = oskar_mem_double2(h->d, status);
            for (i = 0; i < num_pixels; ++i)
            {
                a[i].x += in[i].a.x;
                a[i].y += in[i].a.y;
                b[i].x += in[i].b.x;
                b[i].y += in[i].b.y;
                c[i].x += in[i].c.x;
                c[i].y += in[i].c.y;
                d[i].x += in[i].d.x;
                d[i].y += in[i].d.y;
            }
        }
    }
    else
    {
        if (h->type == OSKAR_SINGLE)
        {
            const float2* in;
            float2 *a;
            in = oskar_mem_float2_const(image_slice, status);
            a  = oskar_mem_float2(h->a, status);
            for (i = 0; i < num_pixels; ++i)
            {
                a[i].x += in[i].x;
                a[i].y += in[i].y;
            }
        }
        else if (h->type == OSKAR_DOUBLE)
        {
            const double2* in;
            double2 *a;
            in = oskar_mem_double2_const(image_slice, status);
            a  = oskar_mem_double2(h->a, status);
            for (i = 0; i < num_pixels; ++i)
            {
                a[i].x += in[i].x;
                a[i].y += in[i].y;
            }
        }
    }

    /* Get indices of the input slice. */
    slice_to_time_and_channel_index(i_slice, h->num_times, &i_channel, &i_time);

    /* If averaging, scale by number of times. */
    if (h->time_average && i_time == h->num_times - 1)
    {
        oskar_mem_scale_real(h->a, 1.0 / h->num_times, status);
        oskar_mem_scale_real(h->b, 1.0 / h->num_times, status);
        oskar_mem_scale_real(h->c, 1.0 / h->num_times, status);
        oskar_mem_scale_real(h->d, 1.0 / h->num_times, status);
    }

    if ((h->time_average && i_time == h->num_times - 1) || !h->time_average)
    {
        /* Loop over all output file types. */
        printf("Writing slice %d (channel %d, time %d)\n", i_slice + 1,
                i_channel + 1, i_time + 1);

        /* Clear accumulation buffers after writing. */
        oskar_mem_clear_contents(h->a, status);
        oskar_mem_clear_contents(h->b, status);
        oskar_mem_clear_contents(h->c, status);
        oskar_mem_clear_contents(h->d, status);
    }
}


static void set_up_host_data(HostData* h, oskar_Log* log, int *status)
{
    int beam_type, cmplx, num_pix;
    const oskar_Settings* s = 0;

    /* Get values from settings. */
    s = &h->s;
    h->type = s->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    h->width                    = s->beam_pattern.size[0];
    h->height                   = s->beam_pattern.size[1];
    h->phase_centre_deg[0]      = s->obs.phase_centre_lon_rad[0] * RAD2DEG;
    h->phase_centre_deg[1]      = s->obs.phase_centre_lat_rad[0] * RAD2DEG;
    h->fov_deg[0     ]          = s->beam_pattern.fov_deg[0];
    h->fov_deg[1]               = s->beam_pattern.fov_deg[1];
    h->time_average             = s->beam_pattern.time_average_beam;
    h->num_times                = s->obs.num_time_steps;
    h->num_channels             = s->obs.num_channels;
    h->start_mjd_utc            = s->obs.start_mjd_utc;
    h->start_freq_hz            = s->obs.start_frequency_hz;
    h->delta_t                  = s->obs.dt_dump_days * 86400.0;
    h->delta_f                  = s->obs.frequency_inc_hz;
    h->station_id               = s->beam_pattern.station_id;
    h->max_chunk_size           = s->sim.max_sources_per_chunk;
    h->average_cross_power_beam = s->beam_pattern.average_cross_power_beam;

    /* Set up telescope model. */
    h->tel = oskar_set_up_telescope(s, log, status);

    /* Get beam data type. */
    cmplx     = h->type | OSKAR_COMPLEX;
    beam_type = cmplx;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
        beam_type |= OSKAR_MATRIX;

    /* Check the station ID is valid. */
    h->num_stations = oskar_telescope_num_stations(h->tel);
    if (h->station_id < 0 || h->station_id >= h->num_stations)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Set up pixel positions. */
    h->x = oskar_mem_create(h->type, OSKAR_CPU, 0, status);
    h->y = oskar_mem_create(h->type, OSKAR_CPU, 0, status);
    h->z = oskar_mem_create(h->type, OSKAR_CPU, 0, status);
    if (s->beam_pattern.average_cross_power_beam)
    {
        h->num_pixels = oskar_beam_pattern_generate_coordinates(
                &h->coord_type, h->x, h->y, h->z, &h->lon0, &h->lat0,
                OSKAR_SPHERICAL_TYPE_EQUATORIAL,
                oskar_telescope_phase_centre_ra_rad(h->tel),
                oskar_telescope_phase_centre_dec_rad(h->tel),
                &s->beam_pattern, status);
    }
    else
    {
        const oskar_Station* st =
                oskar_telescope_station_const(h->tel, h->station_id);
        h->num_pixels = oskar_beam_pattern_generate_coordinates(
                &h->coord_type, h->x, h->y, h->z, &h->lon0, &h->lat0,
                oskar_station_beam_coord_type(st),
                oskar_station_beam_lon_rad(st),
                oskar_station_beam_lat_rad(st),
                &s->beam_pattern, status);
    }

    /* Create host memory buffers to hold a slice. */
    num_pix = h->num_pixels;
    h->slice_cpu[0] = oskar_mem_create(beam_type, OSKAR_CPU, num_pix, status);
    h->slice_cpu[1] = oskar_mem_create(beam_type, OSKAR_CPU, num_pix, status);

    /* Create scratch arrays for reorder & accumulation of output data. */
    h->a = oskar_mem_create(cmplx, OSKAR_CPU, num_pix, status);
    h->b = oskar_mem_create(cmplx, OSKAR_CPU, num_pix, status);
    h->c = oskar_mem_create(cmplx, OSKAR_CPU, num_pix, status);
    h->d = oskar_mem_create(cmplx, OSKAR_CPU, num_pix, status);
    oskar_mem_clear_contents(h->a, status);
    oskar_mem_clear_contents(h->b, status);
    oskar_mem_clear_contents(h->c, status);
    oskar_mem_clear_contents(h->d, status);

    /* Create a file for each requested data product. */
    if (s->beam_pattern.average_cross_power_beam)
    {

    }
    if (s->beam_pattern.fits_image_total_intensity)
    {
        new_fits(h, TOTAL_INTENSITY,
                s->beam_pattern.fits_image_total_intensity, status);
    }
    if (!h->average_cross_power_beam)
    {
        if (s->beam_pattern.fits_image_voltage)
            new_fits(h, VOLTAGE, s->beam_pattern.fits_image_voltage, status);
        if (s->beam_pattern.fits_image_phase)
            new_fits(h, PHASE, s->beam_pattern.fits_image_phase, status);
    }

    /* Check that an output file has been specified. */
    if (h->num_ascii_handles == 0 && h->num_fits_handles == 0)
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        oskar_log_error(log, "No output file(s) specified.");
    }
}


static void slice_to_time_and_channel_index(int slice, int num_times,
        int* i_channel, int* i_time)
{
    /* Time is the fastest varying dimension. */
    *i_channel = slice / num_times;
    *i_time = slice % num_times;
}


static double fov_to_cellsize(double fov_deg, int num_pixels)
{
    double max, inc;
    max = sin(fov_deg * M_PI / 360.0); /* Divide by 2. */
    inc = max / (0.5 * num_pixels);
    return asin(inc) * 180.0 / M_PI;
}


static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, int num_channels,
        double centre_deg[2], double fov_deg[2], double start_time_mjd,
        double delta_time_sec, double start_freq_hz, double delta_freq_hz,
        int* status)
{
    int imagetype;
    long naxes[4];
    double delta;
    fitsfile* f = 0;

    /* Create a new FITS file and write the image headers. */
    if (oskar_file_exists(filename)) remove(filename);
    imagetype = (precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG);
    naxes[0]  = width;
    naxes[1]  = height;
    naxes[2]  = num_times;
    naxes[3]  = num_channels;
    fits_create_file(&f, filename, status);
    fits_create_img(f, imagetype, 4, naxes, status);
    fits_write_date(f, status);
    fits_write_key_str(f, "TELESCOP",
            "OSKAR " OSKAR_VERSION_STR, NULL, status);
    fits_write_history(f, "Created using OSKAR " OSKAR_VERSION_STR, status);

    /* Write axis headers. */
    delta = fov_to_cellsize(fov_deg[0], width);
    oskar_fits_write_axis_header(f, 1, "RA---SIN", "Right Ascension",
            centre_deg[0], -delta, (width + 1) / 2.0, 0.0, status);
    delta = fov_to_cellsize(fov_deg[1], height);
    oskar_fits_write_axis_header(f, 2, "DEC--SIN", "Declination",
            centre_deg[1], delta, (height + 1) / 2.0, 0.0, status);
    oskar_fits_write_axis_header(f, 3, "UTC", "Time",
            start_time_mjd, delta_time_sec, 1.0, 0.0, status);
    oskar_fits_write_axis_header(f, 4, "FREQ", "Frequency",
            start_freq_hz, delta_freq_hz, 1.0, 0.0, status);

    /* Write other headers. */
    fits_write_key_str(f, "TIMESYS", "UTC", NULL, status);
    fits_write_key_str(f, "TIMEUNIT", "s", "Time axis units", status);
    fits_write_key_dbl(f, "MJD-OBS", start_time_mjd, 10, "Start time", status);
    fits_write_key_dbl(f, "OBSRA", centre_deg[0], 10, "RA", status);
    fits_write_key_dbl(f, "OBSDEC", centre_deg[1], 10, "DEC", status);

    return f;
}


static void new_fits(HostData* h, int file_type, const char* name, int* status)
{
    int i;
    i = h->num_fits_handles++;
    h->fits_data_type = (int*) realloc(h->fits_data_type,
            h->num_fits_handles * sizeof(int));
    h->handle_fits = (fitsfile**) realloc(h->handle_fits,
            h->num_fits_handles * sizeof(fitsfile*));
    h->fits_data_type[i] = file_type;
    h->handle_fits[i] = create_fits_file(name, h->type, h->width, h->height,
            (h->time_average ? 1 : h->num_times), h->num_channels,
            h->phase_centre_deg, h->fov_deg, h->start_mjd_utc, h->delta_t,
            h->start_freq_hz, h->delta_f, status);
}


static void new_ascii(HostData* h, int file_type, const char* name)
{
    int i;
    i = h->num_ascii_handles++;
    h->ascii_data_type = (int*) realloc(h->ascii_data_type,
            h->num_ascii_handles * sizeof(int));
    h->handle_ascii = (FILE**) realloc(h->handle_ascii,
            h->num_ascii_handles * sizeof(FILE*));
    h->ascii_data_type[i] = file_type;
    h->handle_ascii[i] = fopen(name, "w");
}


static void set_up_device_data(DeviceData* d, const oskar_Settings* s,
        const oskar_Telescope* tel, int* status)
{
    int max_chunk_size, prec, beam_type;

    /* Get local variables from settings. */
    max_chunk_size = s->sim.max_sources_per_chunk;
    prec           = s->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    beam_type      = prec | OSKAR_COMPLEX;
    if (oskar_telescope_pol_mode(tel) == OSKAR_POL_MODE_FULL)
        beam_type |= OSKAR_MATRIX;

    /* Device memory. */
    d->tel  = oskar_telescope_create_copy(tel, OSKAR_GPU, status);
    d->work = oskar_station_work_create(prec, OSKAR_GPU, status);
    d->x    = oskar_mem_create(prec, OSKAR_GPU, 1 + max_chunk_size, status);
    d->y    = oskar_mem_create(prec, OSKAR_GPU, 1 + max_chunk_size, status);
    d->z    = oskar_mem_create(prec, OSKAR_GPU, 1 + max_chunk_size, status);
    d->beam_data = oskar_mem_create(beam_type, OSKAR_GPU, max_chunk_size,
            status);
    d->jones = 0;
    if (s->beam_pattern.average_cross_power_beam)
        d->jones = oskar_jones_create(beam_type, OSKAR_GPU,
                oskar_telescope_num_stations(tel), max_chunk_size, status);
}


static void free_device_data(int num_gpus, int* cuda_device_ids,
        DeviceData* d, int* status)
{
    int i;
    if (!d) return;
    for (i = 0; i < num_gpus; ++i)
    {
        DeviceData* dd = &d[i];
        if (!dd) continue;
        cudaSetDevice(cuda_device_ids[i]);
        oskar_mem_free(dd->x, status);
        oskar_mem_free(dd->y, status);
        oskar_mem_free(dd->z, status);
        oskar_mem_free(dd->beam_data, status);
        oskar_telescope_free(dd->tel, status);
        oskar_station_work_free(dd->work, status);
        oskar_jones_free(dd->jones, status);
        cudaDeviceReset();
    }
    free(d);
}


static void free_host_data(HostData* h, int* status)
{
    int i;
    for (i = 0; i < h->num_ascii_handles; ++i)
        fclose(h->handle_ascii[i]);
    for (i = 0; i < h->num_fits_handles; ++i)
        ffclos(h->handle_fits[i], status);
    oskar_mem_free(h->a, status);
    oskar_mem_free(h->b, status);
    oskar_mem_free(h->c, status);
    oskar_mem_free(h->d, status);
    oskar_telescope_free(h->tel, status);
    oskar_mem_free(h->x, status);
    oskar_mem_free(h->y, status);
    oskar_mem_free(h->z, status);
    oskar_mem_free(h->slice_cpu[0], status);
    oskar_mem_free(h->slice_cpu[1], status);
    oskar_timer_free(h->tmr_sim);
    oskar_timer_free(h->tmr_load);
    oskar_settings_free(&h->s);
    free(h->handle_ascii);
    free(h->handle_fits);
    free(h->ascii_data_type);
    free(h->fits_data_type);
    free(h);
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
