/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_IMAGER_ACCESSORS_H_
#define OSKAR_IMAGER_ACCESSORS_H_

/**
 * @file oskar_imager_accessors.h
 */

#include <oskar_global.h>
#include <log/oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Returns the imager algorithm.
 *
 * @details
 * Returns a string describing the algorithm used by the imager.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
const char* oskar_imager_algorithm(const oskar_Imager* h);

/**
 * @brief
 * Returns the image cell size.
 *
 * @details
 * Returns the cell (pixel) size of the output images, in arcsec.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_cellsize(const oskar_Imager* h);

/**
 * @brief
 * Returns the flag specifying whether to image each channel separately.
 *
 * @details
 * Returns the flag specifying whether to image each channel separately.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_channel_snapshots(const oskar_Imager* h);

/**
 * @brief
 * Returns the flag specifying whether the imager is in coordinate-only mode.
 *
 * @details
 * Returns the flag specifying whether the imager is in coordinate-only mode.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_coords_only(const oskar_Imager* h);

/**
 * @brief
 * Returns the flag specifying whether to use the GPU for FFTs.
 *
 * @details
 * Returns the flag specifying whether to use the GPU for FFTs.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_fft_on_gpu(const oskar_Imager* h);

/**
 * @brief
 * Returns the image field of view.
 *
 * @details
 * Returns the field of view of the output images, in degrees.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_fov(const oskar_Imager* h);

/**
 * @brief
 * Returns the maximum frequency of visibility data to include in the image.
 *
 * @details
 * Returns the maximum frequency of visibility data to include in the image
 * or image cube. A value less than or equal to zero means no maximum.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_freq_max_hz(const oskar_Imager* h);

/**
 * @brief
 * Returns the minimum frequency of visibility data to include in the image.
 *
 * @details
 * Returns the minimum frequency of visibility data to include in the image
 * or image cube.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_freq_min_hz(const oskar_Imager* h);

/**
 * @brief
 * Returns the flag specifying whether to use the GPU to generate W-kernels.
 *
 * @details
 * Returns the flag specifying whether to use the GPU to generate W-kernels.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_generate_w_kernels_on_gpu(const oskar_Imager* h);

/**
 * @brief
 * Returns the flag specifying whether to use the GPU for gridding.
 *
 * @details
 * Returns the flag specifying whether to use the GPU for gridding.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_grid_on_gpu(const oskar_Imager* h);

/**
 * @brief
 * Returns the image side length.
 *
 * @details
 * Returns the image side length in pixels.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_image_size(const oskar_Imager* h);

/**
 * @brief
 * Returns the image (polarisation) type
 *
 * @details
 * Returns a string describing the image (polarisation) type made by the imager.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
const char* oskar_imager_image_type(const oskar_Imager* h);

/**
 * @brief
 * Returns the list of input files or Measurement Sets.
 *
 * @details
 * Returns the list of input files or Measurement Sets.
 * These are used when calling oskar_imager_run().
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
char* const* oskar_imager_input_files(const oskar_Imager* h);

/**
 * @brief
 * Returns the logger.
 *
 * @details
 * Returns the logger.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
oskar_Log* oskar_imager_log(oskar_Imager* h);

/**
 * @brief
 * Returns the Measurement Set column to use.
 *
 * @details
 * Returns the Measurement Set column to use.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
const char* oskar_imager_ms_column(const oskar_Imager* h);

/**
 * @brief
 * Returns the number of image planes in use.
 *
 * @details
 * Returns the number of image planes in use.
 */
OSKAR_EXPORT
int oskar_imager_num_image_planes(const oskar_Imager* h);

/**
 * @brief
 * Returns the number of input files or Measurement Sets.
 *
 * @details
 * Returns the number of input files or Measurement Sets.
 * These are used when calling oskar_imager_run().
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_num_input_files(const oskar_Imager* h);

/**
 * @brief
 * Returns the number of W-planes in use.
 *
 * @details
 * Returns the number of W-planes in use.
 */
OSKAR_EXPORT
int oskar_imager_num_w_planes(const oskar_Imager* h);

/**
 * @brief
 * Returns the output filename root.
 *
 * @details
 * Returns the output filename root.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
const char* oskar_imager_output_root(const oskar_Imager* h);

/**
 * @brief
 * Returns the grid size required by the algorithm.
 *
 * @details
 * Returns the grid size required by the algorithm.
 * This will be different to the image size when using W-projection.
 */
OSKAR_EXPORT
int oskar_imager_plane_size(oskar_Imager* h);

/**
 * @brief
 * Returns the enumerated grid data type.
 *
 * @details
 * Returns the enumerated grid data type.
 */
OSKAR_EXPORT
int oskar_imager_plane_type(const oskar_Imager* h);

/**
 * @brief
 * Returns the enumerated imager precision.
 *
 * @details
 * Returns the enumerated imager precision (OSKAR_SINGLE or OSKAR_DOUBLE).
 */
OSKAR_EXPORT
int oskar_imager_precision(const oskar_Imager* h);

/**
 * @brief
 * Returns the option to scale image normalisation by the number of input files.
 *
 * @details
 * Returns the option to scale image normalisation by the number of input files.
 */
OSKAR_EXPORT
int oskar_imager_scale_norm_with_num_input_files(const oskar_Imager* h);

/**
 * @brief
 * Sets the algorithm used by the imager.
 *
 * @details
 * Sets the algorithm used by the imager.
 *
 * The \p type string can be:
 * - "FFT" to use standard gridding followed by a FFT.
 * - "W-projection" to use W-projection gridding followed by a FFT.
 * - "DFT 2D" to use a 2D Direct Fourier Transform, without gridding.
 * - "DFT 3D" to use a 3D Direct Fourier Transform, without gridding.
 *
 * This function also sets the default gridding parameters.
 * Call oskar_imager_set_grid_kernel() or oskar_imager_set_oversample()
 * afterwards to override these if required.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     type       The algorithm to use (see description).
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_algorithm(oskar_Imager* h, const char* type,
        int* status);

/**
 * @brief
 * Sets the image cell size.
 *
 * @details
 * Sets the cell (pixel) size of the output images.
 *
 * @param[in,out] h                  Handle to imager.
 * @param[in]     cellsize_arcsec    Cell size, in arcsec.
 */
OSKAR_EXPORT
void oskar_imager_set_cellsize(oskar_Imager* h, double cellsize_arcsec);

/**
 * @brief
 * Sets the flag specifying whether to image each channel separately.
 *
 * @details
 * Sets the flag specifying whether to image each channel separately.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     value      If true, image each channel separately;
 *                           if false, use frequency synthesis.
 */
OSKAR_EXPORT
void oskar_imager_set_channel_snapshots(oskar_Imager* h, int value);

/**
 * @brief
 * Sets the imager to ignore visibility data and only update weights grids.
 *
 * @details
 * Use this method with uniform weighting or W-projection.
 * The grids of weights can only be used once they are fully populated,
 * so this method puts the imager into a mode where it only updates its
 * internal weights grids when calling oskar_imager_update().
 *
 * This should only be called after setting all imager options.
 *
 * Turn this mode off when processing visibilities.
 *
 * The calling sequence should be:
 *
 *     // Update weights grid with all coordinates.
 *     oskar_imager_set_coords_only(true)
 *     (repeat) oskar_imager_update()
 *
 *     // Process actual visibility data.
 *     oskar_imager_set_coords_only(false)
 *     (repeat) oskar_imager_update()
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     flag       If set, ignore visibilities and only update
 *                           weights grids.
 */
OSKAR_EXPORT
void oskar_imager_set_coords_only(oskar_Imager* h, int flag);

/**
 * @brief
 * Clears any direction override.
 *
 * @details
 * Clears any direction override set using oskar_imager_set_direction().
 *
 * @param[in,out] h          Handle to imager.
 */
OSKAR_EXPORT
void oskar_imager_set_default_direction(oskar_Imager* h);

/**
 * @brief
 * Sets the image centre to be different to the observation phase centre.
 *
 * @details
 * Sets the image centre to be different to the observation phase centre.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     ra_deg     The new image Right Ascension, in degrees.
 * @param[in]     dec_deg    The new image Declination, in degrees.
 */
OSKAR_EXPORT
void oskar_imager_set_direction(oskar_Imager* h, double ra_deg, double dec_deg);

/**
 * @brief
 * Sets whether to use the GPU for FFTs.
 *
 * @details
 * Sets whether to use the GPU for FFTs.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     value      If true, use the GPU; if false, use the CPU.
 */
OSKAR_EXPORT
void oskar_imager_set_fft_on_gpu(oskar_Imager* h, int value);

/**
 * @brief
 * Sets the image field of view.
 *
 * @details
 * Sets the field of view of the output images.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     fov_deg    Field of view, in degrees.
 */
OSKAR_EXPORT
void oskar_imager_set_fov(oskar_Imager* h, double fov_deg);

/**
 * @brief
 * Sets the maximum frequency of visibility data to include in the image.
 *
 * @details
 * Sets the maximum frequency of visibility data to include in the image
 * or image cube. A value less than or equal to zero means no maximum.
 *
 * @param[in] h           Handle to imager.
 * @param[in] freq_max_hz The maximum frequency of visibility data, in Hz.
 */
OSKAR_EXPORT
void oskar_imager_set_freq_max_hz(oskar_Imager* h, double freq_max_hz);

/**
 * @brief
 * Sets the minimum frequency of visibility data to include in the image.
 *
 * @details
 * Sets the minimum frequency of visibility data to include in the image
 * or image cube.
 *
 * @param[in] h           Handle to imager.
 * @param[in] freq_min_hz The minimum frequency of visibility data, in Hz.
 */
OSKAR_EXPORT
void oskar_imager_set_freq_min_hz(oskar_Imager* h, double freq_min_hz);

/**
 * @brief
 * Sets whether to use the GPU to generate W-kernels.
 *
 * @details
 * Sets whether to use the GPU to generate W-kernels.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     value      If true, use the GPU to generate the W-kernels.
 */
OSKAR_EXPORT
void oskar_imager_set_generate_w_kernels_on_gpu(oskar_Imager* h, int value);

/**
 * @brief
 * Sets which GPUs will be used by the imager.
 *
 * @details
 * Sets which GPUs will be used by the imager.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     num        The number of GPUs to use.
 * @param[in]     ids        An array containing the GPU IDs to use.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_gpus(oskar_Imager* h, int num, const int* ids,
        int* status);

/**
 * @brief
 * Sets the convolution kernel used for gridding visibilities.
 *
 * @details
 * Sets the convolution kernel used for gridding visibilities.
 *
 * The \p type string can be:
 * - "Spheroidal" to use the spheroidal kernel from CASA.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     type       Type of kernel to use.
 * @param[in]     support    Support size of kernel.
 * @param[in]     oversample Oversampling factor used for look-up table.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_grid_kernel(oskar_Imager* h, const char* type,
        int support, int oversample, int* status);

/**
 * @brief
 * Sets whether to use the GPU for gridding.
 *
 * @details
 * Sets whether to use the GPU for gridding.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     value      If true, use the GPU; if false, use the CPU.
 */
OSKAR_EXPORT
void oskar_imager_set_grid_on_gpu(oskar_Imager* h, int value);

/**
 * @brief
 * Sets image side length.
 *
 * @details
 * Sets the image side length in pixels.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     size       Image side length in pixels.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_image_size(oskar_Imager* h, int size, int* status);

/**
 * @brief
 * Sets the image (polarisation) type
 *
 * @details
 * Sets the polarisation made by the imager.
 *
 * The \p type string can be:
 * - "STOKES" for all four Stokes parameters.
 * - "I" for Stokes I only.
 * - "Q" for Stokes Q only.
 * - "U" for Stokes U only.
 * - "V" for Stokes V only.
 * - "LINEAR" for all four linear polarisations.
 * - "XX" for XX only.
 * - "XY" for XY only.
 * - "YX" for YX only.
 * - "YY" for YY only.
 * - "PSF" for the point spread function.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     type       Image type; see description.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_image_type(oskar_Imager* h, const char* type,
        int* status);

/**
 * @brief
 * Sets the input files or Measurement Sets.
 *
 * @details
 * Sets the input files or Measurement Sets
 * This is used when calling oskar_imager_run().
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     num_files  Number of input files in list.
 * @param[in]     filenames  Input paths.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_input_files(oskar_Imager* h, int num_files,
        const char* const* filenames, int* status);

/**
 * @brief
 * Sets the data column to use from a Measurement Set.
 *
 * @details
 * Sets the data column to use from a Measurement Set.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     column     Name of the column to use.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_ms_column(oskar_Imager* h, const char* column,
        int* status);

/**
 * @brief
 * Sets the number of compute devices used by the imager.
 *
 * @details
 * Sets the number of compute devices used by the imager, either CPU cores
 * or GPU cards.
 * Currently this is only used by the DFT imager.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     value      Number of compute devices to use.
 */
OSKAR_EXPORT
void oskar_imager_set_num_devices(oskar_Imager* h, int value);

/**
 * @brief
 * Sets the root path of output images.
 *
 * @details
 * Sets the root path of output images.
 * FITS images files will be created with appropriate extensions.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     filename   Root path.
 */
OSKAR_EXPORT
void oskar_imager_set_output_root(oskar_Imager* h, const char* filename);

/**
 * @brief
 * Sets kernel oversample factor.
 *
 * @details
 * Sets the kernel oversample factor.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     value      Kernel oversample factor.
 */
OSKAR_EXPORT
void oskar_imager_set_oversample(oskar_Imager* h, int value);

/**
 * @brief
 * Sets the option to scale image normalisation with number of input files.
 *
 * @details
 * Sets the option to scale image normalisation with number of input files.
 * Set this to true if the different files represent multiple
 * sky model components observed with the same telescope configuration
 * and observation parameters.
 * Set this to false if the different files represent multiple
 * observations of the same sky with different telescope configurations
 * or observation parameters.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     value      Option value (true or false).
 */
OSKAR_EXPORT
void oskar_imager_set_scale_norm_with_num_input_files(oskar_Imager* h,
        int value);

/**
 * @brief
 * Sets image side length.
 *
 * @details
 * Sets the image side length in pixels.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     size       Image side length in pixels.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_size(oskar_Imager* h, int size, int* status);

/**
 * @brief
 * Sets the maximum timestamp of visibility data to include in the image.
 *
 * @details
 * Sets the maximum timestamp of visibility data to include in the image
 * or image cube. A value less than or equal to zero means no maximum.
 *
 * @param[in] h                Handle to imager.
 * @param[in] time_max_mjd_utc The maximum timestamp of visibility data,
 *                             as MJD(UTC).
 */
OSKAR_EXPORT
void oskar_imager_set_time_max_utc(oskar_Imager* h, double time_max_mjd_utc);

/**
 * @brief
 * Sets the minimum timestamp of visibility data to include in the image.
 *
 * @details
 * Sets the minimum timestamp of visibility data to include in the image
 * or image cube.
 *
 * @param[in] h                Handle to imager.
 * @param[in] time_min_mjd_utc The minimum timestamp of visibility data,
 *                             as MJD(UTC).
 */
OSKAR_EXPORT
void oskar_imager_set_time_min_utc(oskar_Imager* h, double time_min_mjd_utc);

/**
 * @brief
 * Sets the maximum UV baseline length to image.
 *
 * @details
 * Sets the maximum UV baseline length to image, in wavelengths.
 * A value less than 0 means no maximum.
 *
 * @param[in,out] h               Handle to imager.
 * @param[in]     max_wavelength  Maximum UV distance, in wavelengths.
 */
OSKAR_EXPORT
void oskar_imager_set_uv_filter_max(oskar_Imager* h, double max_wavelength);

/**
 * @brief
 * Sets the minimum UV baseline length to image.
 *
 * @details
 * Sets the minimum UV baseline length to image, in wavelengths.
 *
 * @param[in,out] h               Handle to imager.
 * @param[in]     min_wavelength  Minimum UV distance, in wavelengths.
 */
OSKAR_EXPORT
void oskar_imager_set_uv_filter_min(oskar_Imager* h, double min_wavelength);

/**
 * @brief
 * Sets the UV taper to apply to the weights.
 *
 * @details
 * Sets the UV taper to apply to the weights, in wavelengths.
 *
 * Re-weighting is done using a Gaussian in the same way
 * as the AIPS IMAGR task:
 *
 * weight(i) = weight(i) * exp(Cu U(i)**2 + Cv V(i)**2)
 *
 * where
 *
 *        Cu = log(0.3) / U_TAPER**2
 *        Cv = log(0.3) / V_TAPER**2
 *
 * and U(i) and V(i) are the coordinates of each data sample in wavelengths.
 *
 * @param[in,out] h                   Handle to imager.
 * @param[in]     taper_u_wavelength  Taper in U, in wavelengths.
 * @param[in]     taper_v_wavelength  Taper in V, in wavelengths.
 */
OSKAR_EXPORT
void oskar_imager_set_uv_taper(oskar_Imager* h,
        double taper_u_wavelength, double taper_v_wavelength);

/**
 * @brief
 * Sets the visibility start frequency.
 *
 * @details
 * Sets the frequency of channel index 0, and the frequency increment.
 * This is required even if not applying phase rotation or channel selection,
 * to ensure the FITS headers are written correctly.
 *
 * @param[in,out] h       Handle to imager.
 * @param[in]     ref_hz  Frequency of index 0, in Hz.
 * @param[in]     inc_hz  Frequency increment, in Hz.
 * @param[in]     num     Number of channels in visibility data.
 */
OSKAR_EXPORT
void oskar_imager_set_vis_frequency(oskar_Imager* h,
        double ref_hz, double inc_hz, int num);

/**
 * @brief
 * Sets the coordinates of the visibility phase centre.
 *
 * @details
 * Sets the coordinates of the visibility phase centre.
 * This is required even if not applying phase rotation, to ensure the
 * FITS headers are written correctly.
 *
 * @param[in,out] h          Handle to imager.
 * @param[in]     ra_deg     Right Ascension of phase centre, in degrees.
 * @param[in]     dec_deg    Declination of phase centre, in degrees.
 */
OSKAR_EXPORT
void oskar_imager_set_vis_phase_centre(oskar_Imager* h,
        double ra_deg, double dec_deg);

/**
 * @brief
 * Sets the number of W planes to use.
 *
 * @details
 * Sets the number of W planes, used only for W-projection.
 * A value of 0 or less means 'automatic'.
 *
 * @param[in,out] h            Handle to imager.
 * @param[in] value            Number of W planes to use.
 */
OSKAR_EXPORT
void oskar_imager_set_num_w_planes(oskar_Imager* h, int value);

/**
 * @brief
 * Sets the visibility weighting scheme to use.
 *
 * @details
 * Sets the visibility weighting scheme to use,
 * either "Natural", "Radial" or "Uniform".
 *
 * @param[in,out] h            Handle to imager.
 * @param[in] type             Visibility weighting type string, as above.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_imager_set_weighting(oskar_Imager* h, const char* type, int* status);

/**
 * @brief
 * Returns the image side length.
 *
 * @details
 * Returns the image side length in pixels.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
int oskar_imager_size(const oskar_Imager* h);

/**
 * @brief
 * Returns the maximum timestamp of visibility data to include in the image.
 *
 * @details
 * Returns the maximum timestamp of visibility data to include in the image
 * or image cube. A value less than or equal to zero means no maximum.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_time_max_utc(const oskar_Imager* h);

/**
 * @brief
 * Returns the minimum timestamp of visibility data to include in the image.
 *
 * @details
 * Returns the minimum timestamp of visibility data to include in the image
 * or image cube.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_time_min_utc(const oskar_Imager* h);

/**
 * @brief
 * Returns the maximum UV baseline length to image.
 *
 * @details
 * Returns the maximum UV baseline length to image, in wavelengths.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_uv_filter_max(const oskar_Imager* h);

/**
 * @brief
 * Returns the minimum UV baseline length to image.
 *
 * @details
 * Returns the minimum UV baseline length to image, in wavelengths.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
double oskar_imager_uv_filter_min(const oskar_Imager* h);

/**
 * @brief
 * Returns the visibility weighting scheme.
 *
 * @details
 * Returns a string describing the visibility weighting scheme.
 *
 * @param[in] h  Handle to imager.
 */
OSKAR_EXPORT
const char* oskar_imager_weighting(const oskar_Imager* h);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGER_ACCESSORS_H_ */
