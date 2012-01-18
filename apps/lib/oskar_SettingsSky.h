/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_SETTINGS_SKY_H_
#define OSKAR_SETTINGS_SKY_H_

#include <QtCore/QSettings>

/// Container class for sky settings group.
class oskar_SettingsSky
{
    public:
        void load(const QSettings& settings);

        void print_summary() const;

        QString sky_file() const { return sky_file_; }
        void set_sky_file(const QString& value) { sky_file_ = value; }

        QString output_sky_file() const { return output_sky_file_; }

        // Generator accessor methods
        QString generator() const { return generator_; }
        void set_generator(const QString& value) { generator_ = value; }
        int healpix_nside() const { return healpix_nside_; }
        void set_healpix_nside(const int value) { healpix_nside_ = value; }

        int random_num_sources() const { return random_num_sources_; }
        double random_flux_density_min() const { return random_flux_density_min_; }
        double random_flux_density_max() const { return random_flux_density_max_; }
        double random_threshold() const { return random_threshold_; }
        double random_power() const { return random_power1_; }
        double random_power1() const { return random_power1_; }
        double random_power2() const { return random_power2_; }
        unsigned random_seed() const { return random_seed_; }

        QString noise_model() const { return noise_model_; }
        double noise_spectral_index() const { return noise_spectral_index_; }
        double noise_seed() const { return noise_seed_; }

    private:
        // Sky model file name to load.
        QString sky_file_;

        // Sky model file to write to.
        QString output_sky_file_;

        // Generator settings.
        QString  generator_;                 ///< Generator type.
        int      healpix_nside_;
        int      random_num_sources_;
        double   random_flux_density_min_;
        double   random_flux_density_max_;
        double   random_threshold_;
        double   random_power1_;
        double   random_power2_;
        unsigned random_seed_;

        // Sky noise (Gaussian random component added to visibilities).
        QString noise_model_;         ///< Noise model type.
        double noise_spectral_index_; ///< Frequency spectral index
        unsigned noise_seed_;
};

#endif // OSKAR_SETTINGS_SKY_H_
