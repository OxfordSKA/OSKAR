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

#ifndef OSKAR_SETTINGS_IMAGE_H_
#define OSKAR_SETTINGS_IMAGE_H_

#include <QtCore/QString>
#include <QtCore/QSettings>

/// Container class for image settings group.
class oskar_SettingsImage
{
    public:
        void load(const QSettings& settings);

    public:
        double fov_deg() const { return _fov_deg; }
        void set_fov_deg(const double value) { _fov_deg = value; }

        unsigned size() const { return _size; }
        void set_size(const unsigned value) { _size = value; }

        bool make_snapshots() const { return _make_snapshots; }
        void set_make_snapshots(const bool value) { _make_snapshots = value; }

        unsigned dumps_per_snapshot() const { return _dumps_per_snapshot; }
        void set_dumps_per_snapshot(const unsigned value) { _dumps_per_snapshot = value; }

        bool scan_frequencies() const { return _scan_frequencies; }
        void set_scan_frequencies(const bool value) { _scan_frequencies = value; }

        QString filename() const { return _filename; }
        void set_filename(const QString& value)  { _filename = value; }

    private:
        double   _fov_deg;
        unsigned _size;
        bool     _make_snapshots;
        unsigned _dumps_per_snapshot;
        bool     _scan_frequencies;
        QString  _filename;
};

#endif // OSKAR_SETTINGS_IMAGE_H_
