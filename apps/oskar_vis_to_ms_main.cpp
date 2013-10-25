/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <apps/lib/oskar_vis_write_ms.h>
#include <apps/lib/oskar_OptionParser.h>
#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_vis.h>

#include <cstdio>

int main(int argc, char** argv)
{
    // Check if built with Measurement Set support.
#ifndef OSKAR_NO_MS
    int error = 0;

    oskar_OptionParser opt("oskar_vis_to_ms");
    opt.addRequired("OSKAR vis file");
    opt.addRequired("MS name");
    if (!opt.check_options(argc, argv))
        return OSKAR_FAIL;

    const char* oskar_vis = opt.getArg(0);
    const char* ms_name = opt.getArg(1);

    // Load the visibility file and write it out as a Measurement Set.
    oskar_Vis* vis = oskar_vis_read(oskar_vis, &error);
    oskar_vis_write_ms(vis, ms_name, 1, &error);
    if (error)
        oskar_log_error(0, oskar_get_error_string(error));
    oskar_vis_free(vis, &error);
    return error;

#else
    // No Measurement Set support.
    oskar_log_error(0, "OSKAR was not compiled with Measurement Set support.");
    return OSKAR_FAIL;
#endif
}
