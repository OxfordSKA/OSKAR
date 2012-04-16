/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_UTILITY_H_
#define OSKAR_UTILITY_H_

/**
 * @file oskar_utility.h
 */

#include "utility/oskar_binary_file_read.h"
#include "utility/oskar_binary_file_write.h"
#include "utility/oskar_binary_header_version.h"
#include "utility/oskar_binary_stream_read_header.h"
#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_stream_write_cuda_info.h"
#include "utility/oskar_binary_stream_write_header.h"
#include "utility/oskar_binary_stream_write_metadata.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_binary_tag_index_query.h"
#include "utility/oskar_BinaryHeader.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_blank_parentheses.h"
#include "utility/oskar_cuda_device_info_scan.h"
#include "utility/oskar_cuda_info_create.h"
#include "utility/oskar_cuda_info_free.h"
#include "utility/oskar_cuda_info_print.h"
#include "utility/oskar_CudaDeviceInfo.h"
#include "utility/oskar_CudaInfo.h"
#include "utility/oskar_device_curand_state_init.h"
#include "utility/oskar_Device_curand_state.h"
#include "utility/oskar_endian.h"
#include "utility/oskar_exit.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_get_data_type_string.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_mem_all_headers.h"
#include "utility/oskar_settings_free.h"
#include "utility/oskar_settings_print.h"
#include "utility/oskar_Settings.h"
#include "utility/oskar_SettingsSimulator.h"
#include "utility/oskar_string_to_array.h"
#include "utility/oskar_system_clock_time.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_work_free.h"
#include "utility/oskar_work_init.h"
#include "utility/oskar_Work.h"

#endif /* OSKAR_UTILITY_H_ */
