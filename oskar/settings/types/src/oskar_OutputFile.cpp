/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_OutputFile.h"

namespace oskar {

// Don't put these in the header, as doing so can cause
// undefined vtable entries at link time.

OutputFile::OutputFile() : InputFile() {}

// LCOV_EXCL_START
OutputFile::~OutputFile() {}
// LCOV_EXCL_STOP

} // namespace oskar
