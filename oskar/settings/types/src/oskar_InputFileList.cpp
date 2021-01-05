/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_InputFileList.h"

namespace oskar {

// Don't put these in the header, as doing so can cause
// undefined vtable entries at link time.

InputFileList::InputFileList() : StringList() {}

// LCOV_EXCL_START
InputFileList::~InputFileList() {}
// LCOV_EXCL_STOP

} // namespace oskar
