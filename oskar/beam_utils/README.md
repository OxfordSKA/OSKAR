This folder contains utility functions which are built into a stand-alone
library used by EveryBeam - check to see if they are still being used there
before modifying.

The library can be compiled after cloning the repository, and then running:

```bash
cd oskar/beam_utils
mkdir build
cd build
cmake ..
make install
```

The library and header file will be installed by default
into `/usr/local/lib` and `/usr/local/include/oskar/beam_utils`, but the
install prefix can be changed by setting `CMAKE_INSTALL_PREFIX` in the usual way.
