include(FindPackageHandleStandardArgs)

# Find the Sphinx executable.
find_program(
    SPHINX_EXECUTABLE
    NAMES sphinx-build sphinx-build.exe
)
mark_as_advanced(SPHINX_EXECUTABLE)
find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)
