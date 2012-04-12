#
# src/matlab/binary_file/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+binary_file")

oskar_mex(query       oskar_mex_binary_file_query.cpp)
oskar_mex(read_record oskar_mex_binary_file_read_record.cpp)
oskar_mex(read_group  oskar_mex_binary_file_read_group.cpp)
oskar_mex(read_all    oskar_mex_binary_file_read_all.cpp)


