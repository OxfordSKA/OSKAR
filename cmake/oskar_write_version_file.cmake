# Macro to find the find the Subversion revision.
macro(get_svn_revision dir variable)
    find_program(SVN_EXECUTABLE svn DOC "subversion command line client")
    if (SVN_EXECUTABLE AND EXISTS ${OSKAR_SOURCE_DIR}/.svn)
        execute_process(COMMAND
            ${SVN_EXECUTABLE} info ${dir}/oskar_global.h
            OUTPUT_VARIABLE ${variable}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "^(.*\n)?Revision: ([^\n]+).*"
            "\\2" ${variable} "${${variable}}")
    endif()
endmacro(get_svn_revision)

get_svn_revision(${OSKAR_SOURCE_DIR} svn_revision)
file(REMOVE ${OSKAR_SOURCE_DIR}/version.txt)
if (svn_revision)
    file(WRITE ${OSKAR_SOURCE_DIR}/version.txt "OSKAR ${VERSION}")
endif()
