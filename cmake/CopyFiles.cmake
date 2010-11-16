#
# Defines a macro that creates a custom build target to copy files based on a
# specified pattern (e.g. *.dat).
#
# (The custom build target is addded to the all target of make.)
#
# This is designed to be used to copy data files needed in testing from the
# source to the build directory.
#
#
#
macro(copy_files GLOBPAT DESTINATION target)

    file(GLOB COPY_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${GLOBPAT})

    add_custom_target(${target} ALL COMMENT "Copying files: ${GLOBPAT}")

    foreach(FILENAME ${COPY_FILES})
        set(src "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
        set(dst "${DESTINATION}/${FILENAME}")
        add_custom_command(TARGET ${target} COMMAND ${CMAKE_COMMAND} -E copy ${src} ${dst})
    endforeach(FILENAME)

endmacro(copy_files)
