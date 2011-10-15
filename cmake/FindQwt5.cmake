# Find the Qwt 5.x includes and library, either the version linked to Qt3 or the version linked to Qt4
#
# On Windows it makes these assumptions:
# - the Qwt DLL is where the other DLLs for Qt are (QT_DIR\bin) or in the path
# - the Qwt .h files are in QT_DIR\include\Qwt or in the path
# - the Qwt .lib is where the other LIBs for Qt are (QT_DIR\lib) or in the path
#
# Qwt5_INCLUDE_DIR - where to find qwt.h if Qwt
# Qwt5_Qt4_LIBRARY - The Qwt5 library linked against Qt4 (if it exists)
# Qwt5_Qt3_LIBRARY - The Qwt5 library linked against Qt4 (if it exists)
# Qwt5_Qt4_FOUND - Qwt5 was found and uses Qt4
# Qwt5_Qt3_FOUND - Qwt5 was found and uses Qt3
# Qwt5_FOUND - Set to TRUE if Qwt5 was found (linked either to Qt3 or Qt4)

# Copyright (c) 2007, Pau Garcia i Quiles, <pgquiles@elpauer.org>

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

IF(Qwt5_Qt4_LIBRARY OR Qwt5_Qt3_LIBRARY AND Qwt5_INCLUDE_DIR)
    SET(Qwt5_FIND_QUIETLY TRUE)
ENDIF(Qwt5_Qt4_LIBRARY OR Qwt5_Qt3_LIBRARY AND Qwt5_INCLUDE_DIR)

IF(NOT QT4_FOUND)
    FIND_PACKAGE( Qt4 REQUIRED QUIET )
ENDIF(NOT QT4_FOUND)

IF( QT4_FOUND )

    # Is Qwt5 installed? Look for header files
    FIND_PATH( Qwt5_INCLUDE_DIR qwt.h
    PATHS ${QT_INCLUDE_DIR} /usr/local/qwt/include /usr/local/include /usr/include/qwt /usr/include
    PATH_SUFFIXES qwt qwt5 qwt-qt4 qwt5-qt4 qwt-qt3 qwt5-qt3 include qwt/include qwt5/include qwt-qt4/include qwt5-qt4/include qwt-qt3/include qwt5-qt3/include ENV PATH)

    # Find Qwt version
    IF( Qwt5_INCLUDE_DIR )
        FILE( READ ${Qwt5_INCLUDE_DIR}/qwt_global.h QWT_GLOBAL_H )
        STRING( REGEX MATCH "0x05" QWT_IS_VERSION_5 ${QWT_GLOBAL_H})

        IF( QWT_IS_VERSION_5 )
            STRING(REGEX REPLACE ".*#define[\\t\\ ]+QWT_VERSION_STR[\\t\\ ]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1" Qwt_VERSION "${QWT_GLOBAL_H}")
            
            # Find Qwt5 library linked to Qt4
            FIND_LIBRARY( Qwt5_Qt4_TENTATIVE_LIBRARY NAMES qwt5-qt4 qwt-qt4 qwt5 qwt PATHS /usr/local/qwt/lib /usr/local/lib /usr/lib ${QT_LIBRARY_DIR})
            IF( UNIX AND NOT CYGWIN)
            IF( Qwt5_Qt4_TENTATIVE_LIBRARY )
            IF( NOT APPLE )
            EXECUTE_PROCESS( COMMAND "ldd" ${Qwt5_Qt4_TENTATIVE_LIBRARY} OUTPUT_VARIABLE Qwt_Qt4_LIBRARIES_LINKED_TO )
            ENDIF( NOT APPLE )
            IF( APPLE )
            EXECUTE_PROCESS( COMMAND "otool" "-L" ${Qwt5_Qt4_TENTATIVE_LIBRARY} OUTPUT_VARIABLE Qwt_Qt4_LIBRARIES_LINKED_TO )
            ENDIF( APPLE )
            STRING( REGEX MATCH "QtCore" Qwt5_IS_LINKED_TO_Qt4 ${Qwt_Qt4_LIBRARIES_LINKED_TO})
            IF( Qwt5_IS_LINKED_TO_Qt4 )
            SET( Qwt5_Qt4_LIBRARY ${Qwt5_Qt4_TENTATIVE_LIBRARY} )
            SET( Qwt5_Qt4_FOUND TRUE )
            IF (NOT Qwt5_FIND_QUIETLY)
            MESSAGE( STATUS "Found Qwt: ${Qwt5_Qt4_LIBRARY}" )
            ENDIF (NOT Qwt5_FIND_QUIETLY)
            ENDIF( Qwt5_IS_LINKED_TO_Qt4 )
            ENDIF( Qwt5_Qt4_TENTATIVE_LIBRARY )
            ELSE( UNIX AND NOT CYGWIN)
            # Assumes qwt.dll is in the Qt dir
            SET( Qwt5_Qt4_LIBRARY ${Qwt5_Qt4_TENTATIVE_LIBRARY} )
            SET( Qwt5_Qt4_FOUND TRUE )
            IF (NOT Qwt5_FIND_QUIETLY)
            MESSAGE( STATUS "Found Qwt version ${Qwt_VERSION} linked to Qt4" )
            ENDIF (NOT Qwt5_FIND_QUIETLY)
            ENDIF( UNIX AND NOT CYGWIN)
        ENDIF( QWT_IS_VERSION_5 )

        IF( Qwt5_Qt4_FOUND OR Qwt5_Qt3_FOUND )
            SET( Qwt5_FOUND TRUE )
        ENDIF( Qwt5_Qt4_FOUND OR Qwt5_Qt3_FOUND )
        MARK_AS_ADVANCED( Qwt5_INCLUDE_DIR Qwt5_Qt4_LIBRARY Qwt5_Qt3_LIBRARY )
    ENDIF( Qwt5_INCLUDE_DIR )
    IF (NOT Qwt5_FOUND AND Qwt5_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find Qwt 5.x")
    ENDIF (NOT Qwt5_FOUND AND Qwt5_FIND_REQUIRED)
ENDIF( QT4_FOUND )
