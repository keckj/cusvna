set(LIBAFFA_ROOTDIR "${CUSVNA_EXT_DIR}/libaffa")
set(LIBAFFA_INCLUDE_DIR "${LIBAFFA_ROOTDIR}/src")

file(GLOB LIBAFFA_SRCS "${LIBAFFA_INCLUDE_DIR}/*.cpp")

if (NOT LIBAFFA_SRCS)
    message(FATAL_ERROR "Cannot find libaffa srcs, did you clone the repository recursively ?")
endif()

add_library(affa STATIC ${LIBAFFA_SRCS})
