include(libaffa)

add_executable(test1 ${CMAKE_CURRENT_SOURCE_DIR}/test1.cu)
target_include_directories(test1 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_include_directories(test1 PUBLIC ${LIBAFFA_INCLUDE_DIR})
target_link_libraries(test1 affa)
