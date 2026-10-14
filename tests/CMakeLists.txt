include(libaffa) 

enable_testing()

add_executable(test0 ${CMAKE_CURRENT_SOURCE_DIR}/test0.cu)
target_include_directories(test0 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_include_directories(test0 PUBLIC ${LIBAFFA_INCLUDE_DIR})
target_link_libraries(test0 affa)
add_test(NAME operators COMMAND $<TARGET_FILE:test0>)

add_executable(test1 ${CMAKE_CURRENT_SOURCE_DIR}/test1.cu)
target_include_directories(test1 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_include_directories(test1 PUBLIC ${LIBAFFA_INCLUDE_DIR})
target_link_libraries(test1 affa)
add_test(NAME multiply_add COMMAND $<TARGET_FILE:test1>)

add_executable(test2 ${CMAKE_CURRENT_SOURCE_DIR}/test2.cu)
target_include_directories(test2 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_include_directories(test2 PUBLIC ${LIBAFFA_INCLUDE_DIR})
target_link_libraries(test2 affa)
add_test(NAME 1d_function COMMAND $<TARGET_FILE:test2>)

add_executable(test3 ${CMAKE_CURRENT_SOURCE_DIR}/test3.cu)
target_include_directories(test3 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_include_directories(test3 PUBLIC ${LIBAFFA_INCLUDE_DIR})
target_link_libraries(test3 affa)
add_test(NAME 2d_function COMMAND $<TARGET_FILE:test3>)

add_executable(test4 ${CMAKE_CURRENT_SOURCE_DIR}/test4.cu)
target_include_directories(test4 PUBLIC ${CUSVNA_INCLUDE_DIR})
add_test(NAME generated_symbols COMMAND $<TARGET_FILE:test4>)

add_executable(test5 ${CMAKE_CURRENT_SOURCE_DIR}/test5.cu)
target_include_directories(test5 PUBLIC ${CUSVNA_INCLUDE_DIR})
add_test(NAME soundness COMMAND $<TARGET_FILE:test5>)
