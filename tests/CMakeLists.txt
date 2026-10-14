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

add_executable(test6 ${CMAKE_CURRENT_SOURCE_DIR}/test5.cu)
target_include_directories(test6 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_compile_definitions(test6 PUBLIC CUSVNA_TEST_AF1)
add_test(NAME soundness_af1 COMMAND $<TARGET_FILE:test6>)

add_executable(test7 ${CMAKE_CURRENT_SOURCE_DIR}/test5.cu)
target_include_directories(test7 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_compile_definitions(test7 PUBLIC CUSVNA_TEST_DEVICE)
target_compile_options(test7 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
add_test(NAME soundness_device COMMAND $<TARGET_FILE:test7>)

add_executable(test8 ${CMAKE_CURRENT_SOURCE_DIR}/test5.cu)
target_include_directories(test8 PUBLIC ${CUSVNA_INCLUDE_DIR})
target_compile_definitions(test8 PUBLIC CUSVNA_TEST_AF1 CUSVNA_TEST_DEVICE)
target_compile_options(test8 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
add_test(NAME soundness_af1_device COMMAND $<TARGET_FILE:test8>)
