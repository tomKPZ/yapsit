cmake_minimum_required(VERSION 3.10)

project(Yapsit C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS "-Wall -Wextra -Werror -Wshadow")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_C_COMPILER "clang")

add_custom_command(
  OUTPUT sprites.c constants.h
  COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/gen_header.py"
  DEPENDS gen_header.py)

add_executable(yapsit yapsit.c  "${CMAKE_CURRENT_SOURCE_DIR}/sprites.c")
find_library(ZSTD_LIB zstd REQUIRED)
target_link_libraries(yapsit PRIVATE ${ZSTD_LIB})
set_target_properties(yapsit PROPERTIES LINK_FLAGS_RELEASE -s)
install(TARGETS yapsit DESTINATION bin)
