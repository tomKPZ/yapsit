cmake_minimum_required(VERSION 3.1.0)

project(Yapsit C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE MinSizeRel)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_custom_command(
  OUTPUT pokemon.c
  COMMAND python3 gen_header.py > pokemon.c
  DEPENDS gen_header.py)

add_executable(yapsit yapsit.c pokemon.c)
set_target_properties(yapsit PROPERTIES LINK_FLAGS_RELEASE -s)
install(TARGETS yapsit DESTINATION bin)
