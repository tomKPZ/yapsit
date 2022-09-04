cmake_minimum_required(VERSION 3.1.0)

project(Yapsit C)

add_custom_command(
  OUTPUT pokemon.c
  COMMAND python3 gen_header.py > pokemon.c
  DEPENDS gen_header.py)

add_executable(yapsit yapsit.c pokemon.c)
