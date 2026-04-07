cmake_minimum_required(VERSION 3.16..3.22)

project(ARIESMultiChannelDemo LANGUAGES CXX)

# NOTE(kibum): MSVC는 cp949의 소스를 기대합니다. 우리에 맞게 utf-8로 설정합니다.
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

option(DEMO_DEV_MODE "Use SleepDevice" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

add_subdirectory(src/demo)
