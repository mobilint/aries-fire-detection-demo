set(QBRUNTIME_PATH "" CACHE PATH "Path to qbruntime package root (contains include/, lib/, and on Windows bin/)")
option(FIRE_DETECTION_VENDOR_YAML_CPP "Build yaml-cpp from source instead of using a system package." OFF)

include(FetchContent)

add_library(qbruntime SHARED IMPORTED GLOBAL)
if(WIN32)
    if(NOT QBRUNTIME_PATH)
        message(FATAL_ERROR "On Windows, -DQBRUNTIME_PATH=<path> is required.")
    endif()

    get_filename_component(QBRUNTIME_ABS_PATH
        "${QBRUNTIME_PATH}"
        ABSOLUTE
        BASE_DIR "${CMAKE_BINARY_DIR}"
    )

    set_target_properties(qbruntime PROPERTIES
        IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
        IMPORTED_IMPLIB_DEBUG "${QBRUNTIME_ABS_PATH}/lib/qbruntimed.lib"
        IMPORTED_LOCATION_DEBUG "${QBRUNTIME_ABS_PATH}/bin/qbruntimed.dll"
        IMPORTED_IMPLIB_RELEASE "${QBRUNTIME_ABS_PATH}/lib/qbruntime.lib"
        IMPORTED_LOCATION_RELEASE "${QBRUNTIME_ABS_PATH}/bin/qbruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${QBRUNTIME_ABS_PATH}/include"
    )
else()
    if(QBRUNTIME_PATH)
        get_filename_component(QBRUNTIME_ABS_PATH
            "${QBRUNTIME_PATH}"
            ABSOLUTE
            BASE_DIR "${CMAKE_BINARY_DIR}"
        )
        set(QBRUNTIME_INCLUDE_DIR "${QBRUNTIME_ABS_PATH}/include")
        set(QBRUNTIME_LIBRARY "${QBRUNTIME_ABS_PATH}/lib/libqbruntime.so")
    else()
        find_path(QBRUNTIME_INCLUDE_DIR
            NAMES qbruntime/qbruntime.h
        )
        find_library(QBRUNTIME_LIBRARY
            NAMES qbruntime
        )
    endif()

    if(NOT QBRUNTIME_INCLUDE_DIR OR NOT QBRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "qbruntime not found. Install mobilint-qb-runtime or set -DQBRUNTIME_PATH=<path>."
        )
    endif()

    set_target_properties(qbruntime PROPERTIES
        IMPORTED_LOCATION "${QBRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${QBRUNTIME_INCLUDE_DIR}"
    )
endif()

if(NOT FIRE_DETECTION_VENDOR_YAML_CPP)
    find_package(yaml-cpp CONFIG QUIET)
endif()

if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
elseif(
    EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/yaml-cpp-src/include/yaml-cpp/yaml.h"
    AND EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/yaml-cpp-build/libyaml-cpp.a"
)
    add_library(yaml-cpp-local STATIC IMPORTED GLOBAL)
    set_target_properties(yaml-cpp-local PROPERTIES
        IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/build/_deps/yaml-cpp-build/libyaml-cpp.a"
        INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/build/_deps/yaml-cpp-src/include"
    )
    set(YAML_CPP_TARGET yaml-cpp-local)
else()
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0
    )
    FetchContent_MakeAvailable(yaml-cpp)
    if(TARGET yaml-cpp::yaml-cpp)
        set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
    else()
        set(YAML_CPP_TARGET yaml-cpp)
    endif()
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(DEMO_SOURCES
    demo.cc
    feeder.cc
    model.cc
    model_inference.cc
    post_yolo_anchorless.cc
    post_yolo_anchorless_fire_detection.cc
    yaml_util.cc
)

add_executable(demo ${DEMO_SOURCES})
target_link_libraries(demo PRIVATE
    qbruntime
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    ${YAML_CPP_TARGET}
)
target_include_directories(demo PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(demo PRIVATE cxx_std_17)
target_compile_definitions(demo PRIVATE USE_ARIES2)

if(MSVC)
    set_target_properties(demo PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    find_file(OPENCV_FFMPEG_DLL
        NAMES opencv_videoio_ffmpeg*.dll
        HINTS
            "${OpenCV_DIR}"
            "${OpenCV_DIR}/../.."
            "${OpenCV_DIR}/.."
        PATH_SUFFIXES
            x64/vc17/bin
            x64/vc16/bin
            x64/vc15/bin
            bin
    )

    add_custom_command(TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:demo>
        $<TARGET_FILE_DIR:demo>
        COMMAND_EXPAND_LISTS
    )

    if(OPENCV_FFMPEG_DLL)
        add_custom_command(TARGET demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_FFMPEG_DLL}"
            "$<TARGET_FILE_DIR:demo>"
        )
    endif()
endif()

if(DEMO_DEV_MODE)
    target_compile_definitions(demo PRIVATE USE_SLEEP_DRIVER)
endif()
