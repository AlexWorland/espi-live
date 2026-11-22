cmake_minimum_required(VERSION 3.21)
project(espi_diff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # The per-pixel loops rely on the optimiser to vectorise them.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ESPI_DIFF_SPINNAKER AUTO CACHE STRING "FLIR camera support: AUTO (if the SDK is found), ON (required) or OFF")
set_property(CACHE ESPI_DIFF_SPINNAKER PROPERTY STRINGS AUTO ON OFF)
set(SPINNAKER_ROOT "" CACHE PATH "Spinnaker SDK folder holding include/ and lib/ (or lib64/)")
option(ESPI_DIFF_BUILD_VIEWER "Build the viewer (needs SDL3); OFF builds only the tests" ON)

function(espi_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# Difference, display mapping, luma extraction, synthetic source and the
# processing thread. No SDL, no Spinnaker, no OpenCV: only the C++ library.
add_library(espi_core STATIC
  src/difference.cpp
  src/display.cpp
  src/luma.cpp
  src/pipeline.cpp
  src/synthetic_source.cpp
)
target_include_directories(espi_core PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(espi_core PUBLIC Threads::Threads)
espi_warnings(espi_core)

enable_testing()
add_executable(espi_core_tests tests/core_tests.cpp)
target_link_libraries(espi_core_tests PRIVATE espi_core)
espi_warnings(espi_core_tests)
add_test(NAME core_tests COMMAND espi_core_tests)

if(NOT ESPI_DIFF_BUILD_VIEWER)
  return()
endif()

# --- Spinnaker (FLIR cameras) -------------------------------------------------
set(espi_have_spinnaker FALSE)
if(NOT ESPI_DIFF_SPINNAKER STREQUAL "OFF")
  find_path(SPINNAKER_INCLUDE_DIR Spinnaker.h
    HINTS "${SPINNAKER_ROOT}/include" "${SPINNAKER_ROOT}/include/spinnaker"
    PATHS "C:/Program Files/Teledyne/Spinnaker/include"
          "C:/Program Files/FLIR Systems/Spinnaker/include"
          /opt/spinnaker/include /usr/include/spinnaker /usr/local/include/spinnaker)
  find_library(SPINNAKER_LIBRARY NAMES Spinnaker_v140 Spinnaker
    HINTS "${SPINNAKER_ROOT}/lib64/vs2015" "${SPINNAKER_ROOT}/lib"
    PATHS "C:/Program Files/Teledyne/Spinnaker/lib64/vs2015"
          "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015"
          /opt/spinnaker/lib /usr/local/lib)
  # The Windows SDK ships a separate import library for debug builds.
  find_library(SPINNAKER_LIBRARY_DEBUG NAMES Spinnakerd_v140
    HINTS "${SPINNAKER_ROOT}/lib64/vs2015"
    PATHS "C:/Program Files/Teledyne/Spinnaker/lib64/vs2015"
          "C:/Program Files/FLIR Systems/Spinnaker/lib64/vs2015")
  if(SPINNAKER_INCLUDE_DIR AND SPINNAKER_LIBRARY)
    set(espi_have_spinnaker TRUE)
    message(STATUS "Spinnaker: ${SPINNAKER_LIBRARY}")
  elseif(ESPI_DIFF_SPINNAKER STREQUAL "ON")
    message(FATAL_ERROR "Spinnaker SDK not found; set SPINNAKER_ROOT to its install folder")
  else()
    message(STATUS "Spinnaker SDK not found: building without FLIR camera support "
                   "(set SPINNAKER_ROOT, or -DESPI_DIFF_SPINNAKER=ON to require it)")
  endif()
endif()

# --- SDL3 (window, and video capture devices) ---------------------------------
find_package(SDL3 REQUIRED CONFIG COMPONENTS SDL3)

add_executable(espi_diff src/main.cpp src/capture_source.cpp)
target_link_libraries(espi_diff PRIVATE espi_core SDL3::SDL3)
espi_warnings(espi_diff)

if(espi_have_spinnaker)
  target_sources(espi_diff PRIVATE src/blackfly_source.cpp)
  target_include_directories(espi_diff SYSTEM PRIVATE "${SPINNAKER_INCLUDE_DIR}")
  if(SPINNAKER_LIBRARY_DEBUG)
    target_link_libraries(espi_diff PRIVATE optimized "${SPINNAKER_LIBRARY}" debug "${SPINNAKER_LIBRARY_DEBUG}")
  else()
    target_link_libraries(espi_diff PRIVATE "${SPINNAKER_LIBRARY}")
  endif()
else()
  target_sources(espi_diff PRIVATE src/blackfly_unavailable.cpp)
endif()

if(WIN32)
  # Put SDL3.dll next to the executable. (Spinnaker's DLLs are on PATH after
  # the SDK installer runs.)
  add_custom_command(TARGET espi_diff POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:espi_diff> $<TARGET_FILE_DIR:espi_diff>
    COMMAND_EXPAND_LISTS)
endif()
