#   meow_fft. My Easy Oresome Wonderful Fast Fourier Transform.
#   Copyright (C) 2017 Richard Maxwell <jodi.the.tigger@gmail.com>
#   This file is part of meow_fft
#   meow_fft is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Affero General Public License for more details.
#   You should have received a copy of the GNU Affero General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>

# ----------------------------------------------------------------------------
# Project info
# ----------------------------------------------------------------------------
set(PROJECT_NAME "meow" C CXX)
project(${PROJECT_NAME})

cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)

# Set a default build type for single-configuration
# CMake generators if no build type is set.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ----------------------------------------------------------------------------

set(meow_fft_bench_fftw3 OFF CACHE BOOL "Use fftw3 in fft benchmark.")

# ----------------------------------------------------------------------------
# Compiler Info
# ----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

# Check compiler options
check_cxx_compiler_flag("-march=native" HAS_NATIVE)
check_cxx_compiler_flag("-fno-rtti" CAN_DISABLE_RTTI)
check_cxx_compiler_flag("-fno-exceptions" CAN_DISABLE_EXCEPTIONS)
check_cxx_compiler_flag("-fsanitize=undefined" SANITIZE_UNDEFINED)
check_cxx_compiler_flag("-fsanitize=address" SANITIZE_ADDRESS)
check_cxx_compiler_flag("-fno-omit-frame-pointer" NO_OMIT_FRAME_POINTER)

check_cxx_compiler_flag(-Wall HAS_WALL)
check_cxx_compiler_flag(-W4 HAS_W4)
check_cxx_compiler_flag(-Wextra HAS_WEXTRA)
check_cxx_compiler_flag(-Werror HAS_WERROR)
check_cxx_compiler_flag(-WX HAS_WERROR_MSVC)
check_cxx_compiler_flag("/MP" HAS_MULTIPROCESSOR_COMPILE)
check_cxx_compiler_flag(-Wno-missing-braces HAS_IGNORE_MISSING_BRACES)
check_cxx_compiler_flag(-Wno-unknown-pragmas HAS_IGNORE_UNKNOWN_PRAGMAS)


# ----------------------------------------------------------------------------
# Prerequisites
# ----------------------------------------------------------------------------

if (HAS_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if (CAN_DISABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

if (CAN_DISABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
endif()

# prefer W4 over WALL for MSVC at least.
if (HAS_W4)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4")
elseif(HAS_WALL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

if (HAS_WEXTRA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()

if (HAS_WERROR)
    set(WARNINGS_AS_ERRORS "${WARNINGS_AS_ERRORS} -Werror")
endif()

if (HAS_WERROR_MSVC)
    set(WARNINGS_AS_ERRORS "${WARNINGS_AS_ERRORS} -WX")
endif()

if (HAS_MULTIPROCESSOR_COMPILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

if (HAS_IGNORE_MISSING_BRACES)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
endif()

if (HAS_IGNORE_UNKNOWN_PRAGMAS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Debug
if (SANITIZE_UNDEFINED)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined")
endif()

if (SANITIZE_ADDRESS)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
endif()

if (NO_OMIT_FRAME_POINTER)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
endif()


# ----------------------------------------------------------------------------
# Compiler Features
# ----------------------------------------------------------------------------
set(NEEDED_FEATURES
    cxx_auto_type
    cxx_trailing_return_types
    cxx_right_angle_brackets
    cxx_nullptr)

# ----------------------------------------------------------------------------
# Source
# ----------------------------------------------------------------------------
set(SOURCE_FFT_BENCH
    meow_fft.h
    test/meow_fft_test_and_bench.cpp)

set(SOURCE_KISS_FFT
    third-party/kiss_fft/kiss_fft.c
    third-party/kiss_fft/_kiss_fft_guts.h
    third-party/kiss_fft/kiss_fft.h
    third-party/kiss_fft/kiss_fftr.c
    third-party/kiss_fft/kiss_fftr.h)

set(SOURCE_PFFFT
    third-party/pffft/pffft.c
    third-party/pffft/pffft.h)

# ----------------------------------------------------------------------------
# Link
# ----------------------------------------------------------------------------
include_directories(.)
include_directories(third-party/kiss_fft)
include_directories(third-party/pffft)

# ----------------------------------------------------------------------------
# FFTW3
# ----------------------------------------------------------------------------
if (meow_fft_bench_fftw3)
    add_definitions(-DTEST_WITH_FFTW3)
endif()

add_library(kiss_fft ${SOURCE_KISS_FFT})
add_library(pffft_fft ${SOURCE_PFFFT})

add_executable(${PROJECT_NAME}_fft_bench ${SOURCE_FFT_BENCH})

target_link_libraries(${PROJECT_NAME}_fft_bench kiss_fft)
target_link_libraries(${PROJECT_NAME}_fft_bench pffft_fft)

target_compile_features(${PROJECT_NAME}_fft_bench PRIVATE ${NEEDED_FEATURES})

# ----------------------------------------------------------------------------
# FFTW3
# ----------------------------------------------------------------------------
if (meow_fft_bench_fftw3)
    target_link_libraries(${PROJECT_NAME}_fft_bench fftw3f)
endif()
