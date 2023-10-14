cmake_minimum_required(VERSION 3.5)
project(G6KDist VERSION 0.1)

# ##############################################################################
# Doctest
# ##############################################################################

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/doctest/scripts/cmake")
include(common)
include_directories(SYSTEM "doctest" "doctest/doctest")
enable_testing()

# We'll use our own ones, not the ones that doctest wants us to use.
# This is primarily because G6K isn't fully warning compliant.
set(CMAKE_CXX_FLAGS "")

# ##############################################################################
# General flags
# ##############################################################################

# We use C++17 in this project because it makes certain templates easier to
# write and understand.  At this point it's widely available and used (Chromium, for example,
# uses it by default).
set(CMAKE_CXX_STANDARD 17)

# This produces a json file that let's you jump into various file definitions
# etc This can be useful if you use VSCode, but it's also useful if you have an
# editor that supports clangd more generally.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##############################################################################
# MPI
# ##############################################################################
find_package(MPI REQUIRED COMPONENTS CXX)

# ##############################################################################
# Warnings
# ##############################################################################

# Feel free to add to these. Note that GCC and Clang each support different flags,
# so we'll only add extra flags if we're using GCC.

set(WARNINGS
    -Wall
    -Wextra
    #-Werror
    -Wshadow
    -Wpedantic
    -Wextra-semi
    -Wredundant-decls
    -Wctor-dtor-privacy
    -Wzero-as-null-pointer-constant
    -Wfloat-equal
    -Wformat-security
    -Wnon-virtual-dtor
    -Wunused
    -Woverloaded-virtual
    -Wsign-conversion
    -Wconversion
    -Wmisleading-indentation
    -Wnull-dereference
    -Wshift-overflow
    -Wdouble-promotion
    -Wformat=2
    -Winline
    -Wstrict-overflow=5
    -Weffc++
    -Wcast-qual
    -Wmissing-declarations
    -Wsign-promo
    -Wdisabled-optimization)


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(
    APPEND
    WARNINGS
    -Wstrict-null-sentinel
    -Wformat-signedness
    -Wformat-overflow
    -Wduplicated-branches
    -Wlogical-op
    -Wuseless-cast
    -Wstrict-overflow=5)
endif()

# ##############################################################################
# ASAN
# ##############################################################################

# LIB_ASAN is a memory bounds checker. It should come with your C++ compiler,
# but if it doesn't exist then you can comment this out.
# Note: this if block is to handle the (seemingly common situation) of compiling
# on an M1 Mac: there, ASAN isn't supported it seems. Note that the ordering matters
# here, as UNIX is set on Macos. 
if(APPLE)
  set(LIB_ASAN)
elseif(UNIX)
  set(LIB_ASAN -fno-omit-frame-pointer -fsanitize=address)
else()
  set(LIB_ASAN)
endif()
# #############################################################################
# FLAGS
# #############################################################################

# We compile twice: once in release, and once in debug. This is to help with debugging.
# N.B MPI_DIST_TEST essentially controls whether a load of functions are marked as noexcept
# or not. 
set(G_C_DEBUG_FLAGS "-g3 -D_GLIBCXX_DEBUG -DMPI_DIST_TEST -DG6K_MPI -DENABLE_EXTENDED_STATS" "${WARNINGS}" "${LIB_ASAN}")
set(G_C_OPT_FLAGS "-O3 -DNDEBUG -DMPI_DIST_TEST -DG6K_MPI -DENABLE_EXTENDED_STATS" "${WARNINGS}")

# We also need to do GCC specific things for optimization flags. Note that AppleClang needs to be handled separately
# because it supports fat binaries, which was a problem when compiling G6K for Macs in the past.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND G_C_OPT_FLAGS
    -march=native)
  list(APPEND G_C_DEBUG_FLAGS
    -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  list(APPEND G_C_OPT_FLAGS
    -mcpu=apple-m1)
  list(APPEND G_C_DEBUG_FLAGS
    -mcpu=apple-m1)
endif()

# These are our linker flags.
set(G_L_FLAGS MPI::MPI_CXX ${LIB_ASAN})

# Now we have to split everything up. This is because of how CMAKE
# handles argument lists. 
separate_arguments(C_DEBUG_FLAGS UNIX_COMMAND "${G_C_DEBUG_FLAGS}")
separate_arguments(C_OPT_FLAGS UNIX_COMMAND "${G_C_OPT_FLAGS}")
separate_arguments(L_FLAGS UNIX_COMMAND "${G_L_FLAGS}")

# ##############################################################################
# Compilation
# ##############################################################################

set(FILES main.cpp kernel/layouts.cpp kernel/mpi_cast.cpp kernel/shuffle_siever.cpp kernel/dist_siever.cpp kernel/siever_types.cpp kernel/topology.cpp kernel/mpi_obj.cpp kernel/mpi_wrapper.cpp kernel/bdgl_sieve.cpp kernel/bgj1_sieve.cpp kernel/control.cpp kernel/cpuperf.cpp kernel/fht_lsh.cpp kernel/hk3_sieve.cpp kernel/params.cpp kernel/sieving.cpp kernel/context_change.cpp)

# Because we put our tests alongside our code, we just need a single executable
# for the whole project.
add_executable(main_debug ${FILES})
target_compile_options(main_debug PRIVATE ${C_DEBUG_FLAGS})
target_link_libraries(main_debug PRIVATE ${L_FLAGS} ${CMAKE_THREAD_LIBS_INIT})
doctest_add_test(NO_OUTPUT NAME main_debug COMMAND mpirun -np 3 $<TARGET_FILE:main_debug>)

add_executable(main_opt ${FILES})
target_compile_options(main_opt PRIVATE ${C_OPT_FLAGS})
target_link_libraries(main_opt PRIVATE ${L_FLAGS} ${CMAKE_THREAD_LIBS_INIT})
doctest_add_test(NO_OUTPUT NAME main_opt COMMAND mpirun -np 3 $<TARGET_FILE:main_opt>)




