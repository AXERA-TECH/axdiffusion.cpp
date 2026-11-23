# Cross-compile on an x86 host for the AX8860 board (aarch64, glibc 2.38):
#
#   cmake -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
#         -DTOOLCHAIN_ROOT=/path/to/arm-gnu-toolchain-13.x-x86_64-aarch64-none-linux-gnu \
#         -DAXCL_ROOT=/path/to/axhelix        # a copy of the board's /usr/local/axhelix (include/ and lib/)
#   cmake --build build-aarch64 -j
#
# Use Arm GNU toolchain 13.2 or 13.3; older toolchains target an older glibc than the board runtime needs.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(TOOLCHAIN_ROOT "" CACHE PATH "Arm GNU toolchain directory (contains bin/aarch64-none-linux-gnu-g++)")
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES TOOLCHAIN_ROOT)
if(TOOLCHAIN_ROOT)
  set(_axd_prefix "${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-")
else()
  set(_axd_prefix "aarch64-none-linux-gnu-")
endif()
set(CMAKE_C_COMPILER "${_axd_prefix}gcc")
set(CMAKE_CXX_COMPILER "${_axd_prefix}g++")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
