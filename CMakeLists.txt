cmake_minimum_required(VERSION 3.16)
project(axdiffusion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp-simd")

# AXCL runtime shipped with the board SDK. Override with -DAXCL_ROOT=/path/to/axcl
set(AXCL_ROOT "/usr/local/axhelix" CACHE PATH "AXCL SDK root (contains include/ and lib/)")
# Where libaxcl_rt.so lives on the board at run time (differs from AXCL_ROOT when cross-compiling).
set(AXCL_RUNTIME_DIR "/usr/local/axhelix/lib" CACHE PATH "Directory of the AXCL libraries on the target")

# The built-in web page is web/index.html, compiled in as a string. A running server can also read the page from
# a directory (--web-dir), so the page can be edited without rebuilding.
file(READ ${CMAKE_SOURCE_DIR}/web/index.html AXD_WEB_HTML)
configure_file(src/server/web_ui.hpp.in ${CMAKE_BINARY_DIR}/gen/web_ui.hpp @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/web/index.html)

# libaxcl_rt.so is itself C++ and uses the system libstdc++.so. Linking a second, static copy into the program makes
# the two share template code and allocations across copies, so the shared runtime is the default.
option(AXD_STATIC_LIBSTDCXX "Link libstdc++/libgcc statically (not recommended)" OFF)

file(GLOB AXD_SOURCES CONFIGURE_DEPENDS
  src/runtime/*.cpp
  src/components/*/*.cpp
  src/models/*.cpp
  src/models/*/*.cpp
  src/server/*.cpp)
add_library(axdcore STATIC ${AXD_SOURCES} third_party/unicode.cpp third_party/unicode-data.cpp)
target_include_directories(axdcore PUBLIC
  src/runtime
  src/components/tokenizer src/components/text_qwen src/components/dit_h3 src/components/lora src/components/vae_image src/components/vision_qwen
  src/models src/models/minimax_h3_image
  src/server ${CMAKE_BINARY_DIR}/gen
  third_party ${AXCL_ROOT}/include)
target_link_directories(axdcore PUBLIC ${AXCL_ROOT}/lib)
target_link_libraries(axdcore PUBLIC axcl_rt pthread ${CMAKE_DL_LIBS})
if(CMAKE_CROSSCOMPILING)
  # libaxcl_rt.so depends on other SDK libraries that only need to resolve on the board.
  target_link_options(axdcore PUBLIC -Wl,-rpath-link,${AXCL_ROOT}/lib -Wl,--allow-shlib-undefined)
endif()

foreach(tool axdiffusion-cli axdiffusion-server)
  string(REPLACE "-" "_" src ${tool})
  add_executable(${tool} tools/${src}.cpp)
  target_link_libraries(${tool} axdcore)
  if(AXD_STATIC_LIBSTDCXX)
    target_link_libraries(${tool} -static-libstdc++ -static-libgcc)
  endif()
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(${tool} PROPERTIES SKIP_BUILD_RPATH ON LINK_FLAGS "-Wl,-rpath,${AXCL_RUNTIME_DIR}")
  else()
    set_target_properties(${tool} PROPERTIES BUILD_RPATH "${AXCL_ROOT}/lib" INSTALL_RPATH "${AXCL_ROOT}/lib")
  endif()
endforeach()
install(TARGETS axdiffusion-cli axdiffusion-server RUNTIME DESTINATION bin)
