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

file(GLOB AXD_SOURCES CONFIGURE_DEPENDS
  src/runtime/*.cpp
  src/components/*/*.cpp
  src/models/*.cpp
  src/models/*/*.cpp
  src/server/*.cpp)
add_library(axdcore STATIC ${AXD_SOURCES} third_party/unicode.cpp third_party/unicode-data.cpp)
target_include_directories(axdcore PUBLIC
  src/runtime
  src/components/tokenizer src/components/text_qwen src/components/dit_h3 src/components/lora src/components/vae_image
  src/models src/models/minimax_h3_image
  src/server
  third_party ${AXCL_ROOT}/include)
target_link_directories(axdcore PUBLIC ${AXCL_ROOT}/lib)
target_link_libraries(axdcore PUBLIC axcl_rt pthread)

foreach(tool axdiffusion-cli axdiffusion-server)
  string(REPLACE "-" "_" src ${tool})
  add_executable(${tool} tools/${src}.cpp)
  target_link_libraries(${tool} axdcore -static-libstdc++ -static-libgcc)
  set_target_properties(${tool} PROPERTIES BUILD_RPATH "${AXCL_ROOT}/lib" INSTALL_RPATH "${AXCL_ROOT}/lib")
endforeach()
install(TARGETS axdiffusion-cli axdiffusion-server RUNTIME DESTINATION bin)
