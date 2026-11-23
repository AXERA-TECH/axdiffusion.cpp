# 编译指南

axdiffusion.cpp 只依赖板子自带的 AXCL 运行库，其余第三方代码都在仓库里。有两种编译方式：

| 方式 | 适合 | 全量编译耗时 |
|---|---|---|
| [板端编译](#板端编译) | 只有一块板子，想最快跑起来 | 几分钟 |
| [交叉编译](#交叉编译) | 经常改代码，或者要给多块板子出包 | 约 20 秒（32 线程 x86 主机） |

两种方式产出的程序相同：`axdiffusion-cli`（命令行出图）和 `axdiffusion-server`（HTTP 服务 + 网页）。
libstdc++ 和 libgcc 静态链接，运行时只需要板上的 `libaxcl_rt.so`、`libm`、`libc`。

## 板端编译

### 准备

- AX8860 板子，系统里有 AXCL 运行库（默认在 `/usr/local/axhelix`，含 `include/` 和 `lib/`）
- g++ ≥ 9（需要 C++17）
- CMake ≥ 3.16

检查一下：

```bash
g++ --version
cmake --version
ls /usr/local/axhelix/include/axcl.h /usr/local/axhelix/lib/libaxcl_rt.so
```

### 编译

```bash
git clone https://github.com/AXERA-TECH/axdiffusion.cpp.git
cd axdiffusion.cpp
cmake -S . -B build
cmake --build build -j8
```

AXCL 运行库不在默认位置时，给 CMake 加 `-DAXCL_ROOT=<目录>`。

产物在 `build/axdiffusion-cli` 和 `build/axdiffusion-server`，可以直接运行。

## 交叉编译

在 x86-64 Linux 主机上编译，把产物拷到板子运行。

### 准备

1. **交叉工具链**：Arm GNU Toolchain 13.2.Rel1 或 13.3.Rel1，`x86_64` 主机、`aarch64-none-linux-gnu` 目标。
   从 Arm 官网下载后解压到任意目录，下文记作 `$TOOLCHAIN`。

   ```bash
   $TOOLCHAIN/bin/aarch64-none-linux-gnu-g++ --version
   ```

   这两个版本自带 glibc 2.38，编出的程序在 glibc ≥ 2.38 的板端系统上能运行。更新的工具链可能要求更高的
   glibc，板上运行时会报 `GLIBC_x.xx not found`。

2. **AXCL 头文件和库**：从板子上把 `/usr/local/axhelix` 的 `include/` 和 `lib/` 拷到主机（约 40 MB），下文记作 `$AXCL`。

   ```bash
   mkdir -p axhelix
   ssh root@<板子地址> "cd /usr/local/axhelix && tar cf - include lib" | tar xf - -C axhelix
   export AXCL=$PWD/axhelix
   ```

   请从要运行程序的那块板子（或同一 SDK 版本的板子）上拷，保证头文件和运行库版本一致。

3. CMake ≥ 3.16。

### 编译

```bash
git clone https://github.com/AXERA-TECH/axdiffusion.cpp.git
cd axdiffusion.cpp
cmake -S . -B build-aarch64 \
      -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
      -DTOOLCHAIN_ROOT=$TOOLCHAIN \
      -DAXCL_ROOT=$AXCL
cmake --build build-aarch64 -j
```

工具链的 `bin` 目录已经在 `PATH` 里时，可以省掉 `-DTOOLCHAIN_ROOT`。

确认产物是 aarch64 程序：

```bash
file build-aarch64/axdiffusion-server
# ELF 64-bit LSB executable, ARM aarch64 ...
```

### 部署到板子

```bash
scp build-aarch64/axdiffusion-cli build-aarch64/axdiffusion-server root@<板子地址>:/data/axdiffusion/
```

程序里写入的运行库搜索路径是 `/usr/local/axhelix/lib`。板上的 AXCL 库在别处时，编译时加
`-DAXCL_RUNTIME_DIR=<板上目录>`，或者运行前设置 `LD_LIBRARY_PATH`。

## CMake 选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` 使用 `-O3` |
| `AXCL_ROOT` | `/usr/local/axhelix` | 编译时用的 AXCL 目录（含 `include/` 和 `lib/`）；交叉编译时指向主机上的拷贝 |
| `AXCL_RUNTIME_DIR` | `/usr/local/axhelix/lib` | 板上 AXCL 库所在目录，只在交叉编译时写入程序 |
| `TOOLCHAIN_ROOT` | 空 | 交叉工具链目录，配合 `cmake/aarch64-toolchain.cmake` 使用 |

## 验证

在板子上：

```bash
./axdiffusion-server            # 不带参数会打印用法，说明程序和运行库都能加载
./axdiffusion-cli --model <模型目录> --prompt "a red fox in snow" --out fox.png
```

模型目录是 Hugging Face 上的 `AXERA-TECH/MiniMax-H3-Image-AX8860`，下载方式和运行参数见 [README](../README.md)。

## 关于内置网页

网页源文件是 `web/index.html`，配置阶段读入并编译进 `axdiffusion-server`。修改这个文件后再次执行
`cmake --build` 会自动重新配置并更新内置页面。

只想改网页、不想重新编译时，启动服务加 `--web-dir <目录>`，服务每次请求都从 `<目录>/index.html` 读取页面。

## 常见问题

**`error while loading shared libraries: libaxcl_rt.so`**
板上的 AXCL 库不在程序记录的路径里。用 `ldd ./axdiffusion-server` 查看，然后设置
`LD_LIBRARY_PATH=<库目录>`，或用 `-DAXCL_RUNTIME_DIR` 重新编译。

**`version 'GLIBC_2.xx' not found`**
交叉工具链的 glibc 比板端系统新。换用 Arm GNU Toolchain 13.2 / 13.3，或者改为板端编译。

**链接时报 `libaxcl_rt.so` 里的符号未定义（交叉编译）**
`AXCL_ROOT` 指向的目录缺少 `lib/` 里的其他 SDK 库。把板上 `/usr/local/axhelix/lib` 整个目录拷过来，不要只拷
`libaxcl_rt.so`。

**板端编译很慢或内存不够**
把 `-j8` 调小，或者改用交叉编译。

**程序能启动但加载模型失败**
编译用的 AXCL 头文件与板上运行库版本不一致时可能出现。请用同一块板子（或同一 SDK 版本）的 `include/` 和 `lib/` 重新编译。
