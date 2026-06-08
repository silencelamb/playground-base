# Task 1: CUDA Programming

High performance gemm implementation on Nvidia A100 ([internal feishu doc](https://aicarrier.feishu.cn/wiki/EvivwNtVRij2XVk0i36cBN8Bn1f)).

## 1. Target

Implement a high performance gemm (General Matrix Multiply) function with CUDA on Nvidia A100 for float32 and float16 data types.

The goal is to push performance as high as possible toward the A100 hardware peak (fp16 **312 TFLOPS** on Tensor Cores), using the given benchmarking structure for scoring. There is no library baseline to match or beat — the hardware peak is the only target.

## 2. Quick Start with Bash Script

We provide a convenient bash script `task1.sh` that offers the same operations as the previous Makefile:

```bash
# Show all available commands
./task1.sh help

# 1) Build code with specified FLOAT type and VERSION
./task1.sh build --float f32 --ver 0

# 2) Build and run code, automatically save logs
./task1.sh run --float f16 --ver 0

# 3) Build with debug symbols (RelWithDebInfo)
./task1.sh debug --float f32 --ver 2

# 4) Profile with nsight compute, save reports
./task1.sh profile --float f16 --ver 2

# Clean build files
./task1.sh clean

# Clean log files  
./task1.sh clean-logs
```

### Key Features:
- **Automatic Logging**: Run results are saved to `logs/` directory with timestamp and version info
- **TFLOPS and Error Tracking**: Captures performance metrics and error rates automatically
- **Nsight Compute Integration**: Profile reports saved to `logs/profiles/` with timestamp
- **Selective, fast compile**: only `main` + `v0` (cBLAS reference) + the file for the selected `--ver N` are compiled — not every version in the folder. Builds stay fast as versions pile up, and a broken/experimental sibling version never blocks the one you're building. Re-running the *same* version skips the CMake reconfigure and goes straight to an incremental `ninja`.

## 3. Benchmark cBLAS (correctness reference)

`v0 -> cBLAS` (CPU). It is the **correctness ground truth** only (used to compute Average Error); it is far too slow to be a performance reference. Your own kernels start at `v1`.

> Average Error = mean relative error `mean(|GT - C| / |GT|)`. Elements whose reference `GT` rounds to exactly `0` (rare cancellation, esp. in fp16) are skipped — relative error is undefined there — instead of producing `inf`. A kernel that writes `inf`/`nan` into `C` still fails the check.

### Using build script directly:
```bash
# Build gemm implemented with CBLAS (CPU) under float32:
bash scripts/build-task1.sh -f32 -v0
# Build gemm implemented with CBLAS (CPU) under float16:
bash scripts/build-task1.sh -f16 -v0
```

### Using task1.sh script (Recommended):
```bash
# Build gemm implemented with CBLAS (CPU) under float32:
./task1.sh build --float f32 --ver 0
# Build gemm implemented with CBLAS (CPU) under float16:
./task1.sh build --float f16 --ver 0
```

For more compile options, see "[./scripts/build-task1.sh](../scripts/build-task1.sh)" or run `./task1.sh help`.

> 💡**Note**:  
> 1. Please install the following extensions in VSCode:
>    - llvm-vs-code-extensions.vscode-clangd
>    - twxs.cmake
>    - josetr.cmake-language-support-vscode
> 2. It is suggested to restart clangd server after building (to avoid some code analysis errors).  
> To restart clangd server, press `Ctrl+Shift+P` in VSCode, and select `clangd: Restart language server`.  
> ![restart-clangd](../docs/imgs/restart-clangd.png)

Run the binarys in "[./build/src](../build/src)" directory to get the benchmark results.

### Running directly:
You can set `m`, `n`, `k`, `n_warmup` and `n_test` by passing arguments to binarys built in this task. Use `-h` to print help messages:

```bash
# Run the binary but showing help messages only
./build/src/task1_float16_v0 -h
```

### Running with task1.sh script (Recommended):
```bash
# Build and run with automatic logging
./task1.sh run --float f16 --ver 0

# Run with different configurations
./task1.sh run --float f32 --ver 0
./task1.sh run --float f16 --ver 2
```

The run results will be automatically saved to `logs/` directory with timestamp and version information.

## 3. Add Your Own Implementation

Create a `.cu` file **directly under the data-type directory** — `task-1/src/matmul_f16/` for fp16 (or `matmul_f32/` for fp32) — and implement the matmul template specialization with the macro `PLAYGROUND_MATMUL_DEC`.

> ⚠️ **Name the file with the version as `_vN`** (e.g. `matmul_f16_v2.cu` or `matmul_f16_v2_mykernel.cu`). The build selects the source for `--ver N` by that `_vN` token in the filename, so a file without it won't be picked up. The file must sit directly in the dtype dir (the source glob is non-recursive).

For example, add the following in `task-1/src/matmul_f16/matmul_f16_v2_mykernel.cu` to define `matmul<float16_t, 2>`:

```cpp
// @file: task-1/src/matmul_f16/matmul_f16_v2_mykernel.cu

#include "playground/matmul.hpp"

namespace playground {
// Implement matmul with DType=float16_t and Version=2.
// Macro arg order is (DType, Version, m, n, k, A, B, C); A/B/C are device pointers.
PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    // ......
}
}
```

> 💡**Note**:  
> Do not use version `0` — it is cBLAS (CPU), the correctness reference. Use version `1`, `2`, … for your own kernels. Each `(dtype, version)` must be defined exactly once (duplicate definitions of the same version collide at link time).

Now you are able to build a new binary `task1_float16_v2` to with the following command:

### Using build script directly:
```bash
# Build the test binary with DType=float16 and Version=2:
bash ./scripts/build-task1.sh -v2 -f16
# Run the test binary
./build/src/task1_float16_v2
```

### Using task1.sh script (Recommended):
```bash
# Build and run with automatic logging
./task1.sh run --float f16 --ver 2
```

## 4. Profile Your Kernel with Nsight Compute

Use "[scripts/nsight-profile.sh](../scripts/nsight-profile.sh)" to profile an binary which contains **a self-defined cuda kernel**.

⚠️ **The profiled binary must be built with `RelWithDebInfo` or `RD` flag**. 

### Using build script directly:
For example, to build matmul kernel with `DType=float16`, `Version=2` and `RD` flag:

```bash
# `RD` is the same as `RelWithDebInfo`
bash ./scripts/build-task1.sh RD -f16 -v2 
```

Then you can profile the binary with `ncu` with a tool script:

```bash
bash ./scripts/nsight-profile.sh -t build/src/task1_float16_v2
```

### Using task1.sh script (Recommended):
```bash
# Build with debug symbols and profile automatically
./task1.sh profile --float f16 --ver 2
```

A `.ncu-rep` file will be generated in the `logs/profiles/` directory with timestamp. Download it to your local machine and open it with Nsight Compute GUI.

![ncu-example](../docs/imgs/ncu-example.png)

## 5. Reference Numbers

A100 hardware peaks — the **only** targets (there is no library baseline to compare against):

| | CUDA Core (FP32) | Tensor Core (FP16) |
| --- | --- | --- |
| Theory Peak (TFLOPS) | 19.5 | 312 |

> The goal is to approach the hardware peak. `v0` (cBLAS, CPU) exists only for correctness checking and is orders of magnitude too slow to be a performance reference.

## 6. References
See also: [feishu doc: cuda学习资料](https://aicarrier.feishu.cn/wiki/SFdnw61vHi1AfRkeJVecgMjBnrc)

### CUDA Core

- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 2-3

- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 4-5
- [CUDA编程入门及优化](https://zhuanlan.zhihu.com/p/441146275) 1.2 Thread Block Tile: 利用 Shared Memory 减少重复访存

- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 6-6.3 Thread coarsening
- [how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788) MMult_cuda_4 & MMult_cuda_5
- [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370) Naive 实现的分析：到底差在哪里？

- "Programming Massively Parallel Processors  A Hands-on Approach (Fourth Edition)" Chapter 6-6.1 Memory coalescing, 6.2 Hiding memory latency
- [how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788) MMult_cuda_9
- [cuda/MMult_cuda_9.cu](https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_9.cu)
- [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370) 极致的访存优化
- [CUDA编程入门及优化](https://zhuanlan.zhihu.com/p/441146275) 1.3 Warp Tile 与 Thread Tile: 利用寄存器消除 Shared Memory 瓶颈


- [how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788) MMult_cuda_12
- [cuda/MMult_cuda_12.cu](https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_12.cu)
- [CUDA编程入门及优化](https://zhuanlan.zhihu.com/p/441146275) 1.4 Double Buffer: 让 GEMM 流水并行起来

### Tensor Core

- [cuda学习：学习nvcuda::wmma实现高效gemm](https://zhuanlan.zhihu.com/p/353208013) simple version

- [cuda学习：学习nvcuda::wmma实现高效gemm](https://zhuanlan.zhihu.com/p/353208013) sample version with detailed annotations
- [Official sample provided by NVIDIA](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu)

- [Nvidia Tensor Core-CUDA HGEMM优化进阶](https://zhuanlan.zhihu.com/p/639297098/) 4.5 提高L2 Cache命中率
- [一步步优化 GEMM by Tensorcore](https://zhuanlan.zhihu.com/p/638522893) 调整线程块分配到的计算位置(swizzle)


#### Source code:
- [src/wmma/wmma_async_stage3.cu](https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/wmma/wmma_async_stage3.cu) 3 stages pipeline with WMMA API

Asynchronous data copy:
- [ Data Movement and Conversion Instructions: cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) To know the usage of cp.async instructions
- [Performance Guidance for memcpy_async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) To know the usage of asynchronous data copy

#### Multi-buffer with prefetching:
- [Nvidia Tensor Core-CUDA HGEMM优化进阶](https://zhuanlan.zhihu.com/p/639297098) 5 Pipeline优化-5.2 Stage
- [一步步优化 GEMM by Tensorcore](https://zhuanlan.zhihu.com/p/638522893) 使用数据预取(prefetch)

#### Permute to use memory coalescing and avoid bank conflicts:
- [cuda（cutlass）编程之swizzle](https://www.bilibili.com/video/BV1Jb421e7UN/?spm_id_from=333.999.0.0&vd_source=2fe7991a33356057a2e41a2d37f9b7e0) A more detailed video explanation of swizzle based on CUTLASS

### For Further Study

- [基于 CUTE 的 GEMM 优化【1】—— Baseline 实现](https://zhuanlan.zhihu.com/p/695063154)
- [cute系列讲解](https://www.zhihu.com/people/reed-84-49/posts)

