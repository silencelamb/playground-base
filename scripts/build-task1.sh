#!/bin/bash
export CC="gcc"
export CXX="g++"
export NVCC_CCBIN="gcc"


TEST_MATMUL_VERSION=1
TEST_DATA_TYPE="float32"

SOURCE_DIR=./task-1
BUILD_DIR=./build
BUILD_TYPE=Release
CXX_STANDARD=20
CUDA_STANDARD=20
VCPKG_HOME=$VCPKG_HOME

if [ -t 1 ]; then
    STDOUT_IS_TERMINAL=ON
else
    STDOUT_IS_TERMINAL=OFF
fi


# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v*)
            TEST_MATMUL_VERSION="${1#*v}" ;;
        -f32|--float32)
            TEST_DATA_TYPE="float32" ;;
        -f16|--float16)
            TEST_DATA_TYPE="float16" ;;
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        Release|Debug|RelWithDebInfo|RD)
            BUILD_TYPE=${1/RD/RelWithDebInfo} ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --stdcuda=*)
            CUDA_STANDARD="${1#*=}" ;;
        --rm-build-dir)
            rm -rf $BUILD_DIR ;;
        --vcpkg-home|--vcpkg-dir|--vcpkg-root)
            VCPKG_HOME=$2; shift ;;
        *)
            echo "build fatal: Invalid argument '$1'."; exit 1 ;;
    esac
    shift
done

# Fast path: only (re)configure when needed. Re-running CMake's configure step
# on every build costs seconds; skip it when the build dir is already configured
# for the same version / dtype / build-type and go straight to the incremental
# Ninja build. (Ninja still re-runs CMake automatically if a CMakeLists changed.)
# A version change flips MATMUL_VERSION below -> triggers a reconfigure (which is
# also when the selected-source glob in src/CMakeLists.txt is re-evaluated).
need_configure=1
cache="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$cache" && -f "$BUILD_DIR/build.ninja" ]]; then
    cached_ver=$(sed -n 's/^MATMUL_VERSION:[^=]*=//p'   "$cache" | head -1)
    cached_dt=$(sed -n  's/^TEST_DATA_TYPE:[^=]*=//p'   "$cache" | head -1)
    cached_bt=$(sed -n  's/^CMAKE_BUILD_TYPE:[^=]*=//p' "$cache" | head -1)
    if [[ "$cached_ver" == "$TEST_MATMUL_VERSION" \
       && "$cached_dt"  == "$TEST_DATA_TYPE" \
       && "$cached_bt"  == "$BUILD_TYPE" ]]; then
        need_configure=0
    fi
fi

if [[ "$need_configure" == 1 ]]; then
    cmake -S $SOURCE_DIR -B $BUILD_DIR -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" \
        -DMATMUL_VERSION=$TEST_MATMUL_VERSION \
        -DTEST_DATA_TYPE=$TEST_DATA_TYPE \
        -DSTDOUT_IS_TERMINAL=$STDOUT_IS_TERMINAL \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
        -DCMAKE_CUDA_STANDARD=$CUDA_STANDARD
else
    echo "[build] reuse configured build dir (v${TEST_MATMUL_VERSION}/${TEST_DATA_TYPE}/${BUILD_TYPE}) -> ninja"
fi

cmake --build $BUILD_DIR --parallel 12

