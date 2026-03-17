#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
   echo "Usage: $0 <PRODUCT> <DRIVER_TYPE>" >&2
   exit 1
fi
readonly PRODUCT="$1"
readonly DRIVER_TYPE="$2"
readonly BUILD_DIR=build_package
readonly PACKAGE_DIR="demo-package-${PRODUCT}"

readonly QBRUNTIME_INCLUDE_DIR="${QBRUNTIME_INCLUDE_DIR:-/usr/include/qbruntime}"
readonly QBRUNTIME_LIBRARY_DIR="${QBRUNTIME_LIBRARY_DIR:-/usr/lib/x86_64-linux-gnu}"
readonly YAML_CPP_CACHE_DIR="${YAML_CPP_CACHE_DIR:-build/_deps}"

cd "$(git rev-parse --show-toplevel)"

rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
mkdir "$BUILD_DIR"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFIRE_DETECTION_VENDOR_YAML_CPP=ON \
  -DPRODUCT="${PRODUCT}" \
  -DDRIVER_TYPE="${DRIVER_TYPE}"

cmake --build "$BUILD_DIR" --target demo -j"$(nproc)"

if [[ ! -d "$QBRUNTIME_INCLUDE_DIR" ]]; then
  echo "qbruntime headers not found at $QBRUNTIME_INCLUDE_DIR" >&2
  exit 1
fi

if ! compgen -G "$QBRUNTIME_LIBRARY_DIR/libqbruntime.so*" > /dev/null; then
  echo "qbruntime libraries not found at $QBRUNTIME_LIBRARY_DIR" >&2
  exit 1
fi

mkdir -p "$PACKAGE_DIR/yaml-cpp/lib"
mkdir -p "$PACKAGE_DIR/qbruntime/lib"

cp -r mxq rc src package/Makefile package/README.md "$PACKAGE_DIR"

if [[ -d "$BUILD_DIR/_deps/yaml-cpp-src" && -f "$BUILD_DIR/_deps/yaml-cpp-build/libyaml-cpp.a" ]]; then
  cp -r "$BUILD_DIR/_deps/yaml-cpp-src/include" "$PACKAGE_DIR/yaml-cpp"
  cp -r "$BUILD_DIR/_deps/yaml-cpp-build/libyaml-cpp.a" "$PACKAGE_DIR/yaml-cpp/lib"
elif [[ -d "$YAML_CPP_CACHE_DIR/yaml-cpp-src" && -f "$YAML_CPP_CACHE_DIR/yaml-cpp-build/libyaml-cpp.a" ]]; then
  cp -r "$YAML_CPP_CACHE_DIR/yaml-cpp-src/include" "$PACKAGE_DIR/yaml-cpp"
  cp -r "$YAML_CPP_CACHE_DIR/yaml-cpp-build/libyaml-cpp.a" "$PACKAGE_DIR/yaml-cpp/lib"
else
  echo "yaml-cpp cache not found in $BUILD_DIR/_deps or $YAML_CPP_CACHE_DIR" >&2
  exit 1
fi

cp -r "$QBRUNTIME_INCLUDE_DIR" "$PACKAGE_DIR/qbruntime/include"
cp -r "$QBRUNTIME_LIBRARY_DIR"/libqbruntime.so* "$PACKAGE_DIR/qbruntime/lib"

tar -czvf "$PACKAGE_DIR.tar.gz" "$PACKAGE_DIR"
rm -rf "$PACKAGE_DIR" "$BUILD_DIR"
