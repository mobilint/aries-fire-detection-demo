# ARIES Fire Detection Demo

This project is a multi-channel fire detection demo for ARIES systems. It displays multiple video feeders in a tiled layout and runs object detection with Mobilint QB Runtime on MLA100 or MLA400 devices.

## Overview

- Executable: `build/src/demo/demo`
- Default startup mode: `MLA100`
- Supported runtime modes:
  - `MLA100`: 30 tiled video feeders, 1 device
  - `MLA400`: 96 tiled video feeders, 4 devices
- Default model type: `OBJECT`
- Bundled model file: `mxq/yolo11s-fire-final.mxq`
- Sample inputs: `rc/video/*.mp4`

## Requirements

### Linux

- Ubuntu 20.04 or later recommended
- `cmake`
- `make` or `build-essential`
- `libopencv-dev`
- Mobilint QB Runtime and ARIES driver

The root `CMakeLists.txt` depends on `OpenCV`, `OpenMP`, `yaml-cpp`, and `qbruntime`. On Linux, `qbruntime` is expected to be installed on the system. `yaml-cpp` is used from a system package when available and otherwise fetched from GitHub during configuration.

### Windows

- CMake
- OpenCV 4.x
- Mobilint QB Runtime

Windows builds should be generated from the root `CMakeLists.txt`.

## Quick Start

If the project is already built:

```bash
./run.sh
```

`run.sh` moves to `$HOME/aries-fire-detection-demo/build` and launches `src/demo/demo`.

If you need to build manually:

```bash
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
./src/demo/demo
```

## update.sh

`./update.sh` performs all of the following:

- installs build dependencies such as `build-essential`, `cmake`, and `libopencv-dev`
- adds Mobilint's APT repository and GPG key
- installs `mobilint-aries-driver`, `mobilint-qb-runtime`, and `mobilint-cli`
- runs `git pull`
- configures and builds the project
- updates the desktop shortcut and icon

This means it is not just a build script. It also modifies the system package configuration and desktop integration on the target machine.

## Configuration Files

The application switches between two built-in mode sets:

- `MLA100`
  - `rc/LayoutSetting_MLA100.yaml`
  - `rc/ModelSetting_MLA100.yaml`
  - `rc/FeederSetting_MLA100.yaml`
- `MLA400`
  - `rc/LayoutSetting_MLA400.yaml`
  - `rc/ModelSetting_MLA400.yaml`
  - `rc/FeederSetting_MLA400.yaml`

At startup, the demo loads the `MLA100` set by default. These paths are resolved relative to the executable, so the program should normally be launched from the `build` directory.

### 1. ModelSetting

`MLA100` uses one fire detection model:

```yaml
- model_type: OBJECT
  mxq_path: ../mxq/yolo11s-fire-final.mxq
  dev_no: 0
  num_core: 8
```

`MLA400` is configured to use four model entries, one per device, and enables uint8 input:

```yaml
- model_type: OBJECT
  mxq_path: ../mxq/yolo11s-fire-final.mxq
  subconfig:
    input: uint8
  dev_no: 0
  num_core: 8
```

- Supported model types in the YAML parser: `OBJECT`
- `mxq_path`: path to the model file
- `dev_no`: ARIES device index
- `num_core`: number of inference threads for that model
- `subconfig.input`: optional input type, currently `uint8` or `float32`
- `core_id`: also supported by the parser as an alternative to `num_core`

If neither `num_core` nor `core_id` is configured, the parser falls back to one core.

### 2. FeederSetting

The feeder list is a YAML array of input sources. Each entry uses `feeder_type` and an array-form `src_path`.

Example:

```yaml
- feeder_type: VIDEO
  src_path: [../rc/video/12.mp4]
```

Supported feeder types:

- `CAMERA`: local camera index, such as `["0"]`
- `IPCAMERA`: RTSP or network stream URL
- `VIDEO`: local video file path

`MLA100` defines 30 `VIDEO` feeders. `MLA400` defines 96 `VIDEO` feeders.

### 3. LayoutSetting

The layout contains one background image plus a `worker_layout` that maps each tile to a feeder and model index.

Example:

```yaml
image_layout:
  - path: ../rc/layout/layout_MLA100.png
    roi: [0, 0, 1920, 1080]

worker_layout:
  - {feeder_index: 0, model_index: 0, roi: [0, 137, 320, 184]}
```

- `image_layout`: background images and their display regions
- `worker_layout`: maps each tile to a `feeder_index` and `model_index`
If a `worker_layout` entry points to an out-of-range feeder or model index, that worker is marked invalid and skipped.

## Controls

### Keyboard

| Key | Action |
| --- | --- |
| `D` | Toggle average FPS overlay |
| `T` | Toggle elapsed time overlay |
| `M` | Toggle fullscreen/window mode |
| `C` | Stop all workers |
| `F` | Start all workers |
| `1` | Switch to `MLA100` mode |
| `2` | Switch to `MLA400` mode |
| `Q`, `Esc` | Quit |

Notes:

- The demo starts in `MLA100` mode.
- Only `1` and `2` are handled in the current codebase.

### Mouse

- Left click: start the selected worker
- Right click: stop the selected worker

## Packaging

To create a deployment package:

```bash
./package/package.sh aries2-v4 aries2
```

The script vendors `yaml-cpp`, copies the installed `qbruntime` headers and shared library, and generates `demo-package-<PRODUCT>.tar.gz`.

## Repository Layout

```text
.
├── mxq/                    # MXQ fire detection models
├── rc/                     # YAML configs, layout images, sample videos
├── src/demo/               # demo application source code
├── package/                # packaging script and packaged README
├── run.sh
├── update.sh
└── CMakeLists.txt          # root build entrypoint
```
