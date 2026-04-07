# ARIES Fire Detection Demo Package

This package contains the ARIES multi-channel fire detection demo for MLA100 and MLA400 systems.

## Requirements

- Ubuntu 20.04 or later recommended
- `libopencv-dev`
- Mobilint QB Runtime
- ARIES driver

## Build

From the package root:

```bash
make -j"$(nproc)"
cd build
./demo
```

The executable expects to run from the `build` directory so it can resolve the bundled `../rc/...` configuration paths.

## Runtime Modes

The demo supports two built-in mode sets:

- `MLA100`
  - 30 tiled video feeders
  - `rc/LayoutSetting_MLA100.yaml`
  - `rc/ModelSetting_MLA100.yaml`
  - `rc/FeederSetting_MLA100.yaml`
- `MLA400`
  - 96 tiled video feeders
  - `rc/LayoutSetting_MLA400.yaml`
  - `rc/ModelSetting_MLA400.yaml`
  - `rc/FeederSetting_MLA400.yaml`

The application starts in `MLA100` mode by default.

## Feeder Setting

Each feeder entry uses a `feeder_type` plus an array-form `src_path`.

Example:

```yaml
- feeder_type: VIDEO
  src_path: [../rc/video/12.mp4]
```

Supported feeder types:

- `CAMERA`
- `VIDEO`
- `IPCAMERA`

Notes:

- `VIDEO` inputs loop automatically.

## Model Setting

Example:

```yaml
- model_type: OBJECT
  mxq_path: ../mxq/yolo11s-fire-final.mxq
  dev_no: 0
  num_core: 8
```

Supported model types in the YAML parser:

- `OBJECT`

Supported model fields:

- `mxq_path`
- `dev_no`
- `num_core`
- `core_id`
- `subconfig.input`

Notes:

- `MLA100` uses `../mxq/yolo11s-fire-final.mxq`.
- `MLA400` uses `../mxq/yolo11s-fire-final.mxq` with `subconfig.input: uint8`.
- If `num_core` is omitted, the parser can derive the worker count from `core_id`.

## Controls

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

Mouse controls:

- Left click: start the selected worker
- Right click: stop the selected worker

## Notes

- The package includes `mxq/yolo11s-fire-final.mxq`.
