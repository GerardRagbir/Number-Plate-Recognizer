# Iris reference

The technical detail behind [the README](../README.md): what each part is built on, every setting,
and how tracking and matching work.

It is built directly on the frameworks macOS ships with rather than on a generic CPU/CUDA stack:

| Stage | What does the work |
|---|---|
| Video file decode | **AVFoundation** `AVAssetReader` (VideoToolbox hardware decode) → `CVPixelBuffer` |
| Still images + EXIF | **ImageIO** |
| Screen / window capture | **ScreenCaptureKit** |
| Local cameras | **AVFoundation** capture session |
| Network streams (RTSP/HTTP/HLS) | FFmpeg (PyAV) with the VideoToolbox decoder |
| Vehicle detection + masks | YOLO26-seg as a **Core ML** model, driven by **Vision** (Neural Engine / Metal GPU) |
| Plate detection | ONNX model on ONNX Runtime's **Core ML** execution provider |
| Plate reading (OCR) | **Vision** `VNRecognizeTextRequest` |
| Appearance embedding (re-ID) | **Vision** `VNGenerateImageFeaturePrintRequest` |
| Tile scaling + JPEG encode | **Core Image** on a **Metal** context |

Frames stay in `CVPixelBuffer`s from decode to inference to encode; Python only orchestrates.

## Setup

Requires an Apple-silicon Mac, Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python -m npr setup   # one-off: converts the vehicle model to Core ML, fetches the plate model
uv run python -m npr serve   # → http://127.0.0.1:8314
```

`setup` runs the model conversion in a separate throwaway environment (Python 3.13 + PyTorch +
coremltools, which has no 3.14 build yet), so the app's own environment stays small and pure 3.14.

macOS will ask for **Screen Recording** and **Camera** permission the first time those sources are
used. The permission belongs to the app that launched `npr` (Terminal, VS Code, …).

## Using the dashboard

The interface is layered so that the common decisions are guided and the detail stays out of the way:

1. **Home** — the jobs the app does, as scenario cards: watch cameras live, log plates at a gate,
   review recorded footage, follow a vehicle across cameras, understand movement patterns, and check
   that cameras are connected. Each card says whether it is ready or what it still needs, and its
   button is the next step. A short *Getting set up* checklist sits above them.
2. **Guided steps** — *Add camera* asks one thing at a time: what kind of source, how to connect
   (network addresses are tested before you continue), where it is and which way it faces (click a
   map; can be skipped), then a name and what to track. *Guided setup* asks three questions for the
   general settings: what to look for, when the models should run, and what to do if a camera drops.
3. **Operating views** — **Live** (a wall that arranges itself to the number of cameras, with pages
   past four; masks, boxes, paths, plates and labels toggle from a toolbar at the bottom), **Map** (OpenStreetMap with a marker and field-of-view cone per
   source, measured flows, and the selected vehicle's route) and **Health** (a heartbeat per
   connection, and a step-by-step connection test that says where and why it failed). The camera list
   on the left and the Plates / Vehicles / Patterns panel on the right (**Activity**) stay with you.
4. **Settings** — everything, as conventional lists: a table of cameras with Edit / Test / Start /
   Remove and the full form for the selected one, and a *General* tab with every global option.

Click a vehicle in any tile (or any row in *Plates* / *Vehicles*) to select it and see its route.

Keys: `W` live · `M` map · `N` health · `C` settings · `G` guide · `[` cameras · `]` activity ·
`1`–`9` select tile · `F` maximise · `L` / Page Down next page · Page Up previous page · `Esc` cancel.

### Wall layout

The wall needs no layout choice. A page shows 1, 2, 3 (one large and two small), 4, 6, 9 or 16 tiles
depending on how many cameras are on it. **Settings → Layout** decides how cameras become pages:

- **Automatic** — cameras fill the wall in order; more than *cameras per page* (default 4) spill onto
  further pages.
- **By group** — each camera group gets its own page, named after it.
- **Custom** — you build the pages: a name, a grid, and which camera sits in which tile.

"Show on wall" on a group in the camera list opens that group as a one-off page.

## Camera profiles

`npr/data/cameras.json` is a local database of camera makes and models. Choosing a make and model
in *Add camera* uses it to build the stream address (with the brand's steps for switching the stream
on), to try the likely stream slots when the first one does not answer, and to set the camera type,
lens angle and reach that draw its cone on the map. The headline lens angle makers publish is
normally the diagonal; the cone uses the horizontal angle worked out from it for a 16:9 picture.

Every value records where it came from (`official`, `retailer`, `third-party`, `community`,
`unconfirmed`) with a source link. eufy is the first brand; add brands and models by editing the JSON
file, with no code changes. `GET /api/camera-profiles` serves it.

Buttons throughout are icons that widen to show their name on hover or keyboard focus, with a short
tooltip (name, shortcut, one line).

## Reviewing a recording

Adding recorded video or photos asks how to process them:

- **Process immediately** analyses every frame as fast as the Mac allows (nothing is skipped) and opens
  the **review**: a looping player with what was found drawn on it, and a timeline underneath. While
  analysis runs the timeline is a single strip filling in with objects and plate reads. When it
  finishes it becomes one lane per object type, each object a bar carrying preview images at the
  moments they were taken. Click the timeline to seek.
- **Process temporally** plays the file back in real time on the live wall, placed on the shared
  timeline at the moment it was recorded, so it lines up with cameras and other recordings.

Clicking a preview opens that object's **connections**: linked cards for the frames it was seen in,
how it moved (direction across the picture, towards or away from the camera, its path), where that
puts it on the map if the recording has a position, and any other sighting of the same vehicle. The
map position is approximate: the bearing comes from where the object sits across the picture, and
the distance is only a guess from how low it sits in the frame.

Any recording can be reopened later from its Review action in the camera list or in Settings.

## Searching activity

The Activity drawer is one search bar over one feed of plate reads, vehicles and patterns.

- **Mode** (left of the bar): tick any of Plates, Vehicles, Patterns; none ticked shows everything.
- **Chips**: type a word the search knows and it becomes a chip: a colour (`red`, `maroon`, `silver` …)
  or a vehicle type (`truck`, `motorbike` …). Click a chip to swap it for another value of the same
  kind. Values of one kind are OR-ed, different kinds are AND-ed. Anything left over matches the plate.
- **Filter** (right of the bar): newest or oldest first, and a time range with presets or custom dates.

Colours are coarse families on purpose (red, yellow/orange, green, blue, purple, pink, white,
grey/silver, black): the cameras are uncalibrated, so "red" covers everything red-looking. A vehicle's
family is worked out from the colour signature taken under its mask. The vocabulary lives in
`npr/search.py`; add a facet there and a filter for it in `Store.activity` to teach the search a new
category.

## Settings in detail

**Global** (applies everywhere, saved in the database, takes effect immediately):

- *Objects to track* — untick a type (bicycle, car, motorcycle, bus, truck) to ignore it everywhere.
- *Skip unchanged frames* — see below.
- *Reconnect dropped streams automatically.*
- Default lens angle and range for new cameras.

**Per camera**: name, address, location (latitude/longitude, heading, lens angle, range, mount height
and tilt), type and make/model/notes, and for each object type one of *Global*, *Allow* (whitelist:
tracked here even if blocked globally) or *Block* (blacklist: ignored here). Skipping unchanged frames
can also be forced on or off per camera.

Stream passwords are stored but never sent back to the browser, shown in errors or written to the
log: the address is displayed as `user:***@host`, and leaving `***` in place keeps the stored one.

### Skipping unchanged frames

Each frame is first compared (cheaply, at 320×180) with the last frame that was analysed. If nothing
has changed, the models do not run: vehicles already in view stay tracked, and the tile keeps its
picture. Any change wakes the pipeline, which then analyses every frame for the hold time (2 s) so
tracking and plate reading can settle. A static scene is still re-checked every 10 s. On a quiet
camera this removes almost all of the inference work; the Network view shows the share of frames
actually analysed.

### Streams that drop

A network stream that ends or fails is retried indefinitely (after 2, 4, 8 … up to 30 s). The source
shows *reconnecting* with the reason, and the drop is visible in its heartbeat.

## How vehicles are followed

**Within one source** a ByteTrack-style tracker (Kalman filter + two-stage IoU matching) keeps each
vehicle's identity and path. Plate reads are voted over the life of the track, so one bad frame does
not produce a bad plate.

**Across sources** each track becomes a *sighting*, and sightings are grouped into one vehicle by:

1. plate — fuzzy, so a single mis-read character still matches;
2. appearance — the Vision feature-print plus a colour histogram taken under the vehicle's mask;

both gated by physics: one vehicle cannot be on two separated cameras at once, nor travel between two
placed cameras faster than `NPR_MAX_SPEED` (180 km/h). Placing cameras on the map therefore makes
matching stricter *and* gives distances and speeds.

Recorded files are placed on the same timeline as everything else using their creation time, so
separate recordings of connected locations link up just like live cameras.

### Field of view

The cone's lens angle is taken from the 35 mm-equivalent focal length in EXIF (photos) or QuickTime
metadata (iPhone video), and location/heading from GPS tags when present. Where nothing is recorded —
most CCTV, streams, screen capture, and Mac cameras, which do not report a lens angle — it starts at
an assumed 70° and is labelled as such until you set it.

## Limits worth knowing

- Appearance-only matches (no plate) are a best guess; the general-purpose feature-print is not a
  trained vehicle re-ID model. Matches show how they were made (*plate* or *appearance*) and the
  score; tune with `NPR_APPEARANCE_MATCH` (default 0.80, higher = stricter).
- Plates need roughly 24+ pixels of width to be attempted. Vision's recogniser is general-purpose and
  handles Latin-alphabet plates.
- The server binds to localhost and has no authentication. Put it behind your own access control
  before exposing it to a network.
- Check local law before recording or processing number plates.

## Environment

Environment variables (see `npr/config.py`): `NPR_HOME`, `NPR_VEHICLE_MODEL` (default `yolo26s-seg`;
`yolo26n-seg` for speed … `yolo26m-seg` for accuracy; `yolo11*-seg` also works), `NPR_VEHICLE_CONF`, `NPR_PLATE_CONF`, `NPR_TILE_WIDTH`,
`NPR_MAX_FPS`, `NPR_PLATE_MATCH`, `NPR_APPEARANCE_MATCH`, `NPR_REID_WINDOW`, `NPR_MAX_SPEED`.

Data (SQLite database, uploads, thumbnails) lives in `data/`; models in `models/`.

## Development

```sh
uv run pytest
```

```
npr/
  ingest/     files (AVFoundation, ImageIO), stream (FFmpeg), capture (ScreenCaptureKit, cameras)
  native/     CVPixelBuffer ↔ numpy, Metal-backed JPEG encoder
  vision/     Core ML vehicle segmenter, plate detector, Vision OCR + feature-prints
  tracking/   multi-object tracker
  analysis/   cross-camera re-identification, movement patterns, geodesy
  server/     FastAPI app, websocket hub, pipeline manager
  web/        dashboard (no build step)
  pipeline.py one worker thread per source (motion gate, reconnect, object filters)
  appconfig.py global + per-camera settings · heartbeat.py connection sampling · nettest.py connection test
```

## Licences

The default vehicle model is Ultralytics YOLO26, which is **AGPL-3.0** (or a commercial licence from
Ultralytics). The plate detector comes from [open-image-models](https://github.com/ankandrew/open-image-models)
(MIT). Map data © OpenStreetMap contributors.
