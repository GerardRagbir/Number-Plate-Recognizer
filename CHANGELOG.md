# Changelog

## 1.0.0 — Iris (November 2026)

A complete rewrite, and a new name. Nothing from 0.1 was carried over except the idea of reading
number plates. I rebuilt it after moving from Windows to macOS, to run on what Apple silicon ships
with instead of a CUDA setup.

### Changed

- **Renamed** from "Number Plate Recognizer" to **Iris**. The Python package and command are still
  `npr` for now.
- **Platform:** now an Apple-silicon Mac application on Python 3.14, managed with uv. 0.1 was a
  cross-platform script with an optional CUDA switch and a Docker image.
- **How plates are read:** vehicles are found first, then the plate on each vehicle, then the text is
  read with Apple's Vision framework and voted on across frames. 0.1 ran EasyOCR over the whole
  picture once and returned whatever text it found.
- **What you get back:** a live dashboard, a database of sightings, and a search. 0.1 returned one
  line of text per uploaded picture.

### Added

**Sources**

- Network cameras over RTSP, HTTP, HLS and RTMP, with hardware decoding.
- Recorded video files and photos, including location and lens details from their metadata.
- This Mac's cameras, and any screen or window.
- Automatic reconnection when a stream drops, backing off to once every 30 seconds.

**Recognition and tracking**

- Vehicle detection with an outline (mask) for each vehicle, using YOLO26 on Core ML.
- Tracking of each vehicle through the picture, with its path.
- Recognising the same vehicle on another camera or in another recording, by plate (tolerating one
  misread character) or by appearance, ruled out when the timing or distance is impossible.
- Movement patterns: flows between cameras with travel time and speed, common routes, repeat
  vehicles, activity by hour.
- A colour family for every vehicle.

**Dashboard**

- A live wall that arranges itself to the number of cameras, with pages past four cameras, a page
  per camera group, or pages laid out by hand in Settings.
- A map with a marker and view cone per camera, measured flows, and a selected vehicle's route.
- An activity feed with one search bar: choose what to show, type colours and vehicle types as
  chips, filter by time, sort either way.
- A full-screen review for recordings: looping player, a timeline that builds during analysis, a
  lane per object type with preview images, and a connections view of linked cards for any object
  (frames, motion, map position, other sightings).
- Connection health: a heartbeat per camera and a six-step connection test that explains failures.
- Guided steps for adding a camera and for the general settings, and a Guide page listing what the
  app can do and what each job still needs.
- Camera list and activity feed as drawers that slide out of the way, both resizable.
- Icon buttons that show their name on hover, with short tooltips.

**Configuration**

- General settings, per-camera settings, and camera groups with their own rules. The most specific
  one wins: camera, then group, then general.
- Allow or block each object type everywhere, per group, or per camera.
- A local camera profile database (eufy first) that builds stream addresses, tries likely stream
  slots, and sets the lens angle and reach for the map.
- Stream passwords are stored but never sent to the browser, shown in errors, or written to logs.

**Efficiency**

- Frames in which nothing has changed are not analysed, so a still scene costs almost nothing.
- Video stays in the Mac's native frame format from decode through inference to encoding.

### Removed

- The single `/upload` endpoint that took one image and returned its text.
- EasyOCR, pandas and the CUDA switch.
- The Dockerfile and docker-compose file: Iris depends on macOS frameworks and does not run in a
  Linux container.

## 0.1.0 — Number Plate Recognizer (December 2022)

The original demo, written for UTT Mechatronics students to use with their robot.

- A FastAPI server with one endpoint: post an image, get back the text EasyOCR found in it.
- Optional CUDA acceleration.
- A Dockerfile and docker-compose file for running it in a container.
- Two sample images.
