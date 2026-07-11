# Aakha

Aakha is a real-time assistive navigation app for blind and low-vision users. It uses a phone camera or a local webcam to analyze the scene, then turns the results into spoken guidance and a live dashboard view. The goal of the app is to help a user understand what is in front of them quickly and safely, using computer vision, OCR, speech recognition, and text-to-speech without inventing features that do not exist yet.

The app is built around one shared event bus. Vision and other background tasks publish events to that bus, and a single audio consumer thread speaks them in priority order. A web server can receive camera frames and voice clips from a phone, expose a fullscreen mobile control page, and show a live dashboard plus annotated video stream for a sighted helper.

## What the app currently does

Aakha can:

1. Detect objects in a live camera feed using YOLO11n.
2. Estimate whether detected objects are to the left, ahead, or right of the user.
3. Detect when a tracked object is getting closer and raise a collision warning.
4. Detect crosswalk-like zebra stripes with classic computer vision.
5. Classify traffic-light color from a detected traffic-light box.
6. Estimate the clearest walking direction and path drift.
7. Read text from the current frame with OCR when triggered.
8. Generate a scene caption from the current frame on demand or after a scene change.
9. Handle short voice commands such as "what am I holding", "read this", "describe the scene", and "repeat that".
10. Speak all alerts through a single TTS consumer thread.
11. Stream phone camera frames and voice clips to a laptop backend over WebSockets.
12. Show a dashboard with event history and thread heartbeats.

## How it works

The system is divided into a few cooperating parts:

- `main.py` starts the worker threads and stores the latest frame, detections, and annotated frame in shared memory.
- `vision/detect.py` runs the Tier 1 vision loop. It performs YOLO inference, optional ByteTrack tracking, collision monitoring, crosswalk detection, traffic-light classification, and path guidance. It publishes the resulting messages to the shared event bus.
- `audio/consumer.py` runs exactly one text-to-speech consumer thread. It reads events from the priority queue and speaks them with `pyttsx3`.
- `visuals/ocr.py` waits for a keypress, then OCRs the latest frame with Tesseract and publishes the result.
- `visuals/scene_caption.py` watches for scene changes, loads Moondream2, captions the frame, and publishes the caption.
- `audio/voice_trigger.py` transcribes short audio clips with Vosk, matches them against the supported commands, and triggers the right action.
- `server.py` provides the web app, WebSocket endpoints, dashboard, live monitor, and mobile camera/audio transport.
- `config.py` stores live settings such as `crosswalk_detection`, `traffic_light_detection`, `voice_trigger`, and the target FPS.

The shared event bus is the core contract. Producers publish `Event` objects, and the audio consumer is the only component that removes them from the queue. This keeps speech ordered by priority and avoids overlapping speech from multiple threads.

## Implemented functions and entry points

These are the concrete functions the app uses today.

### `main.py`

- `get_latest_frame()` returns the most recent BGR frame.
- `set_latest_frame(frame)` stores the newest processed frame.
- `get_latest_detections()` returns the latest YOLO detection list.
- `set_latest_detections(detections)` stores the latest detection list.
- `push_frame(jpeg_or_array)` accepts a JPEG byte stream or a numpy frame from the server.
- `get_incoming_frame()` returns the latest frame pushed by the phone client.
- `set_annotated_frame(frame)` stores the latest frame with overlays.
- `get_annotated_frame()` returns the annotated frame for the live monitor.
- `_vision_producer()` starts the Tier 1 vision loop and keeps the thread alive.
- `start_workers()` launches the worker threads.
- `run(block=True)` starts the app workers and optionally blocks.

### `vision/detect.py`

- `ensure_onnx_model()` exports or locates the YOLO ONNX model.
- `zone_for(center_x, width)` maps a detection to left, ahead, or right.
- `phrase_for(name, zone)` builds the spoken obstacle phrase.
- `detections_from(results, width, class_ids=None)` converts YOLO output into app detections.
- `pick_nearest(dets)` selects the closest object by box area.
- `print_dets(dets)` prints detections to the console.
- `announce_directional(dets, publish=True, debounce=None)` speaks the nearest obstacle direction.
- `announce_collisions(dets, frame_area, monitor, publish=True, now=...)` raises approach warnings.
- `announce_traffic_lights(frame, dets, monitor, publish=True, now=...)` publishes traffic-light state.
- `_build_candidates(...)` builds guidance candidates for the arbiter.
- `process_frame(frame, model, ...)` runs one full vision pass.
- `draw_overlay(frame, dets, ...)` draws labels and guidance on the frame.
- `_show(frame)` handles preview display.
- `vision_loop(source=0, ...)` runs the live camera loop.
- `run_on_image(path, ...)` runs the pipeline on a still image.
- `main()` provides the module CLI entry point.

The vision loop uses these supporting modules:

- `vision/collision.py` with `CollisionMonitor.update()` to detect looming objects.
- `vision/crosswalk.py` with `CrosswalkDetector.analyze()` and `CrosswalkDetector.update()` to detect zebra crossings.
- `vision/path.py` with `clearest_path()`, `path_drift()`, and `annotate_path()` for route guidance.
- `vision/guidance.py` with `display_name()`, `Corridor.contains()`, `Corridor.polygon()`, `Candidate.to_event()`, and `GuidanceArbiter.select()` to prioritize what gets spoken.
- `vision/traffic_light.py` with `classify_light()` and `TrafficLightMonitor.update()` to classify traffic lights.

### `audio/consumer.py`

- `get_last_spoken()` returns the last spoken message.
- `_set_last_spoken(msg)` stores the latest spoken message.
- `_consumer_loop()` drains the event bus and speaks events.
- `start_consumer()` starts the single TTS consumer thread.

### `audio/voice_trigger.py`

- `_load_vosk_model()` loads the offline Vosk model.
- `transcribe_clip(audio_bytes, model=None)` turns a short PCM clip into text.
- `_match_command(transcript)` maps text to a supported command.
- `dispatch_command(transcript, get_frame, get_detections=None)` runs the action for a command.
- `_voice_trigger_thread(...)` waits for audio clips and processes them.
- `start_voice_trigger_thread(...)` starts the voice-command thread.

The supported commands are implemented now, not planned:

- `what am I holding` uses MediaPipe Hands plus the latest YOLO detections.
- `read this` runs OCR on the current frame.
- `describe the scene` requests an on-demand scene caption.
- `repeat that` repeats the last spoken TTS message.

### `visuals/ocr.py`

- `_preprocess(frame)` prepares the image for OCR.
- `_run_ocr(frame)` runs Tesseract on the current frame.
- `_ocr_thread(get_frame)` waits for Enter and publishes OCR text.
- `start_ocr_thread(get_frame)` starts the OCR worker thread.

### `visuals/scene_caption.py`

- `_scene_changed(current, reference)` checks whether the scene has changed enough to warrant a caption.
- `_load_moondream()` loads the Moondream2 captioning model.
- `_run_inference(model, bgr_frame)` generates a short caption.
- `_caption_thread(get_frame)` runs the background caption loop.
- `start_scene_caption_thread(get_frame)` starts the caption worker thread.

### `server.py`

- `_default_on_frame(frame)` counts incoming frames until the real vision handler is wired in.
- `set_frame_callback(fn)` points `/camera` at the frame handler.
- `_event_to_payload(event)` converts events into dashboard payloads.
- `_install_bus_tap()` mirrors published events to websocket observers.
- `_known_thread_heartbeats()` builds the thread-status snapshot.
- `_now()` returns the current time.
- `_heartbeat_loop()` broadcasts thread heartbeats.
- `_on_startup()` starts config watching and, if enabled, the worker threads.
- `_on_shutdown()` stops the heartbeat loop.
- `camera_ws()` receives JPEG frames from the browser or phone.
- `control_ws()` handles start/stop and runtime config changes.
- `audio_ws()` receives short voice-command clips.
- `status_ws()` streams events and heartbeats to the dashboard.
- `index()` serves the fullscreen mobile control page.
- `dashboard()` serves the helper dashboard.
- `manifest()`, `service_worker()`, `pcm_worklet()`, `icon_192()`, and `icon_512()` serve app assets.
- `monitor()` serves the live detection monitor page.
- `stream_mjpg()` streams the annotated camera feed as MJPEG.
- `run_server(host="0.0.0.0", port=8000)` launches Uvicorn.

### `config.py`

- `Config.to_dict()` returns the persisted settings.
- `Config._coerce(name, value)` converts settings into the right type.
- `Config._apply(data)` applies updates in place.
- `Config.reload(path=CONFIG_PATH)` reloads `config.json`.
- `Config.update(path=CONFIG_PATH, **kwargs)` updates settings and saves them.
- `Config.save(path=CONFIG_PATH)` writes the current settings to disk.
- `Config.is_enabled(feature)` checks whether a feature toggle is on.
- `Config.cut_next()` disables the next optional feature in the cut order.
- `Config.start_watching(interval=2.0, path=CONFIG_PATH)` starts the file watcher.
- `Config.stop_watching()` stops the watcher.
- `_mtime(path)` reads a file modification time.

### `shared`

- `shared.bus.py` provides the shared priority queue event bus.
- `shared.events.py` defines the `Event` and `Priority` types used across the app.
- `shared/smoke_test.py` has `run()` as the compatibility check for starting `main.run()` and confirming events flow.

## In one sentence

Aakha is a threaded assistive vision system that takes live camera input, detects obstacles and path cues, lets the user request OCR or spoken descriptions on demand, and speaks the results through a single prioritized audio channel.
