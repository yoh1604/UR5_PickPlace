# VLM Sequence Pick and Place

Vision-language pick-and-place pipeline for a UR5 robot. The system uses an
RGB-D scene capture from an Intel RealSense D455, a VLM planner, YOLO-World,
FastSAM, and depth extraction to identify objects and estimate their 3D camera
position. The robot side is intended to be controlled with ROS Noetic and
MoveIt for UR5 motion planning and execution.

## Overview

This repository is organized around a step-by-step pick-and-place workflow:

1. Capture the current RGB-D scene from the D455 camera.
2. Ask the VLM planner to convert a user request into an action plan.
3. Validate the generated action plan.
4. Detect the first target object with YOLO-World.
5. Segment the detected target with FastSAM.
6. Extract the object position from the depth image.
7. Send the target pose to the UR5 control stack.
8. Verify the result with a post-action scene check.

## Robot Stack

The robot-control setup uses:

- ROS Noetic
- MoveIt
- UR5 robot arm
- UR5 MoveIt configuration and planning scene
- Optional direct URScript socket tests for connection checks

MoveIt should handle UR5 motion planning, collision checking, and trajectory
execution. The vision pipeline produces object positions that can be transformed
into the UR5 base frame and used as pick targets.

The included URScript test files are useful for checking basic UR5 network
connectivity before connecting the full ROS/MoveIt control flow:

- `test_ur5_popup.py` sends a popup message to the UR5 teach pendant.
- `test_ur5_small_move.py` sends a small relative TCP motion command.

Update `ROBOT_IP` in these files before running them.

## Main Files

- `run_d455_pipeline.py` runs the main VLM, validation, detection,
  segmentation, and depth pipeline.
- `run_post_pipeline.py` checks the scene after an action and prepares the next
  target if needed.
- `capture_config.py` stores paths, test names, user query text, and output
  locations.
- `vlm_engine.py` calls the VLM planner.
- `validator_engine.py` validates the generated plan.
- `yolo_world_engine.py` runs YOLO-World detection.
- `fastsam_engine.py` runs FastSAM segmentation.
- `depth_engine.py` extracts 3D position from the mask and depth frame.

## Data Layout

The default input files are stored under:

```text
data/d455_capture/
```

Expected input files:

- `current_scene_rgb.jpg`
- `current_scene_depth.png`
- `depth_raw.npy`
- `camera_intrinsics.json`
- `post_scene_rgb.jpg` for post-action checking

Pipeline outputs are saved under:

```text
data/d455_capture/tests/<TEST_NAME>/
```

## Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate pick-place-occlusion
```

Or install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then add at least one API key:

```text
OPENAI_API_KEY=
GEMINI_API_KEY=
```

The current planner configuration prefers Gemini when `GEMINI_API_KEY` is set.
The validator uses OpenAI first when `OPENAI_API_KEY` is available, otherwise it
uses Gemini.

## YOLO-World and FastSAM Setup

YOLO-World and FastSAM model weights are not included in this repository. Keep
the downloaded `.pt` files local and do not push them to git.

Install the environment first:

```bash
conda env create -f environment.yml
conda activate pick-place-occlusion
```

Download the model weights into the project root. The current pipeline expects
these filenames:

- `yolov8l-worldv2.pt`
- `FastSAM-s.pt`

You can let Ultralytics download them automatically by running:

```bash
python3 -c "from ultralytics import YOLOWorld, FastSAM; YOLOWorld('yolov8l-worldv2.pt'); FastSAM('FastSAM-s.pt')"
```

After the download, confirm the files exist:

```bash
ls yolov8l-worldv2.pt FastSAM-s.pt
```

Some test scripts use smaller YOLO-World variants. Download these only if you
need those scripts:

```bash
python3 -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8s-world.pt'); YOLOWorld('yolov8s-worldv2.pt')"
```

## Basic Usage

Edit `capture_config.py` for each experiment:

- `TEST_NAME`
- `USER_QUERY`
- `STEP_INDEX`
- input and output paths if needed

Run the main D455 pipeline:

```bash
python3 run_d455_pipeline.py
```

After the UR5 moves the object, capture a new `post_scene_rgb.jpg`, then run:

```bash
python3 run_post_pipeline.py
```

## UR5 Connection Tests

Send a popup to the robot:

```bash
python3 test_ur5_popup.py
```

Send a small safe test movement:

```bash
python3 test_ur5_small_move.py
```

Use these only when the robot workspace is clear, the emergency stop is ready,
and the UR5 IP address is correct.

## ROS Noetic and MoveIt Notes

Before running the robot-control side, source the ROS Noetic environment and
your catkin workspace:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

Typical UR5 control flow:

1. Bring up the UR5 driver or simulator.
2. Start the UR5 MoveIt planning configuration.
3. Transform the detected object position from the D455 camera frame to the UR5
   base frame.
4. Use MoveIt to plan and execute the pick motion.
5. Capture the post-action image and run the post-check pipeline.

## Safety

Always test motions in simulation or at low speed first. Confirm the camera to
robot transform, object pose, tool center point, payload, collision objects, and
workspace limits before executing motion on the real UR5.
