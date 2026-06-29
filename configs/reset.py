import pyrealsense2 as rs
import time

ctx = rs.context()
devices = ctx.query_devices()

# 1. Force a hardware reset before starting the pipeline
if devices:
    print("Resetting RealSense hardware...")
    devices[0].hardware_reset()
    time.sleep(3) # Give the camera a few seconds to reconnect to the USB bus

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        try:
            # 2. Use a Try/Except block for the wait_for_frames call
            frames = pipeline.wait_for_frames(5000)
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
                
            # Process your frames here...
            
        except RuntimeError as e:
            print(f"Frame timeout: {e}. Retrying...")
            continue # Skip this loop iteration and try grabbing the next frame
            
finally:
    pipeline.stop()