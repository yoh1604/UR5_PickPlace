import json
import numpy as np
import open3d as o3d
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def create_gripper_mesh(width, height, depth, color=[0, 1, 0]):
    """Creates a basic box representing the gripper geometry."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    # Center the box so the grasp point is correctly aligned
    mesh.translate((-width/2, -height/2, 0)) 
    mesh.paint_uniform_color(color)
    return mesh

def main():
    # Define the target test case directory based on your folder structure
    # You can change this to any folder like 'TC_05_GMN_YOLO' etc.
    test_case_dir = 'outputs/TC_04_GMN_VLM' 
    vision_dir = os.path.join(test_case_dir, 'vision_output')
    
    # Corrected file names based on the directory tree
    candidates_file = os.path.join(vision_dir, 'grasp _candidates_camera.json')
    pcd_file = os.path.join(vision_dir, 'masked_object_pointcloud_median_fill.ply')

    # 1. Load the Grasp Data
    if not os.path.exists(candidates_file):
        print(f"Error: {candidates_file} not found. Check your directory path.")
        return

    data = load_json(candidates_file)
    best_grasp = data['candidates'][0]  # The rank 0 grasp
    obj_center = data['object_center_camera']
    
    geometries = []

    # 1b. Load and visualize the actual 3D object Point Cloud
    if os.path.exists(pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        geometries.append(pcd)
    else:
        print(f"Warning: Point cloud {pcd_file} not found. Visualizing grasps only.")

    # 2. Visualize Object Center
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.translate(obj_center)
    sphere.paint_uniform_color([1, 0, 0]) # Red object center
    geometries.append(sphere)

    # 3. Visualize Best Grasp (Coordinate Frame + Gripper Bounding Box)
    best_t = np.array(best_grasp['translation_camera'])
    best_R = np.array(best_grasp['rotation_matrix_camera'])
    
    # Add coordinate frame axes for orientation (RGB = XYZ)
    best_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    best_frame.rotate(best_R, center=(0, 0, 0))
    best_frame.translate(best_t)
    geometries.append(best_frame)

    # Add the best gripper box clearly marked in Green
    best_gripper = create_gripper_mesh(best_grasp['width'], best_grasp['height'], best_grasp['depth'], color=[0, 1, 0])
    best_gripper.rotate(best_R, center=(0,0,0))
    best_gripper.translate(best_t)
    
    best_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(best_gripper)
    best_line_set.paint_uniform_color([0, 1, 0]) # Green
    geometries.append(best_line_set)

    # 4. Plot a sample of other candidates
    # Plotting all candidates will clutter the screen, so we plot every 20th candidate
    for cand in data['candidates'][1::20]:
        t = np.array(cand['translation_camera'])
        R = np.array(cand['rotation_matrix_camera'])
        
        # Represent other candidates with a faint blue bounding box
        gripper = create_gripper_mesh(cand['width'], cand['height'], cand['depth'], color=[0, 0, 1])
        gripper.rotate(R, center=(0,0,0))
        gripper.translate(t)
        
        # Create a line set to make it transparent/wireframe so the point cloud remains visible
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(gripper)
        line_set.paint_uniform_color([0.2, 0.2, 0.8])
        geometries.append(line_set)

    # 5. Render Scene
    print("Visualizing: Red = Object Center | Green Axes/Box = Best Grasp | Blue Boxes = Candidates")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()