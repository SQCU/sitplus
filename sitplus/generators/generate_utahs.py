# sitplus.generators.generate_utahs
import bpy
import json
import os
import random
import math
import sys
from mathutils import Matrix, Vector
import numpy as np

# Add path for IPC protocol
sys.path.append(os.path.abspath("."))
from sitplus.utils.ipc_protocol import pack_log, pack_error, _pack_teapot_data_ipc 

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
STL_PATH = os.path.join(project_root, "sitplus/generators/Utah_teapot_(solid).stl")
STL_PATH = os.path.normpath(STL_PATH)

# Scene Parameter Ranges
TEAPOT_ROTATION_RANGE = (-180, 180)
TEAPOT_SCALE_RANGE = (0.8, 1.2)
TEAPOT_STRETCH_RANGE = (0.7, 1.3)
TEAPOT_SHEAR_RANGE = (-0.5, 0.5)
TEAPOT_FRAME_RATIO_RANGE = (1/9, 12/7)
LIGHT_DISTANCE_RANGE = (5, 30)
LIGHT_BASE_STRENGTH = 1000
LIGHT_STRENGTH_GAMMA = 2.0

# --- UTILITY FUNCTIONS ---

def log(message: str, is_error: bool = False):
    """Sends a log or error message as a packet to stdout."""
    packet = pack_error(message) if is_error else pack_log(message)
    sys.stdout.buffer.write(packet)
    sys.stdout.buffer.flush()

def clear_scene():
    """Deletes all objects in the current scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def look_at(obj, target_vec):
    """Rotates an object to point towards a target vector."""
    direction = target_vec - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def create_shear_matrix(factor):
    """Creates a 4x4 shear matrix."""
    return Matrix((
        (1, 0, factor, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1)
    ))

def setup_scene(stl_filepath):
    """Clears the scene and sets up the core objects."""
    clear_scene()
    
    if not os.path.exists(stl_filepath):
        raise FileNotFoundError(f"STL file not found at: {stl_filepath}")
    bpy.ops.import_mesh.stl(filepath=stl_filepath)
    teapot = bpy.context.selected_objects[0]
    teapot.name = "Teapot"
    
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    teapot.location = (0, 0, 0)
    
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    
    bpy.ops.object.light_add(type='POINT')
    light = bpy.context.object
    light.name = "Light"
    
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)
        
    render_layers_node = tree.nodes.new('CompositorNodeRLayers')
    viewer_node = tree.nodes.new('CompositorNodeViewer')
    tree.links.new(render_layers_node.outputs['Image'], viewer_node.inputs['Image'])

    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.image_settings.file_format = 'PNG'

    return teapot, camera, light

def generate_and_apply_parameters(teapot, camera, light):
    """Generates random parameters and applies them to the scene objects."""
    params = {}
    
    rot_x = math.radians(random.uniform(*TEAPOT_ROTATION_RANGE))
    rot_y = math.radians(random.uniform(*TEAPOT_ROTATION_RANGE))
    rot_z = math.radians(random.uniform(*TEAPOT_ROTATION_RANGE))
    
    scale_u = random.uniform(*TEAPOT_SCALE_RANGE)
    stretch_x = random.uniform(*TEAPOT_STRETCH_RANGE)
    stretch_y = random.uniform(*TEAPOT_STRETCH_RANGE)
    stretch_z = random.uniform(*TEAPOT_STRETCH_RANGE)
    
    shear = random.uniform(*TEAPOT_SHEAR_RANGE)
    
    params['teapot'] = {
        'rotation_euler': (rot_x, rot_y, rot_z),
        'scale': (scale_u * stretch_x, scale_u * stretch_y, scale_u * stretch_z),
        'shear': shear
    }
    
    teapot.rotation_euler = params['teapot']['rotation_euler']
    teapot.scale = params['teapot']['scale']
    
    shear_mat = create_shear_matrix(shear)
    teapot.matrix_world @= shear_mat

    light_dist = random.uniform(*LIGHT_DISTANCE_RANGE)
    light_theta = math.radians(random.uniform(0, 360))
    light_phi = math.radians(random.uniform(0, 180))
    
    light_x = light_dist * math.sin(light_phi) * math.cos(light_theta)
    light_y = light_dist * math.sin(light_phi) * math.sin(light_theta)
    light_z = light_dist * math.cos(light_phi)
    
    light_strength = LIGHT_BASE_STRENGTH / (light_dist**LIGHT_STRENGTH_GAMMA)
    
    params['light'] = {
        'location': (light_x, light_y, light_z),
        'energy': light_strength
    }
    
    light.location = params['light']['location']
    light.data.energy = params['light']['energy']

    teapot_height = teapot.dimensions.z
    frame_ratio = random.uniform(*TEAPOT_FRAME_RATIO_RANGE)
    fov = camera.data.angle
    cam_dist = (teapot_height / frame_ratio) / (2 * math.tan(fov / 2))
    
    cam_theta = math.radians(random.uniform(0, 360))
    cam_phi = math.radians(random.uniform(70, 110))
    
    cam_x = cam_dist * math.sin(cam_phi) * math.cos(cam_theta)
    cam_y = cam_dist * math.sin(cam_phi) * math.sin(cam_theta)
    cam_z = cam_dist * math.cos(cam_phi)
    
    params['camera'] = {
        'location': (cam_x, cam_y, cam_z),
        'distance': cam_dist,
        'frame_ratio': frame_ratio
    }
    
    camera.location = params['camera']['location']
    look_at(camera, Vector((0, 0, 0)))
    
    return params

def render_to_buffer(scene):
    """Renders the scene and returns the pixel data as a NumPy array."""
    bpy.ops.render.render(write_still=False)

    viewer_image = bpy.data.images['Viewer Node']
    width, height = viewer_image.size
    
    if width == 0 or height == 0 or not viewer_image.pixels:
        return np.array([], dtype=np.float32)

    pixels = np.array(viewer_image.pixels[:])
    image = pixels.reshape(height, width, 4)[::-1, :, :]
    
    return image

def render_and_get_params(command_data, teapot, camera, light):
    """Renders an image with the given parameters."""
    res_x = command_data.get("resolution_x", 256)
    res_y = command_data.get("resolution_y", 256)
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y

    teapot.matrix_world = Matrix.Identity(4)
    params = generate_and_apply_parameters(teapot, camera, light)
    image_buffer = render_to_buffer(bpy.context.scene)
    return image_buffer, params

def main_worker_loop():
    """Synchronous worker loop: reads commands from stdin, writes packets to stdout."""
    try:
        log("Blender Worker: Setting up scene...")
        teapot, camera, light = setup_scene(STL_PATH)
        log("Blender Worker: Setup complete. Awaiting commands on stdin.")
    except Exception as e:
        log(f"Blender Worker: FATAL ERROR during setup: {e}", is_error=True)
        return
    
    # Force stdout to flush immediately
    sys.stdout.flush()
    sys.stdout.buffer.flush()
    
    log("Blender Worker: Entering command loop...")
    
    # Read commands line by line from stdin
    while True:
        try:
            log("Blender Worker: Waiting for next command...")
            line = sys.stdin.readline()
            
            if not line:
                log("Blender Worker: stdin closed, exiting.")
                break
            
            log(f"Blender Worker: Received command: {line.strip()}")
                
            command_data = json.loads(line)
            command = command_data.get("command")

            if command == "render":
                log("Blender Worker: Starting render...")
                image_buffer, params = render_and_get_params(command_data, teapot, camera, light)
                
                if image_buffer.size == 0:
                    log("Render failed", is_error=True)
                    continue

                log(f"Blender Worker: Render complete, sending {image_buffer.nbytes} bytes")
                
                # Transform: (params dict, numpy array) → (params dict, raw bytes) → IPC packet
                # Uses ipc_protocol._pack_teapot_data_ipc() - the ONLY place this format is defined
                image_bytes = image_buffer.astype(np.float32).tobytes()
                packet = _pack_teapot_data_ipc(params, image_bytes)
                sys.stdout.buffer.write(packet)
                sys.stdout.buffer.flush()
                
                log("Blender Worker: Data packet sent")

            elif command == "quit":
                log("Quit command received.")
                break
                
        except json.JSONDecodeError as e:
            log(f"JSON decode error: {e}", is_error=True)
        except Exception as e:
            log(f"ERROR in handler: {e}", is_error=True)

if __name__ == "__main__":
    main_worker_loop()