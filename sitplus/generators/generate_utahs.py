"""
# sitplus.generators.generate_utahs
"""
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
LIGHT_BASE_STRENGTH = 10000    #post-multiplied for cycles
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

    # Give the teapot a basic material so it's actually visible in Cycles
    mat = bpy.data.materials.new(name="TeapotMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # Light grey
        bsdf.inputs['Metallic'].default_value = 0.3
        bsdf.inputs['Roughness'].default_value = 0.4

    if teapot.data.materials:
        teapot.data.materials[0] = mat
    else:
        teapot.data.materials.append(mat)

    log("Material assigned to teapot")
    
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    
    # Create 5-point lighting rig
    lights = []
    for i in range(5):
        bpy.ops.object.light_add(type='POINT')
        light = bpy.context.object
        light.name = f"Light_{i}"
        lights.append(light)

    # Light 0 is the key light (strongest)
    lights[0].data.energy = LIGHT_BASE_STRENGTH
    lights[0].name = "KeyLight"

    # Lights 1-4 are fill lights (weaker, form a tetrahedral arrangement)
    for i in range(1, 5):
        lights[i].data.energy = LIGHT_BASE_STRENGTH * 0.3  # 30% of key light
        lights[i].name = f"FillLight_{i}"

    log(f"Created 5-point lighting rig")
    
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)
        
    render_layers_node = tree.nodes.new('CompositorNodeRLayers')
    viewer_node = tree.nodes.new('CompositorNodeViewer')
    tree.links.new(render_layers_node.outputs['Image'], viewer_node.inputs['Image'])

    #scene.render.engine = 'BLENDER_EEVEE'
    # Set the render engine to Cycles
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    # Get Cycles preferences
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
        # Set compute device type (try CUDA, OPTIX for RTX, or HIP for AMD)
    cycles_prefs.compute_device_type = 'OPTIX'  # or 'OPTIX' or 'HIP'
    # Refresh devices
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        if device.type in ['CUDA', 'OPTIX']:
            device.use = True
            log(f"Enabled GPU device: {device.name}")

    # Optimize Cycles for speed over quality
    scene.cycles.samples = 36  # Reduce for faster renders
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.max_bounces = 4  # Reduce bounces
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01


    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.image_settings.file_format = 'PNG'

    return teapot, camera, lights

def generate_and_apply_parameters(teapot, camera, lights):
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

    # === LIGHTING: 5-point randomized setup ===
    
    # 1. Key light (main light source)
    key_dist = random.uniform(*LIGHT_DISTANCE_RANGE)
    key_theta = math.radians(random.uniform(0, 360))
    key_phi = math.radians(random.uniform(30, 150))  # Avoid extreme angles
    
    key_x = key_dist * math.sin(key_phi) * math.cos(key_theta)
    key_y = key_dist * math.sin(key_phi) * math.sin(key_theta)
    key_z = key_dist * math.cos(key_phi)
    
    key_strength = LIGHT_BASE_STRENGTH / (key_dist**LIGHT_STRENGTH_GAMMA)
    
    lights[0].location = (key_x, key_y, key_z)
    lights[0].data.energy = key_strength * 100  # Cycles multiplier
    
    params['lights'] = [{
        'name': 'key',
        'location': (key_x, key_y, key_z),
        'energy': key_strength * 100
    }]
    
    # 2. Fill lights in randomized tetrahedral orbit
    # Random axial tilt for the fill light plane
    tilt_axis_theta = math.radians(random.uniform(0, 360))
    tilt_axis_phi = math.radians(random.uniform(0, 180))
    tilt_angle = math.radians(random.uniform(-45, 45))  # Up to 45° tilt
    
    # Base positions for 4 fill lights (square around teapot)
    fill_base_angles = [0, 90, 180, 270]  # degrees
    fill_dist = random.uniform(LIGHT_DISTANCE_RANGE[0], LIGHT_DISTANCE_RANGE[1])
    
    for i, base_angle in enumerate(fill_base_angles, start=1):
        # Start with position on a circle
        theta = math.radians(base_angle)
        phi = math.radians(90)  # Equatorial
        
        x = fill_dist * math.sin(phi) * math.cos(theta)
        y = fill_dist * math.sin(phi) * math.sin(theta)
        z = fill_dist * math.cos(phi)
        
        # Apply random tilt (rotate around random axis)
        # Simplified: just add random Z offset to break symmetry
        z += random.uniform(-fill_dist * 0.3, fill_dist * 0.3)
        
        fill_strength = (LIGHT_BASE_STRENGTH * 0.3) / (fill_dist**LIGHT_STRENGTH_GAMMA)
        
        lights[i].location = (x, y, z)
        lights[i].data.energy = fill_strength * 100  # Cycles multiplier
        
        params['lights'].append({
            'name': f'fill_{i}',
            'location': (x, y, z),
            'energy': fill_strength * 100
        })

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

    #viewer_image = bpy.data.images['Viewer Node']
    viewer_image = bpy.data.images.get('Viewer Node')
    if viewer_image:
        pixels_sample = np.array(viewer_image.pixels[:100], dtype=np.float32)
        log(f"Pixel sample stats: min={pixels_sample.min():.4f}, max={pixels_sample.max():.4f}, mean={pixels_sample.mean():.4f}")
        
        # Check if it's all zeros or all one value
        if np.allclose(pixels_sample, 0):
            log("WARNING: All pixels are zero - likely blank render", is_error=True)
        elif np.allclose(pixels_sample, pixels_sample[0]):
            log(f"WARNING: All pixels are same value ({pixels_sample[0]:.4f}) - uniform color", is_error=True)
    width, height = viewer_image.size
    
    if width == 0 or height == 0 or not viewer_image.pixels:
        return np.array([], dtype=np.float32)

    pixels = np.array(viewer_image.pixels[:])
    image = pixels.reshape(height, width, 4)[::-1, :, :]
    
    return image

def render_and_get_params(command_data, teapot, camera, lights):
    """Renders an image with the given parameters."""
    res_x = command_data.get("resolution_x", 256)
    res_y = command_data.get("resolution_y", 256)
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y

    teapot.matrix_world = Matrix.Identity(4)
    params = generate_and_apply_parameters(teapot, camera, lights)
    params['render'] = {
        'resolution_x': res_x,
        'resolution_y': res_y,
        'engine': bpy.context.scene.render.engine
    }
    image_buffer = render_to_buffer(bpy.context.scene)
    return image_buffer, params

def main_worker_loop():
    """Synchronous worker loop: reads commands from stdin, writes packets to stdout."""
    try:
        log("Blender Worker: Setting up scene...")
        teapot, camera, lights = setup_scene(STL_PATH)
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
                image_buffer, params = render_and_get_params(command_data, teapot, camera, lights)
                
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