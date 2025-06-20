from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import math
import threading
import time
import atexit
import os
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import json

# ====================== MQTT CONFIGURATION ======================
MQTT_BROKER = "mqtt.ics.ele.tue.nl"
MQTT_AUTH_A = {
    "username": "robot_79_1",
    "password": "faytWakUm0"
}

MQTT_AUTH_B = {
    "username": "robot_80_1",
    "password": "afAjOtVa"
}

ROBOT_A_TOPIC = "/pynqbridge/79/#"
ROBOT_B_TOPIC = "/pynqbridge/80/#"

PUBLISH_TOPIC = "/pynqbridge/79/#"
JSON_FILE_PATH = "paint_grid.json"
PUBLISH_INTERVAL = 1

# ====================== GLOBAL VARIABLES ======================
winh, winw = 900, 900
GRID_ROWS, GRID_COLS = 100, 100

def init_paint_grid(rows, cols):
    return [[{'mapped': False, 'color': None} for _ in range(cols)] for _ in range(rows)]

root_paint_grid = init_paint_grid(GRID_ROWS, GRID_COLS)
grid_lock = threading.Lock()

persistent_paint_layer = Image.new("RGB", (winw, winh), (255, 255, 255))
image_reference = None
image_on_canvas = None


# Robot states: position and orientation per robot
robots_state = {
    'robotA': {'x': 450, 'y': 450, 'orientation': 0},
    'robotB': {'x': 450, 'y': 450, 'orientation': 0}
}
state_lock = threading.Lock()

last_publish_time = 0

# ====================== TKINTER SETUP ======================
root = Tk()
frame = Frame(root)
frame.pack()
C = Canvas(frame, height=winh, width=winw)
C.pack()

for line in range(0, winw, 150):
    C.create_line([(line, 0), (line, winh)], fill='black')
for line in range(0, winh, 150):
    C.create_line([(0, line), (winw, line)], fill='black')

CELL_WIDTH = winw // GRID_COLS
CELL_HEIGHT = winh // GRID_ROWS
cell_rect_ids = [[None]*GRID_COLS for _ in range(GRID_ROWS)]

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        x0 = c * CELL_WIDTH
        y0 = r * CELL_HEIGHT
        x1 = x0 + CELL_WIDTH
        y1 = y0 + CELL_HEIGHT
        rect = C.create_rectangle(x0, y0, x1, y1, fill="", outline="")
        cell_rect_ids[r][c] = rect

robot_cones = {
    'robotA': {'polygon_id': None, 'arc_id': None, 'triangle_points': [400,400,450,450,500,400], 'arc_points': [], 'rotation_origin': (450,450), 'arc_center': (450,400)},
    'robotB': {'polygon_id': None, 'arc_id': None, 'triangle_points': [400,500,450,450,500,500], 'arc_points': [], 'rotation_origin': (450,450), 'arc_center': (450,500)}
}

ARC_STEPS = 20
ARC_RADIUS_X = 50
ARC_RADIUS_Y = 25

def generate_arc_world_points(cx, cy, rx, ry, steps, facing_up=True):
    start_angle = 180 if facing_up else 0
    return [
        (
            cx + rx * math.cos(math.radians(start_angle + (i / steps) * 180)),
            cy + ry * math.sin(math.radians(start_angle + (i / steps) * 180))
        )
        for i in range(steps + 1)
    ]

def setup_robot_cone(robot_key):
    cone = robot_cones[robot_key]
    facing_up = (robot_key == 'robotA')  # Assume 'robotA' faces up, 'robotB' faces down
    cone['arc_points'] = generate_arc_world_points(*cone['arc_center'], ARC_RADIUS_X, ARC_RADIUS_Y, ARC_STEPS, facing_up=facing_up)
    cone['polygon_id'] = C.create_polygon(cone['triangle_points'], fill="", outline="", tags=robot_key)
    arc_flat = [coord for pt in cone['arc_points'] for coord in pt]
    cone['arc_id'] = C.create_polygon(arc_flat, fill="", outline="", tags=robot_key)

setup_robot_cone('robotA')
setup_robot_cone('robotB')

# Paint grid update functions
def cell_color_from_val(val):
    if val == 0:
        return ""
    elif val == 1:
        return "black"
    elif val == 2:
        return "green"
    return ""

def update_canvas_from_grid():
    with grid_lock:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cell = root_paint_grid[r][c]
                if not cell['mapped']:
                    fill = ""
                else:
                    avg_color = cell['color']
                    if avg_color is None:
                        fill = "black"
                    else:
                        fill = '#%02x%02x%02x' % avg_color
                C.itemconfig(cell_rect_ids[r][c], fill=fill)

# Geometry helpers
def rotate_point(x, y, ox, oy, angle_deg):
    angle_rad = math.radians(angle_deg)
    dx, dy = x - ox, y - oy
    x_new = ox + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    y_new = oy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    return x_new, y_new

def get_birdseye_image():
    # Simulated stereo projection: green circle
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 40, (0, 168, 70), -1)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def get_cone_mask_image(triangle_points, arc_points, size):
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    arc_flat = [coord for pt in arc_points for coord in pt]
    cone_poly = triangle_points[:] + arc_flat
    draw.polygon(cone_poly, fill=255)
    return img

def update_robot_cone(robot_key):
    with state_lock:
        x = robots_state[robot_key]['x']
        y = robots_state[robot_key]['y']
        orientation = robots_state[robot_key]['orientation']

    cone = robot_cones[robot_key]

    # Move cone to new position (scaling coordinates if needed)
    dx = (x) - cone['rotation_origin'][0]
    dy = (y) - cone['rotation_origin'][1]
    cone['triangle_points'] = [coord + (dx if i % 2 == 0 else dy) for i, coord in enumerate(cone['triangle_points'])]
    cone['arc_points'] = [(px + dx, py + dy) for (px, py) in cone['arc_points']]
    cone['arc_center'] = (cone['arc_center'][0] + dx, cone['arc_center'][1] + dy)
    cone['rotation_origin'] = (cone['rotation_origin'][0] + dx, cone['rotation_origin'][1] + dy)

    # Rotate cone by orientation
    angle = orientation
    # Rotate points around rotation_origin
    tri_pts = []
    for i in range(0, len(cone['triangle_points']), 2):
        rx, ry = rotate_point(cone['triangle_points'][i], cone['triangle_points'][i+1], *cone['rotation_origin'], angle)
        tri_pts.extend([rx, ry])
    cone['triangle_points'] = tri_pts

    arc_pts_rot = [rotate_point(px, py, *cone['rotation_origin'], angle) for px, py in cone['arc_points']]
    cone['arc_points'] = arc_pts_rot
    cone['arc_center'] = rotate_point(*cone['arc_center'], *cone['rotation_origin'], angle)

    # Update canvas coords
    C.coords(cone['polygon_id'], cone['triangle_points'])
    arc_flat = [coord for pt in cone['arc_points'] for coord in pt]
    C.coords(cone['arc_id'], arc_flat)

def merge_remote_grid(remote_grid):
    with grid_lock:
        for r in range(min(len(remote_grid), GRID_ROWS)):
            for c in range(min(len(remote_grid[0]), GRID_COLS)):
                val = remote_grid[r][c]
                if val == 0:
                    continue
                if val == 1:
                    root_paint_grid[r][c] = {'mapped': True, 'color': (0,0,0)}
                elif val == 2:
                    root_paint_grid[r][c] = {'mapped': True, 'color': (0,168,70)}

def update_cone_texture(robot_key):
    global image_reference, image_on_canvas, persistent_paint_layer

    cone = robot_cones[robot_key]
    base_img = get_birdseye_image().resize((winw, winh))
    mask = get_cone_mask_image(cone['triangle_points'], cone['arc_points'], (winw, winh))
    masked_img = Image.composite(base_img, Image.new("RGB", (winw, winh), (255, 255, 255)), mask)

    persistent_paint_layer.paste(masked_img, (0, 0), mask)

    image_reference = ImageTk.PhotoImage(persistent_paint_layer)
    if image_on_canvas:
        C.itemconfig(image_on_canvas, image=image_reference)
    else:
        globals()['image_on_canvas'] = C.create_image(0, 0, anchor=NW, image=image_reference)

    # Update grid from paint layer
    update_grid_from_paint(persistent_paint_layer)

def update_grid_from_paint(image):
    global root_paint_grid
    img_np = np.array(image)
    h, w, _ = img_np.shape
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    with grid_lock:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                cell_pixels = img_np[y0:y1, x0:x1]
                if np.all(cell_pixels == 255):
                    root_paint_grid[r][c] = {'mapped': False, 'color': None}
                else:
                    mask = np.any(cell_pixels != 255, axis=2)
                    if np.any(mask):
                        avg_color = tuple(np.mean(cell_pixels[mask], axis=0).astype(int))
                        root_paint_grid[r][c] = {'mapped': True, 'color': avg_color}
                    else:
                        root_paint_grid[r][c] = {'mapped': False, 'color': None}

# ====================== MQTT CALLBACKS ======================
def on_connect_A(client, userdata, flags, rc):
    print("Robot A connected with result code", rc)
    client.subscribe(ROBOT_A_TOPIC)

def on_connect_B(client, userdata, flags, rc):
    print("Robot B connected with result code", rc)
    client.subscribe(ROBOT_B_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        robot_key = None
        if msg.topic == ROBOT_A_TOPIC:
            robot_key = 'robotA'
        elif msg.topic == ROBOT_B_TOPIC:
            robot_key = 'robotB'
        if robot_key is None:
            return

        with state_lock:
            if "x" in data:
                robots_state[robot_key]['x'] = data["x"]
            if "y" in data:
                robots_state[robot_key]['y'] = data["y"]
            if "orientation" in data:
                robots_state[robot_key]['orientation'] = data["orientation"]

        if "grid" in data:
            merge_remote_grid(data["grid"])

    except Exception as e:
        print(f"MQTT message error on {msg.topic}:", e)

def mqtt_thread_A():
    client = mqtt.Client()
    client.username_pw_set(MQTT_AUTH_A['username'], MQTT_AUTH_A['password'])
    client.on_connect = on_connect_A
    client.on_message = on_message
    client.connect(MQTT_BROKER)
    client.loop_forever()

def mqtt_thread_B():
    client = mqtt.Client()
    client.username_pw_set(MQTT_AUTH_B['username'], MQTT_AUTH_B['password'])
    client.on_connect = on_connect_B
    client.on_message = on_message
    client.connect(MQTT_BROKER)
    client.loop_forever()

threading.Thread(target=mqtt_thread_A, daemon=True).start()
threading.Thread(target=mqtt_thread_B, daemon=True).start()

# ====================== SAVE GRID TO FILE ======================
def save_grid_to_file():
    with grid_lock:
        if root_paint_grid is None:
            print("Grid data not available yet.")
            return
        filename = JSON_FILE_PATH
        with open(filename, "w") as f:
            for row in root_paint_grid:
                line_vals = []
                for cell in row:
                    if not cell['mapped']:
                        val = 0
                    else:
                        r, g, b = cell['color']
                        if r < 30 and g < 30 and b < 30:
                            val = 1
                        else:
                            val = 2
                    line_vals.append(str(val))
                f.write(" ".join(line_vals) + "\n")
        print(f"Grid saved to {filename}")

Button(frame, text="Save Grid to File", command=save_grid_to_file).pack(pady=10)

# ====================== CANVAS UPDATE + PUBLISH LOOP ======================
def canvas_update_loop():
    global last_publish_time
    while True:
        # Update cone positions
        update_robot_cone('robotA')
        update_robot_cone('robotB')

        # Render and paint each cone's mask
        update_cone_texture('robotA')
        update_cone_texture('robotB')

        # Update grid visuals
        update_canvas_from_grid()

        # Periodic publishing
        if time.time() - last_publish_time >= PUBLISH_INTERVAL:
            save_grid_to_file()
            try:
                with grid_lock:
                    grid_to_publish = [[
                        0 if not cell['mapped'] else (1 if cell['color'][0]<30 and cell['color'][1]<30 and cell['color'][2]<30 else 2)
                        for cell in row
                    ] for row in root_paint_grid]

                payload = json.dumps({"grid": grid_to_publish, "timestamp": time.time()})
                publish.single(PUBLISH_TOPIC, payload, hostname=MQTT_BROKER, auth=MQTT_AUTH_A)
                print("Published merged grid")
                last_publish_time = time.time()
            except Exception as e:
                print("Publishing error:", e)

        time.sleep(0.1)

threading.Thread(target=canvas_update_loop, daemon=True).start()

# ====================== START ======================
root.mainloop()
