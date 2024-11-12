import argparse
import sys
import time

import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer

from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
import time
import board
import busio
from adafruit_pca9685 import PCA9685
import sys
import termios
import tty
import cv2

global args
global slaveEyes

# Initialize the PCA9685 servo controller
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Set to 50 Hz for servos

# Servo configuration with flip parameter
SERVOS = {
    'right_eye_horizontal': {'channel': 0, 'center': 1750, 'min': 1100, 'max': 2100, 'flip': False},
    'right_eye_vertical': {'channel': 1, 'center': 1750, 'min': 1000, 'max': 2500, 'flip': False},
    'left_eye_horizontal': {'channel': 2, 'center': 1400, 'min': 900, 'max': 1900, 'flip': False},
    'left_eye_vertical': {'channel': 3, 'center': 1800, 'min': 1000, 'max': 2500, 'flip': True},
    'eyelids': {'channel': 4, 'center': 1500, 'min': 1000, 'max': 2300, 'flip': False}
}

STEP = 50  # Movement step size for the servos

last_boxes = None
last_scores = None
last_keypoints = None
WINDOW_SIZE_H_W = (480, 640)


def set_servo_pulse(channel, pulse):
    """Set the servo pulse for a given channel on the PCA9685."""
    duty_cycle = int((pulse / 20000) * 65535)
    pca.channels[channel].duty_cycle = duty_cycle

def center_servos():
    """Center all servos."""
    for servoName in {'left_eye_horizontal','left_eye_vertical','right_eye_horizontal','right_eye_vertical'}:
        servo = SERVOS[servoName]
        set_servo_pulse(servo['channel'], servo['center'])
        servo['current'] = servo['center']
    open_eyes()

def move_servo(servo_name, step):
    """Move a servo by a specified step, considering its flip setting."""
    servo = SERVOS[servo_name]
    current_pulse = servo.get('current', servo['center'])
    
    # Apply the flip if needed
    adjusted_step = -step if servo['flip'] else step
    new_pulse = current_pulse + adjusted_step
    
    # Clamp the new pulse to min and max limits
    new_pulse = max(servo['min'], min(servo['max'], new_pulse))
    
    # Set the new pulse and update the current position
    set_servo_pulse(servo['channel'], new_pulse)
    servo['current'] = new_pulse

def set_servo(servo_name, absval):
    """Move a servo to a specified absval"""
    servo = SERVOS[servo_name]
    # Clamp the new pulse to min and max limits
    new_pulse = max(servo['min'], min(servo['max'], absval))
    
    # Set the new pulse and update the current position
    set_servo_pulse(servo['channel'], new_pulse)
    servo['current'] = new_pulse

def set_servo_norm(servo_name, normalisedVal, scale_factor=1.0):
    """Transform normalisedVal (0-1 range) into an actual servo value for servo_name"""
    s = SERVOS[servo_name]
    if (not s['flip'] and normalisedVal < 0.5) or (s['flip'] and normalisedVal >= 0.5):
        scaledVal=0.5 - max(0,min(0.5,(0.5-normalisedVal)*scale_factor))
        val = (s['center'] - s['min']) * scaledVal * 2 + s['min']
    else:
        val = (s['max'] - s['center']) * (normalisedVal - 0.5) * 2 * scale_factor + s['center']
    if servo_name=='left_eye_vertical':
        print(f'\rset_servo_norm normalisedVal={normalisedVal:.2f}\t\t{val}')
        sys.stdout.flush()
    set_servo(servo_name, val)

def set_servo_xy(x, y):
    """Move the eye servos to a specific XY in range 0-1 (origin top left)"""
    
    set_servo_norm('left_eye_horizontal', x, 2.0)
    set_servo_norm('right_eye_horizontal', x, 2.0)

    #set_servo_norm('left_eye_vertical', y)
    #set_servo_norm('right_eye_vertical', y)


def blink_eyelids():
    """Blink the eyelids by closing and then opening them."""
    
    # Close eyelids fully
    close_eyes()
    time.sleep(0.2)
    
    # Open eyelids fully
    open_eyes()

def open_eyes():
    eyelid_servo = SERVOS['eyelids']
    set_servo_pulse(eyelid_servo['channel'], eyelid_servo['max'])
    eyelid_servo['current'] = eyelid_servo['max']

def close_eyes():
    eyelid_servo = SERVOS['eyelids']
    set_servo_pulse(eyelid_servo['channel'], eyelid_servo['min'])
    eyelid_servo['current'] = eyelid_servo['min']


def get_key():
    """Get a single character from standard input (non-blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def look_left():
    """Move both eyes 80% to the left."""
    #set_servo('left_eye_horizontal', int(0.2 * (SERVOS['left_eye_horizontal']['center'] - SERVOS['left_eye_horizontal']['min']) + SERVOS['left_eye_horizontal']['min']))
    #set_servo('right_eye_horizontal', int(0.2 * (SERVOS['right_eye_horizontal']['center'] - SERVOS['right_eye_horizontal']['min']) + SERVOS['right_eye_horizontal']['min']))
    set_servo_norm('left_eye_horizontal', 0.2)
    set_servo_norm('right_eye_horizontal', 0.2)

def look_right():
    """Move both eyes 80% to the right."""
    #set_servo('left_eye_horizontal', int(0.8 * (SERVOS['left_eye_horizontal']['max'] - SERVOS['left_eye_horizontal']['center']) + SERVOS['left_eye_horizontal']['center']))
    #set_servo('right_eye_horizontal', int(0.8 * (SERVOS['right_eye_horizontal']['max'] - SERVOS['right_eye_horizontal']['center']) + SERVOS['right_eye_horizontal']['center']))
    set_servo_norm('left_eye_horizontal', 0.8)
    set_servo_norm('right_eye_horizontal', 0.8)

def look_left_right_sequence():
    """Perform a sequence: look left, then right, then back to center."""
    look_left()
    time.sleep(0.4)
    look_right()
    time.sleep(0.4)
    look_left()
    time.sleep(0.4)
    center_servos()

def servo_main():
    print("Use WASD keys to move the eyes. Press 'E' to toggle eyelids. 'B' to blink. 'C' to center. Press 'Q' to quit.")
    center_servos()

    try:
        while True:
            key = get_key().lower()

            if key == 'q':
                break
            elif key == 'w':
                # Move eyes up
                move_servo('left_eye_vertical', STEP)
                move_servo('right_eye_vertical', STEP)
            elif key == 's':
                # Move eyes down
                move_servo('left_eye_vertical', -STEP)
                move_servo('right_eye_vertical', -STEP)
            elif key == 'a':
                # Move eyes left
                move_servo('left_eye_horizontal', -STEP)
                move_servo('right_eye_horizontal', -STEP)
            elif key == 'd':
                # Move eyes right
                move_servo('left_eye_horizontal', STEP)
                move_servo('right_eye_horizontal', STEP)
            elif key == 'e':
                # Toggle eyelids open/close
                eyelid_servo = SERVOS['eyelids']
                if eyelid_servo.get('current', eyelid_servo['center']) <= eyelid_servo['center']:
                    open_eyes()
                else:
                    close_eyes()
            elif key == 'b':
                # Blink the eyelids
                blink_eyelids()
            elif key == 'c':
                # Blink the eyelids
                center_servos()
            elif key == 'z':
                look_left()
            elif key == 'x':
                look_right()
            elif key == 'l':
                look_left_right_sequence()
            elif key == 'v':
                toggle_slave()

            time.sleep(0.05)  # Small delay to prevent rapid movements

    except KeyboardInterrupt:
        pass
    finally:
        print("\nCentering servos and exiting...")
        center_servos()
        close_eyes()
        pca.deinit()

def toggle_slave():
    global slaveEyes
    """Toggles slaving eyes to camera face position"""
    slaveEyes = not slaveEyes

def ai_output_tensor_parse(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(outputs=np_outputs,
                                                           img_size=WINDOW_SIZE_H_W,
                                                           img_w_pad=(0, 0),
                                                           img_h_pad=(0, 0),
                                                           detection_threshold=args.detection_threshold,
                                                           network_postprocess=True,
                                                           max_num_people=5)

        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)
    return last_boxes, last_scores, last_keypoints


def ai_output_tensor_draw(request: CompletedRequest, boxes, scores, keypoints, stream='main'):
    """Draw the detections for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(m.array, boxes, scores,
                                  np.zeros(scores.shape), keypoints, args.detection_threshold,
                                  args.detection_threshold, request.get_metadata(), picam2, stream)
            #print(keypoints)


def picamera2_pre_callback(request: CompletedRequest):
    """Analyse the detected objects in the output tensor and draw them on the main output image."""
    global slaveEyes
    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    #if args.gui:
        #ai_output_tensor_draw(request, boxes, scores, keypoints)
    stream='main'
    if keypoints is not None and boxes is not None:
        with MappedArray(request, stream) as m:
            for bIndex, bRow in enumerate(boxes):
                if scores[bIndex] > 0.1:
                    k = keypoints[bIndex]
                    # nose
                    kIndex=0
                    if len(k) >= 1:
                    #for kIndex in range(len(k)):
                        confidence=k[kIndex][2]
                        if confidence > 0.4:
                            x, y = max(0, int(k[kIndex][0])), max(0, int(k[kIndex][1]))
                            x0, y0 = x / WINDOW_SIZE_H_W[1], y / WINDOW_SIZE_H_W[0]
                            if slaveEyes and bIndex==0:
                                set_servo_xy(1.0 - x0, y0)
                            elif not slaveEyes:
                                print(f'Person {bIndex}, Nose: {x0:.2f},{y0:.2f}', end="\r\n", flush=True)
                            if args.gui:
                                img = m.array
                                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                                label = f"#{bIndex} b={scores[bIndex]:.2f} c={confidence:.3f}"
                                cv2.putText(img, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--gui", type=bool, help="Show GUI", default=False)
    parser.add_argument("--detection-threshold", type=float, default=0.1,
                        help="Post-process detection threshold")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


def get_drawer():
    categories = intrinsics.labels
    categories = [c for c in categories if c and c != "-"]
    return COCODrawer(categories, imx500, needs_rescale_coords=False)

def reset_terminal():
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSANOW, termios.tcgetattr(fd))
    fd = sys.stdout.fileno()
    termios.tcsetattr(fd, termios.TCSANOW, termios.tcgetattr(fd))

if __name__ == "__main__":
    args = get_args()
    slaveEyes = False

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Network is not a pose estimation task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    drawer = get_drawer()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=args.gui)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = picamera2_pre_callback

    reset_terminal()

    servo_main()
    