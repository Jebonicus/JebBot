import time
import board
import busio
from adafruit_pca9685 import PCA9685
import sys
import termios
import tty

# Initialize the PCA9685 servo controller
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Set to 50 Hz for servos

# Servo configuration with flip parameter
SERVOS = {
    'right_eye_horizontal': {'channel': 0, 'center': 1750, 'min': 1100, 'max': 2100, 'flip': False},
    'right_eye_vertical': {'channel': 1, 'center': 1750, 'min': 1000, 'max': 2500, 'flip': False},
    'left_eye_horizontal': {'channel': 2, 'center': 1450, 'min': 900, 'max': 1900, 'flip': False},
    'left_eye_vertical': {'channel': 3, 'center': 1800, 'min': 1000, 'max': 2500, 'flip': True},
    'eyelids': {'channel': 4, 'center': 1500, 'min': 1000, 'max': 2300, 'flip': False}
}

STEP = 50  # Movement step size for the servos

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
    set_servo('left_eye_horizontal', -int(0.8 * (SERVOS['left_eye_horizontal']['center'] - SERVOS['left_eye_horizontal']['min']) + SERVOS['left_eye_horizontal']['min']))
    set_servo('right_eye_horizontal', -int(0.8 * (SERVOS['right_eye_horizontal']['center'] - SERVOS['right_eye_horizontal']['min']) + SERVOS['right_eye_horizontal']['min']))

def look_right():
    """Move both eyes 80% to the right."""
    set_servo('left_eye_horizontal', int(0.8 * (SERVOS['left_eye_horizontal']['max'] - SERVOS['left_eye_horizontal']['center']) + SERVOS['left_eye_horizontal']['center']))
    set_servo('right_eye_horizontal', int(0.8 * (SERVOS['right_eye_horizontal']['max'] - SERVOS['right_eye_horizontal']['center']) + SERVOS['right_eye_horizontal']['center']))

def look_left_right_sequence():
    """Perform a sequence: look left, then right, then back to center."""
    look_left()
    time.sleep(0.4)
    look_right()
    time.sleep(0.4)
    look_left()
    time.sleep(0.4)
    center_servos()

def main():
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

            time.sleep(0.05)  # Small delay to prevent rapid movements

    except KeyboardInterrupt:
        pass
    finally:
        print("\nCentering servos and exiting...")
        center_servos()
        pca.deinit()

if __name__ == "__main__":
    main()
