from gpiozero import Servo
import time

# Set up the GPIO pins for 5 servos
SERVO_PINS = [5, 6, 13, 19, 26]  # Change these pin numbers if necessary
NUM_SERVOS = len(SERVO_PINS)

# Initialize angles for all servos
servos = [Servo(pin) for pin in SERVO_PINS]
angles = [0.0] * NUM_SERVOS  # Initial angles for all servos (range: -1 to +1)
selected_servo = 0  # The currently selected servo

# Function to set the servo angle (range -1 to +1)
def set_servo_angle(servo_index, angle):
    """Set the angle of a specific servo."""
    angle = max(-1.0, min(1.0, angle))  # Clamp angle to range -1 to 1
    servos[servo_index].value = angle

def cleanup():
    """Reset all servos to neutral position."""
    for servo in servos:
        servo.value = None

try:
    print("Controls: 1-5 to select a servo, u/d to adjust angle up/down, q to quit.")
    while True:
        command = input("Enter command: ").strip().lower()

        # Select servo
        if command in ['1', '2', '3', '4', '5']:
            selected_servo = int(command) - 1
            print(f"Selected servo {selected_servo + 1}")

        # Adjust angle
        elif command == 'u':
            angles[selected_servo] = min(angles[selected_servo] + 0.1, 1.0)
            set_servo_angle(selected_servo, angles[selected_servo])
            print(f"Servo {selected_servo + 1} angle: {angles[selected_servo]:.2f}")
        elif command == 'd':
            angles[selected_servo] = max(angles[selected_servo] - 0.1, -1.0)
            set_servo_angle(selected_servo, angles[selected_servo])
            print(f"Servo {selected_servo + 1} angle: {angles[selected_servo]:.2f}")
        elif command == 'w':
            angles[selected_servo] = min(angles[selected_servo] + 0.3, 1.0)
            set_servo_angle(selected_servo, angles[selected_servo])
            print(f"Servo {selected_servo + 1} angle: {angles[selected_servo]:.2f}")
        elif command == 's':
            angles[selected_servo] = max(angles[selected_servo] - 0.3, -1.0)
            set_servo_angle(selected_servo, angles[selected_servo])
            print(f"Servo {selected_servo + 1} angle: {angles[selected_servo]:.2f}")

        # Quit the program
        elif command == 'q':
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("\nProgram interrupted.")

finally:
    cleanup()
    print("Cleaned up servos.")