from gpiozero import Servo
import time

# Servo Configuration
servos_config = {
    "left_eye_horizontal": {"pin": 13, "center": 0.0, "min": -1.0, "max": 1.0},
    "left_eye_vertical": {"pin": 5, "center": 0.63, "min": -1.0, "max": 1.0},
    "right_eye_horizontal": {"pin": 6, "center": 0.2, "min": -1.0, "max": 1.0},
    "right_eye_vertical": {"pin": 19, "center": 0.54, "min": -1.0, "max": 1.0},
    "eyelids": {"pin": 26, "center": 1.0, "min": -0.4, "max": 1.0}
}

# Initialize servos
servos = {
    name: Servo(config["pin"]) for name, config in servos_config.items()
}

# Initialize positions with the center values
positions = {name: config["center"] for name, config in servos_config.items()}

def set_servo_position(servo_name, position):
    """Set the position of a specific servo with limits."""
    config = servos_config[servo_name]
    # Clamp the position within min and max limits
    #(config["max"] - config["min"]) * 
    position = max(config["min"], min(config["max"], position))
    servos[servo_name].value = position
    positions[servo_name] = position

def cleanup():
    """Reset all servos to their neutral positions."""
    for name, config in servos_config.items():
        servos[name].value = config["center"]

def center_servos():
    positions["left_eye_horizontal"] = servos_config["left_eye_horizontal"]["center"]
    positions["right_eye_horizontal"] = servos_config["right_eye_horizontal"]["center"]
    positions["left_eye_vertical"] = servos_config["left_eye_vertical"]["center"]
    positions["right_eye_vertical"] = servos_config["right_eye_vertical"]["center"]
    update_eye_positions()


def update_eye_positions():
    set_servo_position("left_eye_horizontal", positions["left_eye_horizontal"])
    set_servo_position("right_eye_horizontal", positions["right_eye_horizontal"])
    set_servo_position("left_eye_vertical", positions["left_eye_vertical"])
    set_servo_position("right_eye_vertical", positions["right_eye_vertical"])

try:
    print("Controls: WSAD to move eyes, E to toggle eyelids, Q to quit.")
    
    set_servo_position("eyelids", positions["eyelids"])
    # Set the new positions for the eyes
    update_eye_positions()
    
    while True:
        command = input("Enter command: ").strip().lower()

        # Control eye movement
        if command == 'w':
            # Move eyes up
            positions["left_eye_vertical"] = max(positions["left_eye_vertical"] - 0.1, servos_config["left_eye_vertical"]["min"])
            positions["right_eye_vertical"] = min(positions["right_eye_vertical"] + 0.1, servos_config["right_eye_vertical"]["max"])
        
        elif command == 's':
            # Move eyes down
            positions["left_eye_vertical"] = min(positions["left_eye_vertical"] + 0.1, servos_config["left_eye_vertical"]["max"])
            positions["right_eye_vertical"] = max(positions["right_eye_vertical"] - 0.1, servos_config["right_eye_vertical"]["min"])
        
        elif command == 'a':
            # Move eyes left
            positions["left_eye_horizontal"] = max(positions["left_eye_horizontal"] - 0.1, servos_config["left_eye_horizontal"]["min"])
            positions["right_eye_horizontal"] = min(positions["right_eye_horizontal"] + 0.1, servos_config["right_eye_horizontal"]["max"])
        
        elif command == 'd':
            # Move eyes right
            positions["left_eye_horizontal"] = min(positions["left_eye_horizontal"] + 0.1, servos_config["left_eye_horizontal"]["max"])
            positions["right_eye_horizontal"] = max(positions["right_eye_horizontal"] - 0.1, servos_config["right_eye_horizontal"]["min"])
        
        elif command == 'c':
            center_servos()
        # Control eyelids
        elif command == 'e':
            center_servos()
            # Toggle eyelid position between open and closed
            if positions["eyelids"] < 0.5:
                positions["eyelids"] = 1.0
            else:
                positions["eyelids"] = -1.0
            set_servo_position("eyelids", positions["eyelids"])
            state = "open" if positions["eyelids"] > 0 else "closed"
            print(f"Eyelids {state}")
        elif command == 'b':
            center_servos()
            positions["eyelids"] = -1.0
            set_servo_position("eyelids", positions["eyelids"])
            time.sleep(0.25)
            positions["eyelids"] = 1.0
            set_servo_position("eyelids", positions["eyelids"])
        elif command == 'l':
            positions["left_eye_horizontal"] = servos_config["left_eye_horizontal"]["max"] * 0.7
            positions["right_eye_horizontal"] = servos_config["right_eye_horizontal"]["max"] * 0.7
            update_eye_positions()

            time.sleep(1.45)
            positions["left_eye_horizontal"] = servos_config["left_eye_horizontal"]["min"] * 0.7
            positions["right_eye_horizontal"] = servos_config["right_eye_horizontal"]["min"] * 0.7
            update_eye_positions()
            time.sleep(1.45)
            positions["left_eye_horizontal"] = servos_config["left_eye_horizontal"]["max"] * 0.7
            positions["right_eye_horizontal"] = servos_config["right_eye_horizontal"]["max"] * 0.7
            update_eye_positions()
            time.sleep(1.45)
            positions["left_eye_horizontal"] = servos_config["left_eye_horizontal"]["center"]
            positions["right_eye_horizontal"] = servos_config["right_eye_horizontal"]["center"]
            update_eye_positions()
        # Quit program
        elif command == 'q':
            print("Exiting...")
            break

        # Set the new positions for the eyes
        update_eye_positions()
        



        # Print current positions for debugging
        print(f"Eye Horizontal: {positions['left_eye_horizontal']:.2f}, {positions['right_eye_horizontal']:.2f}")
        print(f"Eye Vertical: {positions['left_eye_vertical']:.2f}, {positions['right_eye_vertical']:.2f}")
        print(f"Eyelid position: {positions['eyelids']:.2f}")
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nProgram interrupted.")

finally:
    cleanup()
    print("Cleaned up servos.")
