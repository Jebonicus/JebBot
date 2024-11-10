import time
import board
import busio
from adafruit_pca9685 import PCA9685

# Initialize I2C bus and PCA9685 module
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Frequency for servos (50 Hz is standard)

# Configuration for the servo range
SERVO_CHANNELS = [0, 1, 2, 3, 4]
MIN_PULSE = 1000  # Minimum pulse length out of 4096
MAX_PULSE = 2000  # Maximum pulse length out of 4096
CENTER_PULSE = 1500  # Center pulse length

def set_servo_pulse(channel, pulse):
    """Set the servo pulse for a given channel."""
    duty_cycle = int((pulse / 20000) * 65535)
    pca.channels[channel].duty_cycle = duty_cycle

def move_servo_full_range(channel):
    """Move servo through full range and return to center."""
    print(f"Moving servo on channel {channel}...")
    
    # Move to minimum position
    set_servo_pulse(channel, MIN_PULSE)
    time.sleep(0.5)
    
    # Move to maximum position
    set_servo_pulse(channel, MAX_PULSE)
    time.sleep(0.5)
    
    # Return to center
    set_servo_pulse(channel, CENTER_PULSE)
    time.sleep(0.5)
    print(f"Servo on channel {channel} centered.")

def main():
    print("Servo Tester - Channels 0 to 4")
    while True:
        try:
            # Get input from user for which servo to test
            channel = input("Enter servo channel (0-4) or 'q' to quit: ")
            
            if channel.lower() == 'q':
                print("Exiting...")
                break
            
            # Validate input
            if not channel.isdigit() or int(channel) not in SERVO_CHANNELS:
                print("Invalid input. Please enter a number between 0 and 4.")
                continue

            channel = int(channel)
            move_servo_full_range(channel)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

    # Turn off all servos before exiting
    for ch in SERVO_CHANNELS:
        set_servo_pulse(ch, CENTER_PULSE)
    pca.deinit()

if __name__ == "__main__":
    main()
