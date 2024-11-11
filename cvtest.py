import cv2
import numpy as np

# Create a blank image (black) with size 640x480
image = np.zeros((480, 640, 3), dtype=np.uint8)

# Create an OpenCV window
cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)

# Display the blank image
cv2.imshow("Test Window", image)

# Wait for a key press (press 'q' to quit)
print("Press 'q' to close the window...")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the window
cv2.destroyAllWindows()

