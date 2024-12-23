import cv2
import numpy as np

canvas = np.zeros((500, 800, 3), dtype="uint8")
drawing = False
brush_color = (255, 255, 255)
brush_size = 5
last_point = None  # Initialize last_point globally

def draw(event, x, y, flags, param):
    global drawing, last_point, brush_color, brush_size

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        last_point = (x, y)  # Set the starting point

    elif event == cv2.EVENT_MOUSEMOVE:  # Draw when moving with button pressed
        if last_point is not None:
            cv2.line(canvas, last_point, (x, y), brush_color, brush_size)  # Connect previous point to current
        last_point = (x, y)  # Update the last point

    elif event == cv2.EVENT_RBUTTONDOWN:  # Stop drawing
        last_point = None

def handle_keys(key):
    global brush_color, brush_size
    if key == ord('r'):
        brush_color = (0, 0, 255)
    elif key == ord('g'):
        brush_color = (0, 255, 0)
    elif key == ord('b'):
        brush_color = (255, 0, 0)
    elif key == ord('w'):
        brush_color = (255, 255, 255)
    elif key == ord('k'):
        brush_color = (0, 0, 0)
    elif key == ord('+'):
        brush_size += 1
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)

# Main loop
cv2.namedWindow("Paint")
cv2.setMouseCallback("Paint", draw)

while True:
    cv2.imshow("Paint", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        handle_keys(key)
    if key == ord('s'):
        cv2.imwrite("my_painting.jpg", canvas)
        print("Painting saved as 'my_painting.jpg'")
    if key == ord('q'):
        break

cv2.destroyAllWindows()
