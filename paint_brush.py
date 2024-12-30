import cv2
import numpy as np

canvas = np.zeros((2000, 2000, 3), dtype="uint8")
drawing = False
brush_color = (255, 255, 255)
brush_size = 5
last_point = None  # Initialize last_point globally
show_instructions = False
inst_button_width = 200
inst_button_height = 100
inst_box_width = 550
inst_box_height = 420

def draw(event, x, y, flags, param):
    global drawing, last_point, brush_color, brush_size, show_instructions

    if event == cv2.EVENT_LBUTTONDOWN:  # Start or stop drawing
        if 0 <= x <= inst_button_width and 0 <= y <= inst_button_height:
            show_instructions = not show_instructions
            return
        drawing = not drawing
        last_point = None  # Reset the last point to avoid abrupt lines
    elif event == cv2.EVENT_MOUSEMOVE:  # Draw when moving with button pressed
        if last_point is not None and drawing:
            cv2.line(canvas, last_point, (x, y), brush_color, brush_size)  # Connect previous point to current
        last_point = (x, y) 
    



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
    elif key == ord('e'):
        brush_color = (0, 0, 0)
    elif key == ord('+'):
        brush_size += 1
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)

def display_instructions(temp_canvas):
    cv2.rectangle(temp_canvas, (0, inst_button_height), (inst_box_width, inst_button_height + inst_box_height), (200, 200, 200), -1)
    instructions = [
        "Press 'r' for Red",
        "Press 'g' for Green",
        "Press 'b' for Blue",
        "Press 'w' for White",
        "Press 'e' for Eraser (Black)",
        "Press '+' to increase brush size",
        "Press '-' to decrease brush size",
        "Press 's' to save the painting",
        "Press 'q' to quit"
    ]
    y_offset = 140  # Initial y-offset for text
    for instruction in instructions:
        cv2.putText(temp_canvas, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += 40


# Main loop
cv2.namedWindow("Paint")
cv2.setMouseCallback("Paint", draw)

while True:
    temp_canvas = canvas.copy()  # Use a copy to overlay instructions
    cv2.rectangle(temp_canvas, (0, 0), (inst_button_width, inst_button_height), (200, 200, 200), -1)
    cv2.putText(temp_canvas, "Instructions", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    if show_instructions:
        display_instructions(temp_canvas)  # Display instructions on the canvas

    cv2.imshow("Paint", temp_canvas)
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        handle_keys(key)
    if key == ord('s'):
        cv2.imwrite("my_painting.jpg", canvas)
        print("Painting saved as 'my_painting.jpg'")
    if key == ord('q'):
        break

cv2.destroyAllWindows()