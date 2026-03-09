import cv2

# 0 is usually your laptop's default built-in webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame (picture) from the camera
    success, frame = cap.read()

    # If the camera fails to grab a frame, stop the program
    if not success:
        print("Failed to grab frame. Is the camera being used by another app?")
        break

    # Show the picture in a window named 'ISL Camera Test'
    cv2.imshow('ISL Camera Test', frame)

    # Wait for 1 millisecond. If the user presses the 'q' key on the keyboard, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: turn off the camera and close the window
cap.release()
cv2.destroyAllWindows()