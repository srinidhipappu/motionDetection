import numpy as np
import cv2
import pyautogui

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Original Video', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    fgmask = fgbg.apply(thresh)

    cv2.imshow('Background Removed', fgmask)

    kernel = np.ones((10, 10), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    cv2.imshow("Eroded Frame", eroded)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        object_center_x = x + w // 2
        object_center_y = y + h // 2

        frame_width, frame_height = frame.shape[1], frame.shape[0]
        screen_width, screen_height = pyautogui.size()

        mouse_x = int(object_center_x / frame_width * screen_width)
        mouse_y = int(object_center_y / frame_height * screen_height)

        pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)

        print(f"Mouse moving to: ({mouse_x}, {mouse_y})")

    cv2.imshow("Contours Applied", frame)

    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()
