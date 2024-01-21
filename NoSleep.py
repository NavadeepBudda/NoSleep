import sys
import os
import tkinter as tk
from tkinter import simpledialog, Menu, messagebox
import cv2
import dlib
import scipy.spatial
import pygame
from PIL import Image, ImageTk
import threading
import time

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = scipy.spatial.distance.euclidean(eye[1], eye[5])
    B = scipy.spatial.distance.euclidean(eye[2], eye[4])
    C = scipy.spatial.distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Global variables for customization
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 1
WEBCAM_SOURCE = 0
ALARM_VOLUME = 0.5
POMODORO_WORK_TIME = 25 * 60  # 25 minutes work
POMODORO_BREAK_TIME = 5 * 60  # 5 minutes break
last_pomodoro_time = time.time()
pomodoro_mode = False
on_pomodoro_break = False
pomodoro_timer_label = None
pomodoro_active = False

# Function to update customization settings
def update_settings():
    global EAR_THRESHOLD, EAR_CONSEC_FRAMES, POMODORO_WORK_TIME, POMODORO_BREAK_TIME

    new_EAR_THRESHOLD = simpledialog.askfloat("Set EAR Threshold", "Enter EAR threshold:", initialvalue=EAR_THRESHOLD)
    if new_EAR_THRESHOLD is not None:
        EAR_THRESHOLD = new_EAR_THRESHOLD

    new_EAR_CONSEC_FRAMES = simpledialog.askinteger("Set Consecutive Frames", "Enter number of consecutive frames:", initialvalue=EAR_CONSEC_FRAMES)
    if new_EAR_CONSEC_FRAMES is not None:
        EAR_CONSEC_FRAMES = new_EAR_CONSEC_FRAMES

    new_POMODORO_WORK_TIME = simpledialog.askinteger("Set Pomodoro Work Time", "Enter work time (minutes):", initialvalue=POMODORO_WORK_TIME / 60)
    if new_POMODORO_WORK_TIME is not None:
        POMODORO_WORK_TIME = new_POMODORO_WORK_TIME * 60

    new_POMODORO_BREAK_TIME = simpledialog.askinteger("Set Pomodoro Break Time", "Enter break time (minutes):", initialvalue=POMODORO_BREAK_TIME / 60)
    if new_POMODORO_BREAK_TIME is not None:
        POMODORO_BREAK_TIME = new_POMODORO_BREAK_TIME * 60

# Function to toggle Pomodoro mode
def toggle_pomodoro_mode():
    global pomodoro_mode, pomodoro_active, last_pomodoro_time, on_pomodoro_break
    pomodoro_mode = not pomodoro_mode
    pomodoro_active = pomodoro_mode
    if pomodoro_mode:
        last_pomodoro_time = time.time()
        on_pomodoro_break = False
        update_pomodoro_timer()

# Function to update Pomodoro timer
def update_pomodoro_timer():
    global pomodoro_time_left, pomodoro_active, last_pomodoro_time, on_pomodoro_break
    if pomodoro_active:
        current_time = time.time()
        elapsed_time = current_time - last_pomodoro_time
        remaining_time = (POMODORO_BREAK_TIME if on_pomodoro_break else POMODORO_WORK_TIME) - elapsed_time

        if remaining_time <= 0:
            on_pomodoro_break = not on_pomodoro_break
            last_pomodoro_time = current_time
            messagebox.showinfo("Pomodoro", "Break time!" if on_pomodoro_break else "Work time!")
            remaining_time = POMODORO_BREAK_TIME if on_pomodoro_break else POMODORO_WORK_TIME

        mins, secs = divmod(int(remaining_time), 60)
        pomodoro_timer_label.config(text=f"{'Break' if on_pomodoro_break else 'Work'}: {mins:02d}:{secs:02d}")
        pomodoro_timer_label.after(1000, update_pomodoro_timer)
    else:
        pomodoro_timer_label.config(text="Pomodoro Off")


# Function to run the main detection logic
def run_detection(window, label):
    global last_pomodoro_time, on_pomodoro_break, pomodoro_mode

    # Initialize Dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()

    # Check if running as a script or frozen executable
    if getattr(sys, 'frozen', False):
        # If the script is running in a bundle (executable)
        base_path = sys._MEIPASS
    else:
        # If the script is running live
        base_path = os.path.dirname(__file__)

    predictor_path = os.path.join(base_path, "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(WEBCAM_SOURCE)

    alarm_sound_path = os.path.join(base_path, "alarm_sound.wav")
    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound(alarm_sound_path)
    alarm_sound.set_volume(ALARM_VOLUME)

    alarm_on = False
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Webcam Error", "Unable to access the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            # Draw a green rectangle around the detected face
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            for i in range(36, 48):
                cv2.circle(frame, shape[i], 2, (0, 0, 255), -1)

            leftEye = shape[42:48]
            rightEye = shape[36:42]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= EAR_CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        alarm_sound.play()
            else:
                counter = 0
                if alarm_on:
                    alarm_on = False
                    alarm_sound.stop()

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_time = time.time()

        # Pomodoro
        if pomodoro_mode:
            if on_pomodoro_break and current_time - last_pomodoro_time > POMODORO_BREAK_TIME:
                messagebox.showinfo("Pomodoro", "Break time is over. Let's get back to work!")
                on_pomodoro_break = False
                last_pomodoro_time = current_time
            elif not on_pomodoro_break and current_time - last_pomodoro_time > POMODORO_WORK_TIME:
                messagebox.showinfo("Pomodoro", "Time for a break!")
                on_pomodoro_break = True
                last_pomodoro_time = current_time

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        window.update_idletasks()
        window.update()

    cap.release()

# Tkinter GUI setup
def setup_gui(window):
    global pomodoro_timer_label

    menu_bar = Menu(window)
    window.config(menu=menu_bar)

    settings_menu = Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Settings", menu=settings_menu)
    settings_menu.add_command(label="Customize Settings", command=update_settings)
    settings_menu.add_command(label="Toggle Pomodoro Mode", command=toggle_pomodoro_mode)

    pomodoro_timer_label = tk.Label(window, text="Pomodoro Off", font=("Helvetica", 14))
    pomodoro_timer_label.pack()

    label = tk.Label(window)
    label.pack()
    return label

def main():
    window = tk.Tk()
    window.title("NoSleep")
    label = setup_gui(window)

    detection_thread = threading.Thread(target=run_detection, args=(window, label), daemon=True)
    detection_thread.start()

    window.mainloop()

if __name__ == "__main__":
    main()
