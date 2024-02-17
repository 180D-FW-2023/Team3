from turtle import bgcolor
import paho.mqtt.client as mqtt
import cv2
import mediapipe as mp
import time
from time import sleep
import numpy as np
from collections import Counter
import speech_recognition as sr  # Added import for speech recognition
import math
import customtkinter as ctk
import threading
from PIL import Image, ImageTk

class BicepCurlApp:
    def __init__(self, app):
        self.app = app
        self.app.title("piTrainer")

        # Set the size of the window
        self.app.geometry("1280x800")  # Set your desired size here

        # Initialize OpenCV video capture
        self.capture = cv2.VideoCapture(0)

        self.old_good_time = 0
        self.old_bad_time = 0

        bottom_frame = ctk.CTkFrame(app, fg_color="black", border_width=2, border_color="blue")
        bottom_frame.pack(side="top", fill="both")  # Place the bottom frame underneath the video stream

        # Create a custom canvas widget
        self.canvas = ctk.CTkCanvas(self.app, width=800, height=450)
        self.canvas.pack(side="right", anchor="n", pady=100)

        # Create a frame as a container for the labels on the left side
        left_frame = ctk.CTkFrame(app, fg_color="black", border_width=2,border_color="blue", width= 200)
        left_frame.pack(side="left", fill="y")

        # Create and style labels for left column
        possible_workouts_label = ctk.CTkLabel(master=left_frame, text="Possible Workouts:", text_color="white", font=("Arial", 20))
        possible_workouts_label.pack(anchor="w", pady = 40,padx = 10)

        possible_workouts_info = ctk.CTkLabel(master=left_frame, text="Bicep Curl\nPushup\nSquat\nLeg Raise\nPlank", text_color="white", font=("Arial", 20))
        possible_workouts_info.pack(anchor="center", pady = 5, padx = 10)

        reps_completion_label = ctk.CTkLabel(master=left_frame, text="Reps for Completion:", text_color="white", font=("Arial", 20))
        reps_completion_label.pack(anchor="w", pady = 40, padx = 10)

        reps_completion_info = ctk.CTkLabel(master=left_frame, text="Bicep Curl: 10\nPushup: 10\nSquat: 10\n Leg Raise: 10\n Plank: 1 min", text_color="white", font=("Arial", 20))
        reps_completion_info.pack(anchor="center", pady = 5, padx = 10)

        instructions_info = ctk.CTkLabel(master=bottom_frame, text="Welcome to piTrainer", text_color="white", font=("Arial", 40), anchor="w")
        instructions_info.pack(pady = 5, padx = 10)

        instructions_label = ctk.CTkLabel(master=bottom_frame, text="Instructions:", text_color="white", font=("Arial", 20))
        instructions_label.pack(anchor="w", padx = 10)

        instructions_info = ctk.CTkLabel(master=bottom_frame, text="When the button states GO please feel free to begin your rep.", text_color="white", font=("Arial", 20))
        instructions_info.pack(anchor="w", padx = 10)

        instructions_info_2 = ctk.CTkLabel(master=bottom_frame, text="When it says PAUSE, please hold the bottom of the rep until it switches back to GO", text_color="white", font=("Arial", 20))
        instructions_info_2.pack(anchor="w",padx = 10, pady=5)

        # Create a frame as a container for the labels
        self.frame = ctk.CTkFrame(app, fg_color= "black", border_width= 2, border_color= "blue", width=200)
        self.frame.pack(side = "left", fill="y")

        self.btn_pause = ctk.CTkButton(self.frame, fg_color="green", corner_radius = 8, text="GO", font=("Arial", 50))
        self.btn_pause.pack(side= "top", pady=30,  padx = 10)

        if(exercise_flag == 1 or exercise_flag == 2 or exercise_flag == 3 or exercise_flag == 4):
            #create dummy button to determine the width
            self.lbl_dummy = ctk.CTkButton(master=self.frame, fg_color="black",text="Im just filling", text_color="black", font=("Arial", 1))
            self.lbl_dummy.pack(padx = 200)
            # Create and style the labels
            self.lbl_rep = ctk.CTkButton(master=self.frame, corner_radius = 8, text="Reps:\n 0", text_color = "white", font=("Arial", 50))
            self.lbl_rep.pack(side="top",  padx = 10)

            self.lbl_error = ctk.CTkButton(self.frame, corner_radius = 8, text="Errors:\n 0", text_color = "white",font=("Arial", 50))
            self.lbl_error.pack(side="top", pady=50,  padx = 10)

            self.lbl_time = ctk.CTkButton(self.frame, corner_radius = 8, text="Time:\n 0 seconds", text_color = "white",  font=("Arial", 50))
            self.lbl_time.pack(side="top", pady=0,  padx = 10)
        if (exercise_flag == 5):
            self.lbl_goodtime = ctk.CTkButton(self.frame, corner_radius = 8, text="Time\n Proper Form:\n 0 seconds", text_color = "white",  font=("Arial", 30))
            self.lbl_goodtime.pack(side="top", pady=50,  padx = 10)

            self.lbl_badtime = ctk.CTkButton(self.frame, corner_radius = 8, text="Time\n Improper Form:\n 0 seconds", text_color = "white",  font=("Arial", 30))
            self.lbl_badtime.pack(side="top", pady=0,  padx = 20)
        # Event to signal when the workout is complete
        self.workout_complete_event = threading.Event()

        self.update_frame()

    def update_frame(self):
        if self.workout_complete_event.is_set():
            pass
            #self.app.quit()  # Terminate the main loop
        # Read a frame from the video capture
        ret, frame = self.capture.read()
        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (800, 450))
            # Convert frame to ImageTk format
            img = Image.fromarray(resized_frame)

            img = img.resize((800, 450), Image.ANTIALIAS)

            img_tk = ImageTk.PhotoImage(img)

            # Display the new frame on canvas
            self.canvas.create_image(0, 0, anchor=ctk.NW, image=img_tk)

            # Keep a reference to the image to prevent it from being garbage collected
            self.canvas.img = img_tk
        # Find time the code has been running
        elapsed = time.time() - code_start
        # Update the labels
        if(exercise_flag == 1 or exercise_flag == 2 or exercise_flag == 3 or exercise_flag == 4):
            self.lbl_rep.configure(text=f"Reps:\n {reps}")
            self.lbl_error.configure(text=f"Errors:\n {errors}")
            self.lbl_time.configure(text=f"Time:\n {int(elapsed)}")

        else:
            self.lbl_goodtime.configure(text=f"Time\nProper Form:\n {(good_time)}")
            self.lbl_badtime.configure(text=f"Time\nImproper Form:\n {(bad_time)}")

        if Pause:
            btn_text = "PAUSE"
            btn_fg_color = "red"
        else:
            btn_text = "GO"
            btn_fg_color = "green"

        if self.btn_pause.cget("text") != btn_text:
            self.btn_pause.configure(text=btn_text)
        if self.btn_pause.cget("fg_color") != btn_fg_color:
            self.btn_pause.configure(fg_color=btn_fg_color)
        # Schedule the next frame update after a delay (in milliseconds)
        self.app.after(10, self.update_frame)

    def run(self):
        # Start the GUI main loop
        self.app.mainloop()

def my_code(workout_complete_event):
    global Pause, good_time, bad_time, errors, prev_joint_positions, smoothing_factor, calibration_pabove_values, calibration_period, reps, errCounterX, errCounterY, errPause, cooldown_period,  calibration_joint_values_x,calibration_joint_values_y, desiredJoint,  Joint_x, Joint_y, Shoulder_x, Shoulder_y,hip_x,hip_y, pTime
    if(exercise_flag == 1):
        blue = (255, 127, 0)
        red = (50, 50, 255)
        green = (127, 255, 0)
        dark_blue = (127, 20, 0)
        light_green = (127, 233, 100)
        yellow = (0, 255, 255)
        pink = (255, 0, 255)
        # Set all time relevant values after we recieve the start command so that these dont begin at the wrong time and mess everything up
        cooldown_start_time = time.time()
        calibration_start_time = time.time()
        cooldown_start_time_err = time.time()
        # This is variables to stop processing
        breakflag = False
        counter = 0
        # This is your actual processing code
        while True:
            if time.time()-code_start > 5 and counter == 0:
                Pause = False
                counter+=1
            if counter >= 31:
                client.publish("Control", "Workout Complete!")
                sleep(1)
                workout_complete_event.set()
                break
            if reps >= 10 and counter == 1:
                breakflag = True
        
            # Read frame
            success, img = cap.read()

            # rgb for skeleton
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            # Clear previous joint positions for the new frame
            current_joint_positions = []

            # Following line is the original skeleton drawing, but we dont really need the connections drawn or the points due to later code
        
            lm = results.pose_landmarks
            lmPose = mpPose.PoseLandmark
            h, w, c = img.shape

            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

            # Right elbow
            r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
            r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

            # Right wrist
            r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
            r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
            
            joints = [(r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), (r_wrist_x, r_wrist_y)]
            # Apply smoothing to the joint positions
            for joint, (joint_x, joint_y) in enumerate(joints):
                if len(prev_joint_positions) > 0:
                    smoothed_x = int(smoothing_factor * joint_x + (1 - smoothing_factor) * prev_joint_positions[joint][0])
                    smoothed_y = int(smoothing_factor * joint_y + (1 - smoothing_factor) * prev_joint_positions[joint][1])
                else:
                    smoothed_x, smoothed_y = joint_x, joint_y
                # Store the current smoothed joint positions
                current_joint_positions.append((smoothed_x, smoothed_y))

            
            # set for the next frame
            prev_joint_positions = current_joint_positions
            cv2.circle(img, (r_shldr_x, r_shldr_y), 7, yellow, -1)
            cv2.circle(img, (r_elbow_x, r_elbow_y), 7, yellow, -1)
            cv2.circle(img, (r_wrist_x, r_wrist_y), 7, yellow, -1)
            
            for joint_position in (current_joint_positions):
                joint_x, joint_y = joint_position
                cv2.circle(img, (joint_x, joint_y), 7, blue, -1)


            # During calibration period
            if (time.time() - calibration_start_time < 3):
                if r_elbow_y != 0:
                    calibration_joint_values_x.append(r_elbow_x)
                    calibration_joint_values_y.append(r_elbow_y)
            # Not calibration period
            else:
                calibrated_elbow_x = np.mean(calibration_joint_values_x)
                calibrated_elbow_y = np.mean(calibration_joint_values_y)
                box_x = int(calibrated_elbow_x - 50)
                box_y = int(calibrated_elbow_y - 75)
                cv2.rectangle(img, (box_x, box_y), (box_x + 100, box_y + 150), (255, 255, 255), 2)
                cv2.line(img, (0, int(calibrated_elbow_y) - 20), (w, int(calibrated_elbow_y) - 20), (255,0,0), 2)

                if time.time() - cooldown_start_time > cooldown_period:
                    if r_wrist_y > int(calibrated_elbow_y) - 20:
                        reps += 1
                        cooldown_start_time = time.time()
                if time.time() - cooldown_start_time_err > cooldown_period:
                    # Check for movement of our desired joint
                    if (r_elbow_x > calibrated_elbow_x + 50) or (r_elbow_x < calibrated_elbow_x - 50):
                        errCounterX += 1
                        data_to_publish = "Error from X position"
                        client.publish("TurnerOpenCV", data_to_publish)
                        cooldown_start_time_err = time.time()
                    if (r_elbow_y > calibrated_elbow_y + 75) or (r_elbow_y < calibrated_elbow_y - 75):
                        errCounterY += 1
                        data_to_publish = "Error from Y position"
                        client.publish("TurnerOpenCV", data_to_publish)
                        cooldown_start_time_err = time.time()

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            if breakflag:
                counter += 1

    ## Pushup code
    if( exercise_flag == 2):
        blue = (255, 127, 0)
        red = (50, 50, 255)
        green = (127, 255, 0)
        dark_blue = (127, 20, 0)
        light_green = (127, 233, 100)
        yellow = (0, 255, 255)
        pink = (255, 0, 255)
        # Set all time relevant values after we recieve the start command so that these dont begin at the wrong time and mess everything up
        cooldown_start_time = time.time()
        calibration_start_time = time.time()
        # This is variables to stop processing
        breakflag = False
        counter = 0
        # This is your actual processing code
        while True:
            if time.time()-code_start > 5 and counter == 0:
                Pause=False
                counter+=1
            if counter >= 31:
                client.publish("Control", "Workout Complete!")
                sleep(1)
                workout_complete_event.set()
                break
            if reps >= 10 and counter == 1:
                breakflag = True
        
            # Read frame
            success, img = cap.read()

            # rgb for skeleton
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            # Clear previous joint positions for the new frame
            current_joint_positions = []

            # Following line is the original skeleton drawing, but we dont really need the connections drawn or the points due to later code
        
            lm = results.pose_landmarks
            lmPose = mpPose.PoseLandmark
            h, w, c = img.shape

            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

            # Right elbow
            r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
            r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
            
            joints = [(r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y)]
            # Apply smoothing to the joint positions
            for joint, (joint_x, joint_y) in enumerate(joints):
                if len(prev_joint_positions) > 0:
                    smoothed_x = int(smoothing_factor * joint_x + (1 - smoothing_factor) * prev_joint_positions[joint][0])
                    smoothed_y = int(smoothing_factor * joint_y + (1 - smoothing_factor) * prev_joint_positions[joint][1])
                else:
                    smoothed_x, smoothed_y = joint_x, joint_y
                # Store the current smoothed joint positions
                current_joint_positions.append((smoothed_x, smoothed_y))

            
            # set for the next frame
            prev_joint_positions = current_joint_positions
            cv2.circle(img, (r_shldr_x, r_shldr_y), 7, yellow, -1)
            cv2.circle(img, (r_elbow_x, r_elbow_y), 7, yellow, -1)
            
            for joint_position in (current_joint_positions):
                joint_x, joint_y = joint_position
                cv2.circle(img, (joint_x, joint_y), 7, blue, -1)


            # During calibration period
            if (time.time() - calibration_start_time < 3):
                if r_elbow_y != 0:
                    calibration_joint_values_x.append(r_elbow_x)
                    calibration_joint_values_y.append(r_elbow_y)
            # Not calibration period
            else:
                calibrated_elbow_x = np.mean(calibration_joint_values_x)
                calibrated_elbow_y = np.mean(calibration_joint_values_y)
                cv2.line(img, (0, int(calibrated_elbow_y) - 20), (w, int(calibrated_elbow_y) - 20), (255,0,0), 2)

                if time.time() - cooldown_start_time > 1.5:
                        Pause = False
                if time.time() - cooldown_start_time > 4: # only one rep per 4 seconds
                    if r_shldr_y > int(calibrated_elbow_y - 20): # check if shoulder joint passes line
                        client.publish("TurnerOpenCV", "pushup pause begin")
                        Pause = True
                        reps += 1
                        cooldown_start_time = time.time()

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            if breakflag:
                counter += 1

    ## Squat code
    if( exercise_flag == 3):
        blue = (255, 127, 0)
        red = (50, 50, 255)
        green = (127, 255, 0)
        dark_blue = (127, 20, 0)
        light_green = (127, 233, 100)
        yellow = (0, 255, 255)
        pink = (255, 0, 255)
        # Set all time relevant values after we recieve the start command so that these dont begin at the wrong time and mess everything up
        cooldown_start_time = time.time()
        calibration_start_time = time.time()
        # This is variables to stop processing
        breakflag = False
        counter = 0
        # This is your actual processing code
        while True:
            if time.time()-code_start > 5 and counter == 0:
                Pause=False
                counter+=1
            if counter >= 31:
                client.publish("Control", "Workout Complete!")
                sleep(1)
                workout_complete_event.set()
                break
            if reps >= 10 and counter == 1:
                breakflag = True
        
            # Read frame
            success, img = cap.read()

            # rgb for skeleton
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            # Clear previous joint positions for the new frame
            current_joint_positions = []

            # Following line is the original skeleton drawing, but we dont really need the connections drawn or the points due to later code
        
            lm = results.pose_landmarks
            lmPose = mpPose.PoseLandmark
            h, w, c = img.shape

            # right hip
            r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
            r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
            # Right knee
            r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
            r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
            
            joints = [(r_hip_x, r_hip_y), (r_knee_x, r_knee_y)]
            # Apply smoothing to the joint positions
            for joint, (joint_x, joint_y) in enumerate(joints):
                if len(prev_joint_positions) > 0:
                    smoothed_x = int(smoothing_factor * joint_x + (1 - smoothing_factor) * prev_joint_positions[joint][0])
                    smoothed_y = int(smoothing_factor * joint_y + (1 - smoothing_factor) * prev_joint_positions[joint][1])
                else:
                    smoothed_x, smoothed_y = joint_x, joint_y
                # Store the current smoothed joint positions
                current_joint_positions.append((smoothed_x, smoothed_y))

            
            # set for the next frame
            prev_joint_positions = current_joint_positions

            cv2.circle(img, (r_hip_x, r_hip_y), 7, yellow, -1)
            cv2.circle(img, (r_knee_x, r_knee_y), 7, yellow, -1)
            
            for joint_position in (current_joint_positions):
                joint_x, joint_y = joint_position
                cv2.circle(img, (joint_x, joint_y), 7, blue, -1)


            # During calibration period
            if (time.time() - calibration_start_time < 3):
                if r_knee_y != 0:
                    calibration_joint_values_x.append(r_knee_x)
                    calibration_joint_values_y.append(r_knee_y)
            # Not calibration period
            else:
                calibrated_knee_x = np.mean(calibration_joint_values_x)
                calibrated_knee_y = np.mean(calibration_joint_values_y)
                cv2.line(img, (0, int(calibrated_knee_y) - 50), (w, int(calibrated_knee_y) - 50), (255,0,0), 2)

                if time.time() - cooldown_start_time > 1.5:
                        Pause = False
                if time.time() - cooldown_start_time > 4: # only one rep per 4 seconds
                    if r_hip_y > int(calibrated_knee_y - 50): # check if shoulder joint passes line
                        client.publish("TurnerOpenCV", "squat pause begin")
                        Pause = True
                        reps += 1
                        cooldown_start_time = time.time()

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            if breakflag:
                counter += 1

    ## Leg Raise code
    if( exercise_flag == 4):
        blue = (255, 127, 0)
        red = (50, 50, 255)
        green = (127, 255, 0)
        dark_blue = (127, 20, 0)
        light_green = (127, 233, 100)
        yellow = (0, 255, 255)
        pink = (255, 0, 255)
        # Set all time relevant values after we recieve the start command so that these dont begin at the wrong time and mess everything up
        cooldown_start_time = time.time()
        calibration_start_time = time.time()
        errortime = time.time()
        # This is variables to stop processing
        breakflag = False
        counter = 0
        # variables for the line
        angle_deg = 40
        angle_rad = math.radians(angle_deg)
        line_len = 700
        # This is your actual processing code
        while True:
            if time.time()-code_start > 5 and counter == 0:
                Pause=False
                counter+=1
            if counter >= 31:
                client.publish("Control", "Workout Complete!")
                sleep(1)
                workout_complete_event.set()
                break
            if reps >= 10 and counter == 1:
                breakflag = True
            
            # Read frame
            success, img = cap.read()

            # rgb for skeleton
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            # Clear previous joint positions for the new frame
            current_joint_positions = []

            # Following line is the original skeleton drawing, but we dont really need the connections drawn or the points due to later code
        
            lm = results.pose_landmarks
            lmPose = mpPose.PoseLandmark
            h, w, c = img.shape

            # right hip
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
            # Right knee
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

            l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
            
            joints = [(l_hip_x, l_hip_y), (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y)]
            # Apply smoothing to the joint positions
            for joint, (joint_x, joint_y) in enumerate(joints):
                if len(prev_joint_positions) > 0:
                    smoothed_x = int(smoothing_factor * joint_x + (1 - smoothing_factor) * prev_joint_positions[joint][0])
                    smoothed_y = int(smoothing_factor * joint_y + (1 - smoothing_factor) * prev_joint_positions[joint][1])
                else:
                    smoothed_x, smoothed_y = joint_x, joint_y
                # Store the current smoothed joint positions
                current_joint_positions.append((smoothed_x, smoothed_y))

            
            # set for the next frame
            prev_joint_positions = current_joint_positions

            
            cv2.circle(img, (l_hip_x, l_hip_y), 7, yellow, -1)
            cv2.circle(img, (l_knee_x, l_knee_y), 7, yellow, -1)
            cv2.circle(img, (l_ankle_x, l_ankle_y), 7, yellow, -1)
            
            for joint_position in (current_joint_positions):
                joint_x, joint_y = joint_position
                cv2.circle(img, (joint_x, joint_y), 7, blue, -1)


            # During calibration period
            if (time.time() - calibration_start_time < 3):
                if l_hip_x != 0:
                    calibration_joint_values_x.append(l_hip_x)
                    calibration_joint_values_y.append(l_hip_y)
            # Not calibration period
            else:
                calibrated_hip_x = np.mean(calibration_joint_values_x)
                calibrated_hip_y = np.mean(calibration_joint_values_y)
                end_x = int(calibrated_hip_x - line_len * math.cos(angle_rad))
                end_y = int(calibrated_hip_y - line_len * math.sin(angle_rad))
                cv2.line(img, (int(calibrated_hip_x), int(calibrated_hip_y)), (end_x, end_y), red, 2)
                y_line_at_point = calibrated_hip_y + int((l_knee_x - calibrated_hip_x) * math.tan(angle_rad))
                knee_inclination = findAngle(l_knee_x, l_knee_y, l_hip_x, l_hip_y)
                ankle_inclination = findAngle(l_knee_x, l_knee_y, l_ankle_x, l_ankle_y)

                if time.time() - cooldown_start_time > 1.5:
                        Pause = False
                # Count reps
                if time.time() - cooldown_start_time > 4: # only one rep per 4 seconds
                    if l_knee_y < y_line_at_point:
                        reps += 1
                        Pause = True
                        cooldown_start_time = time.time()
                if Pause == True:
                    if l_knee_y > y_line_at_point:
                        if time.time() - errortime > 1.5:
                            errors += 1
                            errortime = time.time()
                        


            # Display
            resized_img = cv2.resize(img, (1300, 700))
            #cv2.imshow('MediaPipe Pose', resized_img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            if breakflag:
                counter += 1

    # plank code
    if( exercise_flag == 5):
        
        blue = (255, 127, 0)
        red = (50, 50, 255)
        green = (127, 255, 0)
        dark_blue = (127, 20, 0)
        light_green = (127, 233, 100)
        yellow = (0, 255, 255)
        pink = (255, 0, 255)
        good_frames = 0
        bad_frames = 0
        fps = 0
        Pause = False
        start_time = time.time()
        # This is your actual processing code
        while True:
            if good_time > 60:
                sleep(1)
                workout_complete_event.set()
                break
            
            # Read frame
            success, img = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            # rgb for skeleton
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks is None:
                continue  # Skip to the next iteration if no person is detected
            # Clear previous joint positions for the new frame
            current_joint_positions = []

            # Following line is the original skeleton drawing, but we dont really need the connections drawn or the points due to later code
        
            lm = results.pose_landmarks
            lmPose = mpPose.PoseLandmark
            h, w, c = img.shape

            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

            # right hip
            r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
            r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
            # Right knee
            r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
            r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
            
            joints = [(r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), (r_knee_x, r_knee_y)]
            # Apply smoothing to the joint positions
            for joint, (joint_x, joint_y) in enumerate(joints):
                if len(prev_joint_positions) > 0:
                    smoothed_x = int(smoothing_factor * joint_x + (1 - smoothing_factor) * prev_joint_positions[joint][0])
                    smoothed_y = int(smoothing_factor * joint_y + (1 - smoothing_factor) * prev_joint_positions[joint][1])
                else:
                    smoothed_x, smoothed_y = joint_x, joint_y
                # Store the current smoothed joint positions
                current_joint_positions.append((smoothed_x, smoothed_y))

            
            # set for the next frame
            prev_joint_positions = current_joint_positions

            cv2.circle(img, (r_shldr_x, r_shldr_y), 7, yellow, -1)
            cv2.circle(img, (r_hip_x, r_hip_y), 7, yellow, -1)
            cv2.circle(img, (r_knee_x, r_knee_y), 7, yellow, -1)
            
            for joint_position in (current_joint_positions):
                joint_x, joint_y = joint_position
                cv2.circle(img, (joint_x, joint_y), 7, blue, -1)


            # calculate angles
            
            knee_inclination = findAngle(r_knee_x, r_knee_y, r_hip_x, r_hip_y)
            torso_inclination = findAngle(r_hip_x, r_hip_y, r_shldr_x, r_shldr_y)
            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Knee : ' + str(int(knee_inclination)) + '  Torso : ' + str(int(torso_inclination))

            # calculate diff
            angle_diff = np.abs(knee_inclination - torso_inclination)
            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if angle_diff < 5:
                good_frames += 1
                
                cv2.putText(img, angle_text_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, light_green, 2)
                cv2.putText(img, str(int(knee_inclination)), (r_shldr_x + 10, r_shldr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, light_green, 2)
                cv2.putText(img, str(int(torso_inclination)), (r_hip_x + 10, r_hip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, light_green, 2)

                # Join landmarks.
                cv2.line(img, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), green, 4)
                cv2.line(img, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), green, 4)

            else:
                bad_frames += 1

                cv2.putText(img, angle_text_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, light_green, 2)
                cv2.putText(img, str(int(knee_inclination)), (r_shldr_x + 10, r_shldr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, light_green, 2)
                cv2.putText(img, str(int(torso_inclination)), (r_hip_x + 10, r_hip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, light_green, 2)

                # Join landmarks.
                cv2.line(img, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), green, 4)
                cv2.line(img, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), green, 4)

            # Calculate the time of remaining in a particular posture.
            if fps != 0:
                good_time = (1 / fps) * good_frames
                bad_time =  (1 / fps) * bad_frames

                good_time = round(good_time, 1)
                bad_time = round(bad_time, 1)

            # Pose time.
            if good_time > 0:
                time_string_good = 'Proper Plank time : ' + str(good_time) + 's'
                cv2.putText(img, time_string_good, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2)
                time_string_bad = 'Bad Plank Time : ' + str(bad_time) + 's'
                cv2.putText(img, time_string_bad, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX,  0.9, red, 2)

            # Display
            resized_img = cv2.resize(img, (1300, 700))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime


    cap.release()
    cv2.destroyAllWindows()

# Name functions for skeleton usage
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Get video directly from camera
cap = cv2.VideoCapture(0)
pTime = 0

## Initialize variables used for OpenCV
# Initialize empty list to store previous joint positions
prev_joint_positions = []
# Smoothing factor (adjust as needed) (for skeleton)
smoothing_factor = 0.5
# Initialize a list to store pabove values during calibration
calibration_pabove_values = []
# Calibration period (in seconds)
calibration_period = 5  # Adjust as needed
# Rep counter 
reps = 0
# Err Counter
errCounterX = 0
errCounterY = 0
errPause = 0
errors = 0
# Cooldown start values for counting reps
cooldown_period = 4  # Cooldown period (in seconds)
# Init lists to store desired joint positional values during calibration
calibration_joint_values_x = []
calibration_joint_values_y = []
# int to store id of desired joint
desiredJoint = 0
Joint_x = 0
Joint_y = 0
Shoulder_x = 0
Shoulder_y = 0
hip_x = 0
hip_y = 0

good_time = 0
bad_time = 0

Pause = True

## MQTT functionality
# MQTT callback functions
def on_connect(client, userdata, flags, rc):
   global flag_connected
   flag_connected = 1
   print("Connected to MQTT server")

def on_disconnect(client, userdata, rc):
   global flag_connected
   flag_connected = 0
   print("Disconnected from MQTT server")

def user_callback(client, userdata, msg):
    # When we recieve the notification that the bottom of the pushup is reached
    # Ask esp32 callback to look for no movement for one second
    global errors
    converted_msg = str(msg.payload.decode('utf-8'))
    print('OpenCV message: ', converted_msg)
    if converted_msg == "true error":
        errors += 1
    if converted_msg == "Workout Complete!":
        exit()

def mqtt_thread():
    # MQTT subscription and message processing logic here
    client.subscribe("Control")
    client.message_callback_add('Control', user_callback)


client = mqtt.Client("openCV client") # this should be a unique name
flag_connected = 0

client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.connect('192.168.99.113',1883)
#client.connect('131.179.39.148',1883)
# start a new thread
client.loop_start()
print("......client setup complete............")

## Speech Recognition
exercise_flag = 0

def ask_Exercise():
    global exercise_flag
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please state your desired exercise.\n")
        print("Options: Bicep curl, Pushup, Squat, Leg Raise, Plank\n")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        if "bicep curl" in command:
            print("Bicep curl chosen\n")
            exercise_flag = 1
            return True
        if "push-up" in command:
            print("Pushup chosen\n")
            exercise_flag = 2
            return True
        if "push up" in command:
            print("Pushup chosen\n")
            exercise_flag = 2
            return True
        if "squat" in command:
            print("Squat chosen\n")
            exercise_flag = 3
            return True
        if "leg raise" in command:
            print("Leg Raise chosen\n")
            exercise_flag = 4
            return True
        if "plank" in command:
            print("Plank chosen\n")
            exercise_flag = 5
            return True
        else:
            print("The exercise you requested was: " + command)
            print("This exercise was not recognized. Please choose one of the options stated.\n")
            return False
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")
        return False
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return False

def listen_for_start_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say 'start' to begin the program.")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        if "start" in command:
            print("Start command recognized. Starting the program.")
            client.publish("Control", exercise_flag)
            client.publish("Control", "Start")
            return True
        else:
            print(command)
            print("Start command not recognized. Please say 'start' to begin.")
            return False
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")
        return False
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return False

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / math.pi) * theta
    return degree
# These will continualy prompt the user until they pass the prompts
# Ask for excercise
while not ask_Exercise():
    pass
# Ask for prompt to start the execerise program
while not listen_for_start_command():
    pass

if __name__ == "__main__":
    code_start = time.time()
    app = ctk.CTk()
    bicep_curl_app = BicepCurlApp(app)

    # Create a new thread for running your additional code
    code_thread = threading.Thread(target=my_code, args=(bicep_curl_app.workout_complete_event,))
    code_thread.start()

    # Create a new thread for MQTT
    mqtt_thread = threading.Thread(target=mqtt_thread)
    mqtt_thread.start()

    # Run the GUI main loop
    bicep_curl_app.run()