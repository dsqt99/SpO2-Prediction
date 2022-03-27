import cv2
import numpy as np
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
blue, green, red, yellow, purple = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX


def CalKa(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    Kas = []

    for face in faces:
        # Get the Cheek ROI of Face
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        start_face = (face.left(), face.top())
        end_face = (face.right(), face.bottom())
        start_cheekl = (landmarks[4][0], landmarks[29][1])
        end_cheekl = (landmarks[48][0], landmarks[33][1])
        start_cheekr = (landmarks[54][0], landmarks[29][1])
        end_cheekr = (landmarks[12][0], landmarks[33][1])

        cv2.rectangle(frame, start_face, end_face, green, 2)
        cv2.rectangle(frame, start_cheekl, end_cheekl, green, 1)
        cv2.rectangle(frame, start_cheekr, end_cheekr, green, 1)

        Ka = []

        # Calculate Ka left cheek
        image = frame[start_cheekl[1]:end_cheekl[1], start_cheekl[0]:end_cheekl[0]]
        (B, G, R) = cv2.split(image)
        DCB, ACB, DCR, ACR = np.mean(B), np.std(B), np.mean(R), np.std(R)
        Ka.append((ACR / DCR) / (ACB / DCB))

        # Calculate Ka right cheek
        image = frame[start_cheekr[1]:end_cheekr[1], start_cheekr[0]:end_cheekr[0]]
        (B, G, R) = cv2.split(image)
        DCB, ACB, DCR, ACR = np.mean(B), np.std(B), np.mean(R), np.std(R)
        
        Ka.append((ACR / DCR) / (ACB / DCB))

        if len(Ka) > 0:
            Kas.append([np.mean(Ka), start_face])

    return Kas


def Puttext(frame, spo2, point):
    org = point
    fontScale = 1
    spo2 = round(spo2, 0)
    if spo2 > 96:
        color = green
        text = str(spo2) + " Good"
    elif spo2 >= 94:
        color = yellow
        text = str(spo2) + " Normal"
    elif spo2 >= 90:
        color = purple
        text = str(spo2) + " Bad"
    else:
        color = red
        text = str(spo2) + " Emergency"
    thickness = 1
    cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


def CalSpo2(video_file):
    cap = cv2.VideoCapture(video_file)
    spo2_list = []
    frame_count = 0

    print("Start to calculate SpO2")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_count += 1
        if not ret:
            break

        if frame_count % 30 == 0:
            print(frame_count)

        Kas = CalKa(frame)
        for (Ka, start_face) in Kas:
            spo2 = 86.6 + 11.5 * Ka
            spo2_list.append(spo2)

    spo2 = int(round(np.mean(spo2_list), 0))

    print("Done !!!")
    return spo2


def CalSpo2_realtime():
    cap = cv2.VideoCapture(0)
    spo2_list, frame_list = [], []
    frame_count = 0

    fig = plt.figure()
    line, = plt.plot(frame_list, spo2_list, 'b-')
    plt.xlabel("frame")
    plt.ylabel("SPO2")
    plt.ylim([0, 100])
    fig.canvas.draw()

    while True:
        ret, frame = cap.read()
        Kas = CalKa(frame)
        if len(Kas) == 0:
            text = "No faces in frame"
            point = (frame.shape[1]//2, frame.shape[0]//2)
            cv2.putText(frame, text, point, font, 1, red, 2, cv2.LINE_AA)
        for (Ka, start_face) in Kas:
            spo2_cal = 86.6 + 11.5 * Ka

            spo2_list.append(spo2_cal)
            frame_list.append(frame_count)
            spo2_list = spo2_list[-300:]
            frame_list = frame_list[-300:]
            frame_count += 1

            if len(spo2_list) == 300:
                spo2_avr = round(np.mean(spo2_list), 0)
                Puttext(frame, spo2_avr, start_face)
            else:
                Puttext(frame, spo2_cal, start_face)

            line.set_data(frame_list, spo2_list)
            plt.xlim([frame_list[0], frame_list[0]+300])
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("SPO2 realtime", img)

        cv2.imshow("out", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("SPO2:", round(np.mean(spo2_list), 0))
            break