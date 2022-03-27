import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from pickle import dump
from SPO2_Pred import CalKa

DATADIR = "E:/20211/ĐATN/Code/SPO2/data/data_spo2.csv"
all_data = pd.read_csv(DATADIR)

feature = all_data.columns[1]
label = all_data.columns[0]

X_train = []
y_train = all_data[label]
x = [i for i in range(len(y_train))]

video_files = all_data[feature]

print("Start Training")

for video_file in video_files:
    print("video: ", video_file)
    print("Frame count:")
    cap = cv2.VideoCapture(video_file)
    Ka_list = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if count % 30 == 0:
            print(count)
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        Kas = CalKa(frame)
        if len(Kas) > 0:
            for (Ka, _) in Kas:
                Ka_list.append(Ka)

    X_train.append(np.mean(Ka_list))

x = X_train.copy()
print(x, y_train)
print(len(x))
X_train = pd.DataFrame(X_train)

print("Done!!!")

model = LinearRegression().fit(X_train, y_train)
dump(model, open('save_model.pkl', 'wb'))

A, B = model.intercept_, model.coef_
print("Parameters A, B:", A, B)
print("MSE:", mse(y_train, model.predict(X_train)))

x0 = np.linspace(int(min(x)), int(max(x))+1)
y0 = A + B*x0
plt.plot(x0, y0, 'b-')
plt.plot(x, y_train, 'ro')
plt.xlabel("Ti le hap thu")
plt.ylabel("SpO2")
plt.xlim(int(min(x)), int(max(x))+1)
plt.ylim(70, 100)
plt.title("SpO2")
plt.show()