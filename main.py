from SPO2_Pred import CalSpo2, CalSpo2_realtime
from pickle import load

if __name__ == '__main__':

    # video_file = "E:/20211/ĐATN/Code/SPO2/data/2.mp4"
    # print("SpO2", CalSpo2(video_file))

    # model = load(open('save_model.pkl', 'rb'))
    # print(model.coef_, model.intercept_)

    CalSpo2_realtime()