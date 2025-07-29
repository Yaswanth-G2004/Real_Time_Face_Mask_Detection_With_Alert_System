# 🎭 Real-Time Face Mask Detection with Alert System

A real-time face mask detection system using a pre-trained deep learning model. It detects whether a person is wearing a mask or not through the webcam and plays an audio alert if no mask is detected.

## 🔍 About the Project

The main objective of this project is to build a real-time system that:
- Detects faces from a live webcam feed.
- Classifies each detected face as **"Mask"** or **"No Mask"**.
- Plays an **alert sound** when a person is **not wearing a mask**.

This solution is ideal for use at entrances of offices, public places, or events where monitoring mask compliance is required.

## 📂 Dataset Used

We used the **Face Mask Detection dataset** by [Andrew Mvd on Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), which includes:
- Images with people wearing masks
- Images with people not wearing masks


## 🚀 Features

- Real-time webcam feed detection  
- Alerts via sound when "No Mask" is detected  
- Uses OpenCV and Keras/TensorFlow  
- Lightweight and easy to use

## 🧠 Technologies Used

- Python  
- OpenCV  
- Keras  
- TensorFlow  
- NumPy  
- playsound (for alerts)

## 📦 Installation

```bash
pip install opencv-python
pip install tensorflow
pip install keras
pip install numpy
pip install playsound
```

## 🔊 Alert System

- When a person is **not wearing a mask**, an audio alert (`mixkit-appliance-ready-beep-1076`) will play.
- You can replace `mixkit-appliance-ready-beep-1076` with any custom sound.

## 📌 Model

This project uses a pre-trained classification model (`mask_detector.model`) trained on the Kaggle Face Mask Detection Dataset.

## 👨‍💻 Author

**Garikipati Yaswanth**  
📧 Email: garikipatiyaswanth2004@gmail.com  
🔗 [GitHub](https://github.com/Yaswanth-G2004)  
🔗 [LinkedIn](https://www.linkedin.com/in/yaswanth-garikipati-516821288)

## 📄 License

This project is licensed under the **MIT License** – free to use, modify, and distribute.
