# 🎭 Real-Time Face Mask Detection with Alert System

A real-time face mask detection system using a pre-trained deep learning model. It detects whether a person is wearing a mask or not through the webcam and plays an audio alert if no mask is detected.

## 📂 Dataset

Used the open-source [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) from Kaggle. It contains images of people with and without masks.

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

## ▶️ How to Use

```bash
python detect_mask_video.py
```

Make sure your webcam is connected.

## 🔊 Alert System

- When a person is **not wearing a mask**, an audio alert (`alert.mp3`) will play.
- You can replace `alert.mp3` with any custom sound.

## 📌 Model

This project uses a pre-trained classification model (`mask_detector.model`) trained on the Kaggle Face Mask Detection Dataset.

## 🎓 Author

**Garikipati Yaswanth**  
B.Tech CSE | 3rd Year  
[GitHub Profile](https://github.com/Yaswanth-G2004)

## 📄 License

This project is licensed under the **MIT License** – free to use, modify, and distribute.
