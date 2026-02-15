# 🔬 AI-Powered Histopathologic Cancer Detection

An end-to-end deep learning project to detect metastatic cancer in histopathologic scans of lymph node tissue. This application is built with TensorFlow and deployed as a user-friendly web interface using Streamlit.

---

## 📜 Project Overview

This project addresses the challenge of identifying metastatic cancer in small image patches taken from larger digital pathology scans. The goal is to build an accurate and efficient model that can classify a given tissue sample as either containing cancerous cells (Malignant) or not (Benign).

The core of this project is a **Convolutional Neural Network (CNN)**, utilizing transfer learning with the MobileNetV2 architecture, trained on the **PatchCamelyon (PCam)** dataset. The final model is wrapped in a Streamlit web app where users can upload an image and receive a real-time prediction and a detailed analysis report.


## ✨ Key Features

- **Deep Learning Model:** A robust CNN fine-tuned for high accuracy in histopathologic image classification.
- **Interactive Web UI:** A clean and intuitive user interface built with Streamlit for easy image uploads.
- **Real-Time Predictions:** Fast analysis of tissue samples.
- **Detailed Findings Report:** Provides a prediction, confidence score, and interpretation of the results.
- **Scalable and Modular Code:** The project is structured to be easily understood, maintained, and extended.

---

## 💻 Tech Stack

- **Backend & Modeling:** Python, TensorFlow, Keras
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy, OpenCV
- **Version Control:** Git & GitHub

---

## 🚀 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/suryasatwikboddepalli-a11y/Histopathologic-Cancer-Detection-using-Deep-Learning](https://github.com/suryasatwikboddepalli-a11y/Histopathologic-Cancer-Detection-using-Deep-Learning)
    cd Cancer-Detection-App
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Dataset (for training):**
    - The model was trained on the [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/data) dataset from Kaggle.
    - You will need to download and organize it to run the `train_model.py` script.

4.  **Run the application:**
    - Make sure the pre-trained `histopathology_model.h5` file is in the root directory.
    - Start the Streamlit server:
    ```bash
    streamlit run app.py
    ```

---

## (Usage)

Once the application is running, open your web browser and navigate to the local URL provided by Streamlit. Upload a 96x96 pixel histopathologic image (`.jpg`, `.png`, or `.tif`) to receive an instant analysis.

---

## ⚠️ Disclaimer

This project is an educational tool and is **not intended for actual clinical or diagnostic use**. The predictions made by the AI are for informational purposes only and should never be used as a substitute for a diagnosis by a qualified medical professional.
