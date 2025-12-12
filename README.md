# 🧬 MammoSegment AI

**Automated Pectoral Muscle Removal & Breast Tissue Segmentation in MLO Mammograms**

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📌 Overview
**MammoSegment AI** is a biomedical image processing tool designed to automate the segmentation of breast tissue from MLO (Mediolateral Oblique) mammograms. The core challenge in MLO views is the similarity in intensity between the pectoral muscle and the glandular tissue. This project utilizes a geometry-aware hybrid approach to effectively isolate the breast region.

## 🚀 Features
* **Preprocessing:** CLAHE (Contrast Limited Adaptive Histogram Equalization) and Median Filtering for noise reduction.
* **Segmentation:** Otsu's Thresholding for initial binary masking.
* **Muscle Removal:** Advanced geometric approach using **Hough Line Transform** to detect and remove the pectoral muscle boundary.
* **Interactive UI:** Built with Streamlit for real-time analysis and visualization.
* **Report Viewer:** Integrated PDF viewer to access the full IEEE technical report.

## 🛠️ Tech Stack
* **Python**
* **Streamlit** (Web Interface)
* **OpenCV** (Image Processing)
* **NumPy** (Matrix Operations)

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/berkantaksyy/MammoSegment-AI.git](https://github.com/berkantaksyy/MammoSegment-AI.git)
   cd MammoSegment-AI


2. **Install dependencies::**
```bash
pip install -r requirements.txt
```

4. **Run the app:::**
```bash
streamlit run app.py
```

📂 Project Structure
```Plaintext
MammoSegment-AI/
│
├── app.py                       # Main Streamlit application file
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── 210402043_final_project.pdf  # IEEE Format Project Report
└── background.mp4               # UI Background Video
```




📄 Technical Report
This project includes a detailed academic report written in IEEE conference format. It covers the mathematical background, algorithm logic, and performance metrics (Dice Score, Jaccard Index).



Supervisor: Asst. Prof. Dr. Onan GÜREN
