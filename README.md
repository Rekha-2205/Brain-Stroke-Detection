# Brain Stroke Identification & Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2-green.svg)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://wwww.tensorflow.org/)

An advanced medical diagnostic model designed to identify and classify brain strokes from neuroimages (CT Scans) using a hybrid Deep Learning architecture of **ResNet-50**, **Bidirectional LSTM**, and **Genetic Algorithms**.

---

##  Overview
Stroke is a life-threatening medical emergency. Early and accurate detection is critical for patient survival. This project presents a state-of-the-art diagnostic system that leverages Transfer Learning and Evolutionary Algorithms to achieve high precision in stroke identification.

The system features a complete **Django-based Web Portal** where medical professionals can:
- Upload and analyze CT scans in real-time.
- Manage datasets for continuous model improvement.
- Monitor training progress via a live dashboard.
- Generate detailed diagnostic reports.

---

##  Key Features

- **Hybrid AI Architecture**: Combines ResNet-50 (spatial features) with BiLSTM (sequential dependencies) and Genetic Algorithms (hyperparameter optimization).
- **Advanced Preprocessing**: Implementation of CLAHE, Non-local Means Denoising, and Bilateral Filtering to enhance stroke visibility in low-contrast CT scans.
- **Real-Time Admin Dashboard**:
    - **Live Training Monitor**: Track model training progress with a real-time progress bar.
    - **Performance Analytics**: Compare accuracy, precision, recall, and AUC-ROC across multiple model versions.
    - **System Logs**: Comprehensive logging for security and audit trails.
- **Reporting System**: Automated generation of PDF reports for patient diagnostics.
- **Data Augmentation**: Medical-grade augmentation (elastic deformations, grid distortions) to improve model generalization.

---

##  Model Architecture

The core diagnostic engine follows a sophisticated pipeline:

1.  **Input**: Grayscale CT Scans (224x224).
2.  **Feature Extraction**: **ResNet-50** backbone (pretrained on ImageNet) for deep hierarchical feature extraction.
3.  **Sequence Modeling**: **Bidirectional LSTM** to capture bilateral symmetries and sequential patterns in neuroimages.
4.  **Optimization**: **Genetic Algorithms (GA)** used for finding the optimal weights and architecture parameters.
5.  **Classification**: Fully connected layers with **Focal Loss** to handle class imbalance (Stroke vs. Non-Stroke).

---

## Performance Results

The system significantly outperforms standard CNN architectures, achieving:

| Metric | Accuracy | Precision | Recall | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| **Proposed System** | **97.5%** | **96.2%** | **98.8%** | **0.99** |
| Standard CNN | 58.5% | 55.1% | 61.2% | 0.65 |

---

##  Technology Stack

- **Backend**: Python 3, Django 4.2
- **Deep Learning**: TensorFlow 2.13, Keras
- **Image Processing**: OpenCV, Scikit-Image, Albumentations
- **Database**: MySQL
- **Visualization**: Matplotlib, Seaborn
-

---

##  Project Structure

```text
Stroke_Project/
├── accounts/               # User authentication & profile management
├── admin_panel/            # Real-time training & performance dashboard
├── detection/              # Stroke detection core logic & UI
├── ml_models/              # AI Architecture (ResNet50, BiLSTM, GA)
│   ├── preprocessing.py    # CLAHE & Denoising filters
│   ├── enhanced_model.py   # Hybrid architecture logic
│   └── trained_models/     # Saved .h5 model files
├── media/                  # Uploaded scans & generated reports
├── static/                 # CSS, JS, and UI assets
├── manage.py               # Django entry point
└── requirements.txt        # Project dependencies
```

---

##  Installation & Setup

### 1. Prerequisites
- Python 3.8 or 3.9 (Recommended)
- MySQL Server

### 2. Clone the Repository
```bash
git clone https://github.com/Rekha-2205/Brain-Stroke-Detection.git
cd Brain-Stroke-Detection
```

### 3. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # Linux/Mac
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Database Configuration
Create a schema in MySQL named `stroke_detection_db` and update the `DATABASES` settings in `stroke_detection/settings.py` if necessary.

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Run the Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000` to access the portal.

---

##  Usage

1.  **Login**: Use admin credentials to access the dashboard.
2.  **Upload**: Navigate to the "Upload Scan" section to analyze a patient's CT scan.
3.  **Train**: Go to "Model Training" to train a new version of the AI using the latest datasets.
4.  **Results**: View detailed diagnostic overlays and metrics in the "Performance" tab.

---

##  Acknowledgments
- This project was developed as a B.Tech Final Year project.
- Special thanks to the researchers behind ResNet-50 and Genetic Algorithm-based optimizations in medical imaging.

---

**Author**: [Rekha](https://github.com/Rekha-2205)
