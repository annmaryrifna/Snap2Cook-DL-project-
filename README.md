# Snap2Cook – Food Ingredient Recognition and Recipe Recommendation System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![ResNet50](https://img.shields.io/badge/ResNet50-CNN-red)
![AWS](https://img.shields.io/badge/AWS-Cloud-yellow)

AI-powered food ingredient recognition and recipe recommendation web application using deep learning and transfer learning techniques.

---

## ✨ Features

- **Ingredient Recognition**
  - Ingredient image classification using ResNet50
  - Deep feature extraction with transfer learning
  - High prediction accuracy

- **Recipe Recommendation**
  - Content-based recipe recommendation
  - Ingredient-to-recipe matching
  - Intelligent food suggestion system

- **Web Application**
  - Responsive user interface
  - Flask backend integration
  - Secure database connectivity

- **Cloud Deployment**
  - AWS EC2 hosting
  - MySQL database integration
  - GitHub version control

---

##  Project Overview

Snap2Cook is a deep learning-based web application that identifies food ingredients from uploaded images and recommends suitable recipes automatically.

The system uses the pretrained ResNet50 convolutional neural network model with transfer learning for accurate ingredient classification. The web application is developed using Flask, integrated with MySQL for database management, and deployed using AWS cloud services.

The application simplifies ingredient identification and helps users discover recipes efficiently using artificial intelligence techniques.

---

## 🔄 Workflow of the System

1. User uploads an ingredient image through the web application.
2. Image preprocessing and resizing (224×224) are performed.
3. ResNet50 extracts deep image features.
4. The trained model predicts the ingredient category.
5. The recommendation system suggests related recipes.
6. Results are displayed through the web interface.

---

##  Deep Learning Model

The project uses the pretrained ResNet50 CNN architecture with transfer learning.

### Model Architecture

- Input Image Size: `224×224×3`
- Pretrained Base Model: `ResNet50`
- Global Average Pooling Layer
- Dropout Layer `(0.5)`
- Dense Softmax Output Layer

<p align="center">
  <img src="images/architecture.png" width="500">
</p>

---

## 🛠️ Technologies Used

- Python
- Flask
- TensorFlow
- Keras
- ResNet50
- OpenCV
- NumPy
- Pandas
- HTML
- CSS
- JavaScript
- MySQL
- AWS EC2
- AWS RDS
- GitHub

---

##  Training and Performance

### Model Performance

- Training Accuracy: **98%**
- Validation Accuracy: **94%**

The model achieved high classification performance using transfer learning and deep feature extraction techniques.

---

##  Recipe Recommendation System

The recipe recommendation module suggests recipes based on predicted ingredients using content-based filtering techniques.

### Recommendation Process

- Ingredient prediction
- Ingredient matching
- Recipe filtering
- Recipe recommendation generation

---

##  AWS Cloud Deployment

The project is deployed using AWS cloud services for scalability and accessibility.

### AWS Services Used

- AWS EC2 for application hosting
- AWS RDS for MySQL database management
- GitHub for version control and deployment updates

---

## 📁 Project Structure

```text
Project/
│
├── model/                   # Saved trained models
├── static/                  # CSS, JS, and static assets
├── templates/               # HTML templates
├── app.py                   # Main Flask application
├── requirements.txt         # Python dependencies
└── README.md
```

---

##  Prerequisites

- Python 3.10+
- pip
- MySQL
- AWS Account (Optional for deployment)

---

##  Installation

### Clone the Repository

```bash
git clone https://github.com/annmaryrifna/Snap2Cook-DL-project-.git
cd Project
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

### Configure Database

Create a MySQL database and update database credentials in the Flask configuration.

---

##  Running the Application

```bash
python app.py
```

Access the application at:

```text
http://127.0.0.1:5000
```

---

##  Backend & Database Integration

The application integrates:

- Flask backend for request handling
- MySQL database for recipe and user data management
- Deep learning model for ingredient classification

---

##  Future Enhancements

- Real-time ingredient detection using camera
- Multi-ingredient recognition from single image
- Mobile application support
- Voice assistant integration
- Personalized recipe recommendation

---
