# DeepPressNet: Depression Detection from Text using SBERT + Neural Networks

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-MLP-green)]()
[![Built with](https://img.shields.io/badge/built%20with-TensorFlow-blue)]()
[![Last Updated](https://img.shields.io/badge/last%20updated-July%202025-orange)]()

DeepPressNet is a deep learning pipeline that classifies Reddit posts as either "Depressed" or "Not Depressed" based on textual input. It leverages Sentence-BERT (SBERT) for embedding generation and a Multilayer Perceptron (MLP) neural network optimized using Optuna for classification.

### Live Link: https://huggingface.co/spaces/avdvh/DeepPressNet

## Features

- End-to-end NLP pipeline
- SBERT for powerful semantic embeddings
- Optuna-based hyperparameter tuning
- Evaluation with precision, recall, F1, ROC AUC, confusion matrix
- CSV logs and .txt reports for model training
- Real-time sentence prediction with confidence score
- Visualizations: training curves, metric comparisons


## Tech Stack

- SBERT (sentence-transformers)
- TensorFlow / Keras
- Optuna for hyperparameter optimization
- Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## Model Performance
<img width="579" height="272" alt="image" src="https://github.com/user-attachments/assets/52e35dbc-9119-4222-b087-bfda1e43bcff" />

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/9cba5fd6-0730-4944-8a6a-27ab90fe50f5" />
<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/b1c5be0d-74b4-4669-8408-1b48be9851e3" />

### After Optuna Fine-Tuning
<img width="719" height="235" alt="image" src="https://github.com/user-attachments/assets/96e2a394-6b87-4304-b19e-61a354f7f483" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/6d287740-f320-40b8-876b-0c04a467d003" />
<img width="1165" height="490" alt="image" src="https://github.com/user-attachments/assets/de4d78bc-06ff-417d-b547-fa310f5bc0d2" />



## Real-Time Prediction

<img width="1025" height="641" alt="image" src="https://github.com/user-attachments/assets/3f5f451d-aa29-4c3d-9aa3-f45e8cf5f4d4" />

