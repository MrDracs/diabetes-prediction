# Diabetes Checkup – Clinical Decision Support

A Streamlit web application that predicts diabetes risk using machine learning based on patient clinical data.

## Features

- **Patient Data Input**: Interactive sidebar for entering patient information
- **Risk Assessment**: Predicts high or low diabetes risk with probability score
- **Interactive Visualizations**:
  - HbA1c vs Blood Glucose scatter plot with population context
  - BMI vs Predicted Risk trend line
  - Feature importance chart showing top risk factors
- **Model Accuracy**: Displays Random Forest classifier accuracy on test data
- **Clinical Context**: Population-level statistics for comparison

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## Installation

1. Clone or download this repository
2. Navigate to the project folder:
   ```bash
   cd "c:\Users\Supriya\Downloads\GITHUB\prediction model"
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data

- **File**: `diabetes_dataset.csv`
- **Required Columns**:
  - `age` (numeric)
  - `bmi` (numeric)
  - `HbA1c_level` (numeric)
  - `blood_glucose_level` (numeric)
  - `gender` (categorical)
  - `smoking_history` (categorical)
  - `hypertension` (0 or 1)
  - `heart_disease` (0 or 1)
  - `diabetes` (target: 0 or 1)

## Usage

Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use:
1. Enter patient data in the left sidebar
2. View training data statistics
3. Check the Risk Assessment result
4. Review visualizations for clinical context
5. View model accuracy and feature importance

## Model Details

- **Algorithm**: Random Forest Classifier
- **Train/Test Split**: 80/20
- **Features**: Age, BMI, HbA1c level, blood glucose, hypertension, heart disease, gender, smoking history

## Output

- **Risk Score**: Probability of diabetes (0.0 - 1.0)
- **Classification**: High Risk or Low Risk
- **Feature Importance**: Top factors contributing to prediction
- **Model Accuracy**: Percentage accuracy on test dataset

## Disclaimer

This application is designed as a clinical decision support tool. It should not be used as a substitute for professional medical diagnosis. Always consult with healthcare professionals for medical advice.

## Author

Created: February 2026