# Visit with Us – Wellness Tourism Package Purchase Prediction

## Project Overview

This project develops an end-to-end Machine Learning Operations (MLOps) solution for **Visit with Us**, a tourism company interested in predicting whether a customer is likely to purchase a Wellness Tourism Package.

The solution covers dataset registration, data preparation, model development, hyperparameter tuning, experiment tracking, model deployment, and automated CI/CD using GitHub Actions.

## Live Application

**Streamlit Application:**  
https://visit-with-us-tourism-mlops-kf7iwqhwl4cznjqyz236um.streamlit.app/

**GitHub Repository:**  
https://github.com/02vjiya/visit-with-us-tourism-mlops

## Business Objective

The objective is to build a machine learning model that predicts the likelihood of a customer purchasing a Wellness Tourism Package.

The prediction can help the business:
- Identify customers with a higher likelihood of purchasing.
- Improve customer targeting and engagement.
- Support data-driven marketing decisions.
- Optimize tourism package promotion strategies.

## Technology Stack

- **Programming Language:** Python 3.11
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Random Forest
- **Hyperparameter Tuning:** GridSearchCV
- **Experiment Tracking:** MLflow
- **Model Serialization:** Joblib
- **Application Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub
- **CI/CD Automation:** GitHub Actions

## Project Structure

```text
visit-with-us-tourism-mlops/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml
│
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv
│   │
│   ├── deployment/
│   │   ├── app.py
│   │   ├── model_metrics.json
│   │   ├── requirements.txt
│   │   └── tourism_model.joblib
│   │
│   └── model_building/
│       ├── data_register.py
│       ├── prep.py
│       └── train.py
│
└── Us_Tourism_MLOps.ipynb
```

## MLOps Workflow

### 1. Data Registration
The dataset is registered and made available to the subsequent data preparation stage.

### 2. Data Preparation
The data preparation pipeline performs the required preprocessing and generates training and testing datasets.

### 3. Model Building and Experiment Tracking
A Random Forest classification model is developed and tuned using GridSearchCV. Model evaluation metrics and experiment information are tracked using MLflow.

### 4. Model Deployment
The trained model is serialized and integrated into a Streamlit application. Users can enter customer information and receive a purchase prediction.

### 5. CI/CD with GitHub Actions
The GitHub Actions workflow automates the following stages:

1. Dataset registration
2. Data preparation
3. Model training and experiment tracking

The workflow also saves the trained model and evaluation metrics to the repository.

### 6. Output Evaluation
The deployed application is tested using customer inputs. The application displays the predicted purchase probability and an interpretation of the prediction.

## Run the Project Locally

### Prerequisites
- Python 3.11
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/02vjiya/visit-with-us-tourism-mlops.git
cd visit-with-us-tourism-mlops
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install -r tourism_project/requirements.txt
```

### Step 4: Run the Streamlit application

```bash
python -m streamlit run tourism_project/deployment/app.py
```

The application will open in your browser.

## Model Evaluation

The project stores model evaluation information in:

`tourism_project/deployment/model_metrics.json`

The trained model is stored in:

`tourism_project/deployment/tourism_model.joblib`

MLflow is used for experiment tracking during model training.

## CI/CD Execution

The GitHub Actions workflow is available at:

`.github/workflows/pipeline.yml`

It runs automatically on pushes to the `main` branch, subject to the configured path filters, and can also be triggered manually using `workflow_dispatch`.

**GitHub Actions:**  
https://github.com/02vjiya/visit-with-us-tourism-mlops/actions

## Deployment

The Streamlit application is deployed using Streamlit Community Cloud.

**Live URL:**  
https://visit-with-us-tourism-mlops-kf7iwqhwl4cznjqyz236um.streamlit.app/