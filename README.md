# Predictive Maintenance of Aircraft Engines using Machine Learning

## Overview

This project predicts the Remaining Useful Life (RUL) of turbofan
aircraft engines using NASA’s C-MAPSS dataset. The system simulates
an industrial predictive maintenance system with real-time monitoring,
failure risk estimation, and maintenance recommendations.

The goal is to demonstrate how machine learning can be applied to
prevent unexpected equipment failure and support Industry 4.0 systems.

## Problem Statement

Unexpected failure of critical machinery can lead to severe safety
risks and economic losses. Predictive maintenance aims to estimate
how long a component will continue to operate before failure.

This project focuses on predicting the Remaining Useful Life (RUL)
of aircraft engines based on sensor measurements collected over time.

## Dataset

The project uses the NASA C-MAPSS dataset, which contains simulated
run-to-failure data for turbofan engines.

Each engine is monitored through multiple sensors across operating
cycles until failure occurs.

Key characteristics:

- Multivariate time-series data  
- Multiple engines with different degradation patterns  
- Sensor measurements and operating conditions  
- Separate training and test sets  

Dataset source:  
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## Objectives

- Predict Remaining Useful Life (RUL) of aircraft engines  
- Identify degradation patterns from sensor data  
- Estimate failure risk for maintenance planning  
- Demonstrate an end-to-end predictive maintenance system  

## Methodology

1. Data preprocessing and cleaning  
2. Feature engineering and normalization  
3. Remaining Useful Life (RUL) computation  
4. Model training using Random Forest regression  
5. Evaluation on unseen test engines  
6. Deployment through an interactive dashboard  

## Machine Learning Model

A Random Forest Regressor was used due to its ability to capture
nonlinear relationships and robustness to noise.

Features include selected sensor readings, operating settings,
and cycle information.

Performance was evaluated using RMSE and MAE on validation data.

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Plotly  
- Matplotlib & Seaborn  

## Dashboard Features

- Real-time engine degradation simulation  
- Failure probability estimation  
- Risk classification (Healthy / Warning / Critical)  
- Feature importance visualization  
- Maintenance recommendations  
- Sensor data upload support  

## Results

The model successfully predicts Remaining Useful Life for unseen
engines and identifies high-risk conditions.

Evaluation metrics such as RMSE and MAE indicate reliable performance,
demonstrating the feasibility of machine learning for predictive
maintenance tasks.

The dashboard shows how these predictions can support maintenance
decision-making in real-world scenarios.

## Dashboard Preview

![Dashboard](images/dashboard.png)


## Requirements

Python 3.10+  
See requirements.txt for the full list of dependencies.

## How to Run

1. Install dependencies:

   pip install -r requirements.txt

2. Launch dashboard:

   streamlit run app/app.py

## RUL Definition

Remaining Useful Life (RUL) was computed as:

RUL = max_cycle − current_cycle

## Challenges Faced

- Handling multivariate time-series data  
- Ensuring consistency between training and deployment pipelines  
- Designing realistic failure probability estimation  
- Implementing real-time simulation in Streamlit  

## Future Work

- Incorporating deep learning models  
- Using survival analysis for probabilistic failure prediction  
- Extending to multi-engine fleet monitoring  
- Deploying the system in cloud environments  

## Author Contribution

This project was independently designed and implemented by me.
Key contributions include data preprocessing, model development,
evaluation, dashboard design, and deployment preparation.

## License

This project is intended for academic and educational purposes.

## References

NASA C-MAPSS Dataset  
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/