# Machine Learning Regression Analysis on Red Wine Quality Dataset

This project was completed as part of the *Principles of Machine Learning* module on the MSc Computer Science with Artificial Intelligence programme.  
The aim is to build and evaluate regression models that can predict wine quality based on its physicochemical properties.

---

## **Project Overview**

The goal of this coursework is to:
- Explore and analyse the **Red Wine Quality dataset**.
- Preprocess and clean the data (handle outliers, scale features, train/test split).
- Develop multiple **machine learning regression models**.
- Optimise models using **GridSearchCV** and evaluate their performance.
- Compare models based on their accuracy and error metrics.

---

## **Dataset**

- **Dataset name:** Red Wine Quality (UCI Machine Learning Repository)
- **Target variable:** `quality` (wine quality score from 0–10)
- **Input features include:**
  - `fixed acidity`
  - `volatile acidity`
  - `citric acid`
  - `residual sugar`
  - `chlorides`
  - `free sulfur dioxide`
  - `total sulfur dioxide`
  - `density`
  - `pH`
  - `sulphates`
  - `alcohol`

---

## **Data Preprocessing Steps**

✔ Import dataset using `pandas`  
✔ Visualise distributions and detect outliers using boxplots  
✔ Remove outliers using IQR filtering  
✔ Standardise data using `StandardScaler()`  
✔ Split into **training and test sets** using `train_test_split()`  

---

## **Models Implemented**

| Model | Optimised? | Notes |
|--------|------------|-------|
| Linear Regression | ✅ Yes | Baseline model |
| Ridge Regression | ✅ Yes | Tuned with GridSearchCV |
| Decision Tree Regressor | ✅ Yes | Depth and splits optimised |
| Support Vector Regressor (SVR) | ✅ Yes | Scaled data + RBF kernel |
| Random Forest Regressor | ✅ Yes | Tuned with GridSearchCV |
| (Optional) Gradient Boosting / Ensemble models | ✅ Possible extension |

---

## **Model Evaluation**

Each model is evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Cross-validation MSE (5-fold)**
- **R² score**

GridSearchCV is used to find the best hyperparameters for:
✔ Ridge Regression  
✔ Decision Trees  
✔ Random Forest  
✔ SVR  

Results are printed and compared to determine the best-performing model.

---

## **Tools & Libraries Used**

- **Python**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn (sklearn)**  
  - `train_test_split`, `StandardScaler`,  
  - `LinearRegression`, `Ridge`, `DecisionTreeRegressor`,  
  - `RandomForestRegressor`, `SVR`, `GridSearchCV`, `cross_val_score`

---

## **Next Steps / Future Improvements**

- Add ensemble models like Gradient Boosting or XGBoost  
- Deploy model using Flask or Streamlit  
- Add interactive dashboards (Tableau / Power BI)  
- Convert into `.py` scripts for production-ready pipelines  

---

## **Author**

**Darren Ironside**  
MSc Computer Science with Artificial Intelligence  
University of Hertfordshire

---
