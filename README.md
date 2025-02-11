# Real Estate Price Prediction - Linear Regression Project

## 📌 Introduction
This project aims to predict real estate prices based on property features such as square footage, number of bedrooms and bathrooms, and location. The dataset used is from Bengaluru, India, and has been cleaned and processed for better model accuracy.

## 📊 Problem Statement
- Remove data errors like 6 BHK apartments with 1020 sqft and 8 BHK apartments with 600 sqft.
- Remove **2 BHK** apartments with price per sqft **less than the mean price per sqft of 1 BHK apartments**.
- Ensure that **number of bathrooms does not exceed bedrooms + 1**.
- Apply **One Hot Encoding** for categorical variables (locations).

## 📂 About the Data
The dataset consists of various features related to real estate properties:
- **area_type**: The type of property area (e.g., Super Built-up Area, Carpet Area, etc.).
- **availability**: Availability status of the property (e.g., Ready to Move, Immediate Possession, etc.).
- **location**: The locality where the property is situated.
- **size**: Number of bedrooms in the property (e.g., 2 BHK, 3 BHK, etc.).
- **society**: The name of the society or housing project.
- **total_sqft**: The total area of the property in square feet.
- **bath**: Number of bathrooms available in the property.
- **balcony**: Number of balconies.
- **price**: The price of the property in lakh Indian Rupees.

## 🛠 Workflow

### 1️⃣ Data Cleaning
- Handled missing values by removing NA entries.
- Removed outliers in `total_sqft` that did not fit expected patterns.
- Converted string-based sqft values to numerical format.

### 2️⃣ Feature Engineering
- Extracted `bhk` (number of bedrooms) from `size` column.
- Created `price_per_sqft` feature.
- Applied **One Hot Encoding** for the `location` feature.

### 3️⃣ Outlier Detection & Removal
- Removed properties where `total_sqft/bhk` was less than 300 sqft.
- Dropped 2 BHK apartments priced lower per sqft than 1 BHK apartments in the same location.
- Ensured total bathrooms did not exceed **bedrooms + 1**.

### 4️⃣ Model Selection
- Used **Linear Regression** as the primary model.
- Applied **GridSearchCV** to compare regression models:
  - **Linear Regression**
  - **Lasso Regression**
  - **Decision Tree Regressor**
- Chose **Linear Regression** as the best-performing model.

## 📚 Libraries Used
| Category | Libraries Used |
|----------|---------------|
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Feature Engineering** | `ShuffleSplit`, `GridSearchCV` |

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Run the Python Script or Jupyter Notebook
```sh
python real_estate_price_prediction.py
```
Or open `real_estate_price_prediction.ipynb` and run all cells.

## 📌 Results & Insights
- Linear Regression provided the best performance for price prediction.
- Removing anomalies and irrelevant features improved model accuracy.
- **Price per sqft** was an important feature in detecting pricing outliers.
- **Location Encoding** played a key role in improving predictions.

## 🏆 Conclusion
This model provides a solid foundation for real estate price prediction and can be further improved with advanced machine learning models.

## 📌 Future Improvements
- Implement **Random Forest and XGBoost** for better accuracy.
- Add **Time-Series Analysis** to track market trends.
- Apply **Deep Learning models** for better feature learning.

## 📩 Contact
If you have any questions or suggestions, feel free to reach out!

✉️ Email: `mukulmanav0@gmail.com`  
📌 GitHub: [mukulmanav](https://github.com/mukulmanav)

🚀 **Happy Coding!** 🎉
