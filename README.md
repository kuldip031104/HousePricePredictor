# 🏠 House Price Prediction App

This is a **Streamlit web application** for predicting house prices based on features such as **total square feet, number of bathrooms, number of bedrooms (BHK), balconies, and area type**.  
It also provides interactive **charts and insights** about the dataset.

---

## 🚀 Features
- Predict house price using **Random Forest Regressor**  
- User-friendly **Streamlit UI**  
- Data visualizations:
  - Correlation heatmap
  - Average price by BHK
  - Maximum price by area type
- Displays **model accuracy (R² score)**

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit**
- **Pandas**
- **Seaborn**
- **Matplotlib**
- **Scikit-learn**

---

## 📂 Dataset
The dataset used is `Housing.csv`, which contains columns such as:
- `total_sqft`
- `bath`
- `bhk`
- `balcony`
- `area_type`
- `price`

👉 Make sure to update the file path inside `load_data()` function in the script:
```python
df = pd.read_csv("C:\\Users\\KULDIP\\OneDrive\\Desktop\\house_predicition\\house_predicition\\Housing.csv")
