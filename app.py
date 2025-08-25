import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Page 
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\KULDIP\\OneDrive\\Desktop\\house_predicition\\house_predicition\\Housing.csv")
    df = df[['total_sqft', 'bath', 'bhk', 'balcony', 'area_type', 'price']]
    return df

df = load_data()

# Split data and train model
X = df[['total_sqft', 'bath', 'bhk', 'balcony']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Navigation Menu
menu = st.sidebar.radio("Navigate", ["Predict Price", "Charts"])


if menu == "Predict Price":
    st.subheader("Enter House Details")

    st.markdown(f"**Model Accuracy:** `{r2*100:.2f}%`")

    total_sqft = st.number_input("Total Sqft Area", min_value=100.0, max_value=10000.0, step=10.0)
    bath = st.slider("Number of Bathrooms", 1, 5, 2)
    bhk = st.slider("Number of BHK", 1, 6, 2)
    balcony = st.slider("Number of Balconies", 0, 5, 2)

    if st.button("Predict Price "):
        input_df = pd.DataFrame([[total_sqft, bath, bhk, balcony]],
                                columns=['total_sqft', 'bath', 'bhk', 'balcony'])
        predicted_price = model.predict(input_df)[0]
        st.success(f"Estimated Price: ₹ {predicted_price:,.2f} lakhs")

elif menu == "Charts":
    st.subheader(" Data Insights and Visualizations")

    # 1. Heatmap
    st.markdown("###  Correlation Heatmap")
    fig1, ax1 = plt.subplots()
    sns.heatmap(df[['total_sqft', 'bath', 'bhk', 'balcony', 'price']].corr(), annot=True, cmap='coolwarm', ax=ax1)
    st.pyplot(fig1)

    # Create df2 for grouped charts
    df2 = df.copy()

    # 2. Avg price per BHK
    st.markdown("### Average Price by BHK")
    fig2, ax2 = plt.subplots()
    df2.groupby("bhk")["price"].mean().plot(kind="line", marker="o", ax=ax2)
    ax2.set_ylabel("Average Price (in lakhs)")
    ax2.set_xlabel("BHK")
    ax2.grid(True)
    st.pyplot(fig2)

    # 3 Area type vs Max Price
    st.markdown("###  Area-wise Maximum Price")
    fig3, ax3 = plt.subplots()
    df2.groupby("area_type")["price"].max().plot(kind="bar", ax=ax3, color='skyblue')
    ax3.set_ylabel("Max Price (in lakhs)")
    ax3.set_xlabel("Area Type")
    ax3.set_title("Maximum Price by Area Type")
    ax3.grid(True)
    st.pyplot(fig3)
