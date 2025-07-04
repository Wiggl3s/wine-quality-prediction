import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('wine_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page settings
st.set_page_config(page_title="Wine Quality Checker", page_icon="🍷", layout="centered")

# App title
st.title("🍷 Wine Quality Checker")
st.markdown("Enter the wine's chemical properties below to check if it meets premium quality standards.")

# --- Input fields in one column ---
fixed_acidity = st.number_input('Fixed Acidity', value=7.4)
volatile_acidity = st.number_input('Volatile Acidity', value=0.7)
citric_acid = st.number_input('Citric Acid', value=0.0)
residual_sugar = st.number_input('Residual Sugar', value=1.9)
chlorides = st.number_input('Chlorides', value=0.076)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', value=11.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', value=34.0)
density = st.number_input('Density', value=0.9978, format="%.5f")
pH = st.number_input('pH', value=3.51)
sulphates = st.number_input('Sulphates', value=0.56)
alcohol = st.number_input('Alcohol (%)', value=9.4)

# --- Show ideal values on button click ---
if st.button("📋 Show Ideal Values for Good Quality Wine"):
    st.markdown("### 🍇 Ideal Chemical Properties for Good Quality Red Wine (Quality ≥ 7)")
    st.markdown("""
    | Feature                 | Ideal Range / Value     | Why It Matters                          |
    |-------------------------|--------------------------|------------------------------------------|
    | Fixed Acidity           | 6.5 – 8.5                | Adds freshness and structure             |
    | Volatile Acidity        | 0.25 – 0.55              | Avoids sour/vinegar taste                |
    | Citric Acid             | 0.30 – 0.50              | Enhances fruity flavor                   |
    | Residual Sugar          | 1.5 – 3.0 g/L            | Balances acidity and sweetness           |
    | Chlorides               | 0.045 – 0.070            | Affects saltiness                        |
    | Free Sulfur Dioxide     | 10 – 30 mg/L             | Prevents spoilage and oxidation          |
    | Total Sulfur Dioxide    | 30 – 70 mg/L             | Maintains freshness over time            |
    | Density                 | 0.994 – 0.997 g/cm³      | Reflects sugar/alcohol balance           |
    | pH                      | 3.2 – 3.5                | Influences taste and stability           |
    | Sulphates               | 0.60 – 0.80              | Affects preservation and flavor          |
    | Alcohol (%)             | 10.5 – 13.0              | Adds aroma and body                      |
    """)
    st.info("These ranges are based on data from red wines rated 7 or higher. Matching these values improves your chance of a good quality prediction.")

# --- Prepare input data for model ---
input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

# --- Predict quality on button click ---
if st.button("🚀 Predict Wine Quality"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    if prediction == 1:
        st.success("✅ This wine is GOOD quality!")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")
        st.info("📝 Recommendation: You can label and market this wine as part of your premium collection.")

        # Allow user to name the wine
        wine_name = st.text_input("📝 Name this wine (optional):", placeholder="e.g. Sanborn’s Reserve 2025")

        if "good_wines" not in st.session_state:
            st.session_state.good_wines = []

        if wine_name:
            if wine_name not in st.session_state.good_wines:
                st.session_state.good_wines.append(wine_name)
                st.success(f"✅ Wine **'{wine_name}'** saved to your good wine list!")

        if st.session_state.good_wines:
            st.markdown("### 📜 Saved Good Quality Wines:")
            for i, wine in enumerate(st.session_state.good_wines, 1):
                st.markdown(f"{i}. 🍷 **{wine}**")

    else:
        st.error("❌ This wine is NOT good quality.")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        # Ideal value thresholds for good quality wine
        ideal_ranges = {
            "Fixed Acidity": (6.5, 8.5),
            "Volatile Acidity": (0.25, 0.55),
            "Citric Acid": (0.30, 0.50),
            "Residual Sugar": (1.5, 3.0),
            "Chlorides": (0.045, 0.070),
            "Free Sulfur Dioxide": (10, 30),
            "Total Sulfur Dioxide": (30, 70),
            "Density": (0.994, 0.997),
            "pH": (3.2, 3.5),
            "Sulphates": (0.60, 0.80),
            "Alcohol": (10.5, 13.0)
        }

        user_inputs = {
            "Fixed Acidity": fixed_acidity,
            "Volatile Acidity": volatile_acidity,
            "Citric Acid": citric_acid,
            "Residual Sugar": residual_sugar,
            "Chlorides": chlorides,
            "Free Sulfur Dioxide": free_sulfur_dioxide,
            "Total Sulfur Dioxide": total_sulfur_dioxide,
            "Density": density,
            "pH": pH,
            "Sulphates": sulphates,
            "Alcohol": alcohol
        }

        problems = []
        for key, (low, high) in ideal_ranges.items():
            if not (low <= user_inputs[key] <= high):
                problems.append(f"- **{key}** is `{user_inputs[key]}` but should be between `{low}` and `{high}`.")

        if problems:
            st.warning("⚠️ **Possible reasons why this wine is not good quality:**")
            for item in problems:
                st.markdown(item)
        else:
            st.info("This wine is close to ideal ranges. Minor tuning may still improve it.")

# Optional: Clear the wine list
if "good_wines" in st.session_state and st.session_state.good_wines:
    if st.button("🗑️ Clear Good Wines List"):
        st.session_state.good_wines = []
        st.info("✅ Good wine list has been cleared.")
