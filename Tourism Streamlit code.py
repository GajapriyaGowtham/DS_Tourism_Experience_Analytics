import streamlit as st
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import plotly.express as px
import numpy as np


st.set_page_config(page_title=" Tourism Predictor", layout="wide")
st.title("🌍 Tourism Experience Predictor & Recommender")


# Caching Functions

@st.cache_data
def load_data():
    engine = create_engine("mariadb+pymysql://root:@localhost/tourism_db")
    df = pd.read_sql("SELECT * FROM preprocessed_data", con=engine)

    for col in ["UserContinent", "UserRegion", "UserCountry", "UserCity", "AttractionType"]:
        df[col] = df[col].astype(str).str.strip()

    return df

@st.cache_resource
def get_encoders(data):
    encoders = {
        "UserContinent": LabelEncoder().fit(data["UserContinent"]),
        "UserRegion": LabelEncoder().fit(data["UserRegion"]),
        "UserCountry": LabelEncoder().fit(data["UserCountry"]),
        "UserCity": LabelEncoder().fit(data["UserCity"]),
        "AttractionType": LabelEncoder().fit(data["AttractionType"]),
    }
    return encoders

@st.cache_resource
def train_models(X, y_reg, y_cls):
    reg_model = RandomForestRegressor().fit(X, y_reg)
    clf_model = RandomForestClassifier().fit(X, y_cls)
    return reg_model, clf_model


# Load Data 

df = load_data()
encoders = get_encoders(df)

# Add encoded columns to df
df["UserContinent_Encoded"] = encoders["UserContinent"].transform(df["UserContinent"])
df["UserRegion_Encoded"] = encoders["UserRegion"].transform(df["UserRegion"])
df["UserCountry_Encoded"] = encoders["UserCountry"].transform(df["UserCountry"])
df["UserCity_Encoded"] = encoders["UserCity"].transform(df["UserCity"])
df["AttractionType_Encoded"] = encoders["AttractionType"].transform(df["AttractionType"])

# Feature and targets
X = df[[ 'VisitYear', 'VisitMonth',
         'UserContinent_Encoded', 'UserRegion_Encoded', 'UserCountry_Encoded',
         'UserCity_Encoded', 'AttractionType_Encoded',
         'Total_Visits_User', 'Avg_Rating_User_Scaled' ]]

y_reg = df['Rating']
y_cls = df['VisitMode']

reg_model, clf_model = train_models(X, y_reg, y_cls)


#  Sidebar Input

st.sidebar.title(" User Input")

visit_year = st.sidebar.selectbox("Visit Year", sorted(df["VisitYear"].unique()))
visit_month = st.sidebar.selectbox("Visit Month", sorted(df["VisitMonth"].unique()))

continent = st.sidebar.selectbox("Continent", sorted(df["UserContinent"].unique()))
regions = df[df["UserContinent"] == continent]["UserRegion"].unique()
region = st.sidebar.selectbox("Region", sorted(regions))

countries = df[(df["UserContinent"] == continent) & (df["UserRegion"] == region)]["UserCountry"].unique()
country = st.sidebar.selectbox("Country", sorted(countries))

cities = df[df["UserCountry"] == country]["UserCity"].unique()
city = st.sidebar.selectbox("City", sorted(cities))

attraction_type = st.sidebar.selectbox("Attraction Type", sorted(df["AttractionType"].unique()))
total_visits = st.sidebar.slider("Total Visits by User", 1, 20, 3)
avg_rating = st.sidebar.slider("Avg Rating (scaled)", 0.0, 1.0, 0.75)


#  Predict Function


def predict_user(
    visit_year, visit_month, continent, region, country, city,
    attraction_type, total_visits, avg_rating, encoders
):
    try:
        input_dict = {
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "UserContinent_Encoded": encoders["UserContinent"].transform([continent])[0],
            "UserRegion_Encoded": encoders["UserRegion"].transform([region])[0],
            "UserCountry_Encoded": encoders["UserCountry"].transform([country])[0],
            "UserCity_Encoded": encoders["UserCity"].transform([city])[0],
            "AttractionType_Encoded": encoders["AttractionType"].transform([attraction_type])[0],
            "Total_Visits_User": total_visits,
            "Avg_Rating_User_Scaled": avg_rating
        }

        input_df = pd.DataFrame([input_dict])
        predicted_rating = reg_model.predict(input_df)[0]
        predicted_mode = clf_model.predict(input_df)[0]

        return round(predicted_rating, 2), predicted_mode

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

#  Tabs 

tab1, tab2, tab3 = st.tabs([" Predict & Recommend", " Data Insights", " Raw Data"])

with tab1:
    st.subheader(" Prediction and Recommendations")

    if st.button(" Predict Now"):
        pred_rating, pred_mode = predict_user(
            visit_year, visit_month, continent, region, country,
            city, attraction_type, total_visits, avg_rating, encoders
        )

        if pred_rating is not None and pred_mode is not None:
            st.success(f" Predicted Rating: {pred_rating} / 5")
            st.success(f"Predicted Visit Mode: {pred_mode}")

            st.subheader("Recommended Attractions")
            filtered_df = df[
                (df["UserContinent"] == continent) &
                (df["UserRegion"] == region) &
                (df["AttractionType"] == attraction_type)
            ]

            st.write(f"Found  matching records.")

            if "Attraction" not in df.columns:
                st.error("'Attraction' column not found in dataset.")
            elif filtered_df.empty:
                st.warning(" No matching attractions found for the selected filters.")
            else:
                top_recs = (
                    filtered_df.groupby("Attraction")["Rating"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(5)
                    .reset_index()
                )
                st.dataframe(top_recs)

with tab2:
    st.subheader("Insights from Dataset")

    col1, col2 = st.columns(2)

    with col1:
        mode_count = df["VisitMode"].value_counts().reset_index()
        mode_count.columns = ["VisitMode", "Count"]
        fig1 = px.bar(mode_count, x="VisitMode", y="Count", title="Visit Mode Distribution", color="VisitMode")
        st.plotly_chart(fig1)

    with col2:
        avg_rating_by_region = df.groupby("UserRegion")["Rating"].mean().reset_index().sort_values("Rating", ascending=False)
        fig2 = px.bar(avg_rating_by_region, x="UserRegion", y="Rating", title="Avg Rating by Region", color="UserRegion")
        st.plotly_chart(fig2)

with tab3:
    st.subheader("Full Dataset Sample")
    st.dataframe(df.sample(50))