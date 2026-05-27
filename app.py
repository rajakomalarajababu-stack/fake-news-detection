import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)
st.markdown(
"""
<style>

textarea{
font-weight:bold;
}

button{
background-color:#2196F3;
color:white;
}

</style>
""",
unsafe_allow_html=True
)
st.title(
    "📰 Fake News Detection Web App"
)

data = pd.read_csv(
    "news.csv"
)

X = data["text"]

y = data["label"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

news = st.text_area(
    "Enter News Text"
)

if st.button(
    "Predict"
):

    news_vector = vectorizer.transform(
        [news]
    )

    prediction = model.predict(
        news_vector
    )[0]

    if prediction=="FAKE":

        st.error(
            "⚠️ FAKE NEWS DETECTED"
        )

    else:

        st.success(
            "✅ REAL NEWS"
        )