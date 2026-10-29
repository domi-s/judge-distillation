from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from preprocessing import tokenise_code


METHOD_DIR = Path(__file__).resolve().parent
MODEL_PATH = METHOD_DIR / "model.pkl"
VECTORIZER_PATH = METHOD_DIR / "vectorizer.pkl"


st.subheader("Code submission score predictor")
st.write(
    "Paste a C++ submission to predict its expected judge score (0 - 100)."
)

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
  st.error(
      "Model files not found! Please run "
      "`python methods\\tfidf-linear-svm\\train.py` first."
  )
else:
  model = joblib.load(MODEL_PATH)
  vectorizer = joblib.load(VECTORIZER_PATH)
  default_code = """#include <iostream>

using namespace std;

int main()
{
    int a, b;
    cin >> a >> b;
    cout << a + b << endl;
    return 0;
}"""
  user_code = st.text_area("C++ Source Code", value=default_code, height=300)

  if st.button("Predict score"):
    if not user_code.strip():
      st.warning("Please enter some code first.")
    else:
      code_vector = vectorizer.transform([user_code])
      predicted_score = float(np.clip(model.predict(code_vector)[0], 0, 100))
      st.success(f"Predicted Score: **{predicted_score:.2f} / 100**")
