import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

st.title("Irisデータのナイーブベイズ分類")

# =========================
# データの読み込み
# =========================
iris = datasets.load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

# =========================
# ナイーブベイズモデル
# =========================
model = GaussianNB()
model.fit(X, y)

st.subheader("花の特徴量を入力")

sepal_length = st.slider("がく片の長さ", 4.0, 8.0, 5.1)
sepal_width  = st.slider("がく片の幅", 2.0, 4.5, 3.5)
petal_length = st.slider("花弁の長さ", 1.0, 7.0, 1.4)
petal_width  = st.slider("花弁の幅", 0.1, 2.5, 0.2)

x_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# =========================
# 予測（事後確率）
# =========================
probs = model.predict_proba(x_input)[0]
pred_class = model.predict(x_input)[0]

st.subheader("分類結果")

st.write(f"**分類結果：{labels[pred_class]}**")

# =========================
# 確率の可視化
# =========================
fig, ax = plt.subplots()
ax.bar(labels, probs)
ax.set_ylabel("probability")
ax.set_ylim(0, 1)
st.pyplot(fig)

# =========================
# 確率表示（数値）
# =========================
for label, p in zip(labels, probs):
    st.write(f"{label}：{p:.2f}")
