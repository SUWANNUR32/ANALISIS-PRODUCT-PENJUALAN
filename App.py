import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_pattern import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# =========================
# JUDUL APLIKASI
# =========================
st.set_page_config(page_title="Analisis Asosiasi Apriori", layout="wide")

st.title("📊 Analisis Asosiasi Produk (Apriori)")
st.write("""
Aplikasi ini menampilkan hasil **Market Basket Analysis**  
menggunakan algoritma **Apriori** pada data transaksi Bread Basket.
""")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("bread basket.csv")

st.subheader("📄 Preview Dataset")
st.dataframe(df.head())

# =========================
# PREPROCESSING
# =========================
transactions = df.groupby('Transaction')['Item'].apply(list)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# =========================
# SIDEBAR PARAMETER
# =========================
st.sidebar.header("⚙️ Parameter Apriori")

min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.01,
    max_value=0.1,
    value=0.02,
    step=0.01
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.05
)

# =========================
# APRIORI
# =========================
frequent_itemsets = apriori(
    df_trans,
    min_support=min_support,
    use_colnames=True
)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=min_confidence
)

# =========================
# FREQUENT ITEMSETS
# =========================
st.subheader("🛒 Frequent Itemsets")
st.write("Kombinasi produk yang sering muncul bersama.")

st.dataframe(
    frequent_itemsets
    .sort_values("support", ascending=False)
    .head(10)
)

# =========================
# ASSOCIATION RULES
# =========================
st.subheader("🔗 Association Rules")

rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

st.dataframe(
    rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    .sort_values("lift", ascending=False)
)

# =========================
# FILTER ATURAN TERBAIK
# =========================
st.subheader("⭐ Aturan Terbaik")

rules_best = rules[
    (rules['confidence'] >= 0.6) &
    (rules['lift'] > 1)
]

st.dataframe(
    rules_best[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
)

# =========================
# VISUALISASI
# =========================
st.subheader("📈 Visualisasi Confidence vs Lift")

fig = px.scatter(
    rules_best,
    x="confidence",
    y="lift",
    size="support",
    hover_data=["antecedents", "consequents"],
    title="Hubungan Confidence dan Lift"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# KESIMPULAN
# =========================
st.subheader("📝 Kesimpulan")

st.write("""
- **Lift > 1** menunjukkan hubungan antar produk yang kuat  
- Aturan ini dapat digunakan untuk:
  - Rekomendasi produk
  - Penempatan barang
  - Paket promo
""")
