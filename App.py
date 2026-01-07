import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ==================================
# CONFIG HALAMAN
# ==================================
st.set_page_config(
    page_title="Analisis Asosiasi Apriori - Bread",
    layout="wide"
)

# ==================================
# JUDUL APLIKASI
# ==================================
st.title("📊 Analisis Asosiasi Produk (Apriori)")
st.markdown("""
Aplikasi ini menampilkan **Market Basket Analysis** menggunakan  
algoritma **Apriori**, dengan fokus analisis pada **item Bread**.
""")

# ==================================
# LOAD DATA
# ==================================
df = pd.read_csv("bread basket.csv")

st.subheader("📄 Preview Dataset")
st.dataframe(df.head())

# ==================================
# PREPROCESSING DATA
# ==================================
transactions = df.groupby('Transaction')['Item'].apply(list)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# ==================================
# SIDEBAR PARAMETER
# ==================================
st.sidebar.header("⚙️ Parameter Apriori")

min_support = st.sidebar.slider(
    "Minimum Support",
    0.01, 0.1, 0.02, 0.01
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.1, 1.0, 0.6, 0.05
)

# ==================================
# APRIORI
# ==================================
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

# ==================================
# FILTER RULES YANG MENGANDUNG BREAD
# ==================================
rules_bread = rules[
    rules['antecedents'].apply(lambda x: 'Bread' in x) |
    rules['consequents'].apply(lambda x: 'Bread' in x)
]

# ==================================
# FREQUENT ITEMSETS
# ==================================
st.subheader("🛒 Frequent Itemsets Teratas")

st.dataframe(
    frequent_itemsets
    .sort_values("support", ascending=False)
    .head(10)
)

# ==================================
# ASSOCIATION RULES (BREAD)
# ==================================
st.subheader("🔗 Association Rules (Item Bread)")

rules_bread['antecedents'] = rules_bread['antecedents'].apply(
    lambda x: ', '.join(list(x))
)
rules_bread['consequents'] = rules_bread['consequents'].apply(
    lambda x: ', '.join(list(x))
)

st.dataframe(
    rules_bread[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    .sort_values("lift", ascending=False)
)

# ==================================
# FILTER ATURAN TERBAIK
# ==================================
st.subheader("⭐ Aturan Terbaik (Confidence ≥ 0.6 & Lift > 1)")

rules_best = rules_bread[
    (rules_bread['confidence'] >= 0.6) &
    (rules_bread['lift'] > 1)
]

st.dataframe(
    rules_best[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
)

# ==================================
# VISUALISASI
# ==================================
st.subheader("📈 Visualisasi Confidence vs Lift (Bread)")

if not rules_best.empty:
    fig = px.scatter(
        rules_best,
        x="confidence",
        y="lift",
        size="support",
        hover_data=["antecedents", "consequents"],
        title="Confidence vs Lift - Item Bread"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Tidak ada aturan yang memenuhi kriteria.")

# ==================================
# KESIMPULAN
# ==================================
st.subheader("📝 Kesimpulan Analisis")

st.markdown("""
- **Lift > 1** menunjukkan hubungan kuat antar produk  
- Aturan dengan **Bread** dapat digunakan untuk:
  - Paket promo (bundling)
  - Rekomendasi produk
  - Strategi penempatan barang
""")
