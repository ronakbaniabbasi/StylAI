import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io

# =====================
# 1. PRODUCT CATALOG
# =====================
def generate_product_catalog(num_products=200):
    categories = ['Dress', 'Shirt', 'Pants', 'Skirt', 'Jacket', 'Shoes', 'Accessories']
    styles = ['Casual', 'Formal', 'Bohemian', 'Streetwear', 'Vintage', 'Sporty', 'Business']
    colors = ['Black', 'White', 'Red', 'Blue', 'Green', 'Yellow', 'Patterned']
    materials = ['Cotton', 'Silk', 'Denim', 'Leather', 'Wool', 'Polyester']

    products = []
    for i in range(num_products):
        category = random.choice(categories)
        price = round(random.uniform(15, 300), 2)
        discount = random.choice([0, 0, 0, 0.1, 0.2, 0.3])
        products.append({
            'product_id': f'prod_{i:04d}',
            'name': f"{random.choice(['Modern', 'Classic', 'Trendy', 'Elegant', 'Chic'])} {category}",
            'category': category,
            'style': random.choice(styles),
            'color': random.choice(colors),
            'material': random.choice(materials),
            'price': price,
            'discounted_price': round(price * (1 - discount), 2),
            'brand': f"Brand_{random.randint(1, 20)}",
            'rating': round(random.uniform(3, 5), 1),
            'description': f"A {random.choice(['beautiful', 'stylish', 'versatile', 'unique'])} {category.lower()} "
                           f"in {random.choice(colors)} {random.choice(materials)}"
        })
    return pd.DataFrame(products)

product_df = generate_product_catalog()

# =====================
# 2. RECOMMENDER ENGINE
# =====================
class FashionRecommender:
    def __init__(self, product_df):
        self.product_df = product_df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._create_style_clusters()

    def _create_style_clusters(self):
        tfidf_matrix = self.vectorizer.fit_transform(self.product_df['description'])
        kmeans = KMeans(n_clusters=7, random_state=42)
        self.product_df['style_cluster'] = kmeans.fit_predict(tfidf_matrix)

    def recommend_outfits(self, budget, style_pref, num_outfits=3):
        # Filter by budget & style
        filtered = self.product_df[
            (self.product_df['discounted_price'] <= budget) &
            (self.product_df['style'] == style_pref)
        ]

        if filtered.empty:
            filtered = self.product_df[self.product_df['discounted_price'] <= budget]

        outfits = []
        for _ in range(num_outfits):
            outfit = filtered.sample(min(3, len(filtered))).to_dict('records')
            outfits.append(outfit)
        return outfits

recommender = FashionRecommender(product_df)

# =====================
# 3. STREAMLIT APP
# =====================
st.title("👗 StyleAI – AI Fashion Stylist & Marketplace")
st.markdown("Smart outfit recommendations based on your budget and preferences.")

# User input
budget = st.slider("Set Your Budget ($)", 50, 2000, 500, 50)
style_pref = st.selectbox("Preferred Style", product_df['style'].unique())

# Show sample catalog
st.subheader("🛍 Browse Product Catalog")
st.dataframe(product_df[['name', 'category', 'style', 'color', 'discounted_price', 'brand']].head(10))

# Generate recommendations
if st.button("✨ Recommend Outfits"):
    outfits = recommender.recommend_outfits(budget, style_pref)
    
    for i, outfit in enumerate(outfits, 1):
        st.markdown(f"### Outfit {i}")
        total_price = sum(item['discounted_price'] for item in outfit)
        cols = st.columns(len(outfit))

        for col, item in zip(cols, outfit):
            with col:
                st.markdown(f"**{item['name']}**")
                st.markdown(f"{item['category']} | {item['color']}")
                st.markdown(f"💲 {item['discounted_price']}")
                st.markdown(f"⭐ {item['rating']}")
        
        st.markdown(f"**Total Outfit Price: ${total_price:.2f}**")
        st.divider()
