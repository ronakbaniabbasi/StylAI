# %% [markdown]
"""
# AI-Powered Fashion E-Commerce & Stylist Platform
## Prototype Implementation

This notebook contains a prototype implementation of:
1. Product catalog management
2. User preference learning
3. AI-powered outfit recommendations
4. Budget-aware filtering
5. Basic UI simulation
"""
# %%
# Import required libraries
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from PIL import Image
import io
import requests
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
"""
## 1. Product Catalog Simulation
First, let's create a simulated product catalog with fashion items from different brands
"""
# %%
# Create sample product catalog
def generate_product_catalog(num_products=200):
    categories = ['Dress', 'Shirt', 'Pants', 'Skirt', 'Jacket', 'Shoes', 'Accessories']
    styles = ['Casual', 'Formal', 'Bohemian', 'Streetwear', 'Vintage', 'Sporty', 'Business']
    colors = ['Black', 'White', 'Red', 'Blue', 'Green', 'Yellow', 'Patterned']
    materials = ['Cotton', 'Silk', 'Denim', 'Leather', 'Wool', 'Polyester']
    
    products = []
    for i in range(num_products):
        category = random.choice(categories)
        price = round(random.uniform(15, 300), 2)
        discount = random.choice([0, 0, 0, 0.1, 0.2, 0.3])  # Mostly no discount
        
        product = {
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
            'inventory': random.randint(5, 100),
            'description': f"A {random.choice(['beautiful', 'stylish', 'versatile', 'unique'])} {category.lower()} "
                          f"in {random.choice(colors)} {random.choice(materials)} "
                          f"for {random.choice(['everyday wear', 'special occasions', 'work', 'parties'])}"
        }
        products.append(product)
    
    return pd.DataFrame(products)

product_df = generate_product_catalog()
product_df.head()

# %%
# Add some popular items that go well together (for better recommendations)
popular_combinations = [
    {'category': 'Shirt', 'style': 'Business', 'color': 'White'},
    {'category': 'Pants', 'style': 'Business', 'color': 'Black'},
    {'category': 'Dress', 'style': 'Formal', 'color': 'Black'},
    {'category': 'Jacket', 'style': 'Streetwear', 'color': 'Denim'},
    {'category': 'Shoes', 'style': 'Casual', 'color': 'White'},
]

# Fix 1: Replace DataFrame.append() with pd.concat()
# In the product catalog generation section, replace:
# product_df = product_df.append(new_product, ignore_index=True)

# Fix 2: Correct the typo in 'colors' (was 'colors')
# In the recommend_outfits method, change:
# (self.product_df['color'].isin(top_colors))
# From:
# (self.product_df['color'].isin(top_colors))

# Here's the corrected complete code for those sections:

# In the product catalog generation:
for combo in popular_combinations:
    matches = product_df[
        (product_df['category'] == combo['category']) & 
        (product_df['style'] == combo['style']) & 
        (product_df['color'] == combo['color'])
    ]
    if len(matches) == 0:
        new_product = {
            'product_id': f'prod_comb_{len(product_df)}',
            'name': f"Classic {combo['color']} {combo['category']}",
            'category': combo['category'],
            'style': combo['style'],
            'color': combo['color'],
            'material': 'Cotton' if combo['category'] != 'Shoes' else 'Leather',
            'price': round(random.uniform(50, 200), 2),
            'discounted_price': round(random.uniform(40, 180), 2),
            'brand': "Brand_Classic",
            'rating': round(random.uniform(4, 5), 1),
            'inventory': random.randint(20, 100),
            'description': f"A classic {combo['color'].lower()} {combo['category'].lower()} "
                        f"perfect for {combo['style'].lower()} occasions"
        }
        product_df = pd.concat([product_df, pd.DataFrame([new_product])], ignore_index=True)

# In the recommend_outfits method:
def recommend_outfits(self, user_profile, num_outfits=3, items_per_outfit=3):
    """Recommend complete outfits based on user preferences"""
    # Get user preferences
    top_styles = user_profile.get_top_preferences('styles')
    top_colors = user_profile.get_top_preferences('colors')  # Fixed typo
    top_categories = user_profile.get_top_preferences('categories')
    budget = user_profile.budget
    
    # Filter products by preferences
    filtered_products = self.product_df[
        (self.product_df['style'].isin(top_styles)) |
        (self.product_df['color'].isin(top_colors)) |  # Fixed variable name
        (self.product_df['category'].isin(top_categories))
    ]
    
    # Rest of the method remains the same...

# %% [markdown]
"""
## 2. User Preference Modeling
Create a system to learn and store user preferences
"""
# %%
class UserProfile:
    def __init__(self):
        self.preferences = {
            'styles': {},
            'colors': {},
            'categories': {},
            'price_range': [0, float('inf')],
            'brands': {},
            'click_history': [],
            'purchase_history': []
        }
        self.budget = 500  # Default budget
    
    def update_preferences(self, item, interaction_type='click'):
        # Update style preference
        style = item['style']
        self.preferences['styles'][style] = self.preferences['styles'].get(style, 0) + 1
        
        # Update color preference
        color = item['color']
        self.preferences['colors'][color] = self.preferences['colors'].get(color, 0) + 1
        
        # Update category preference
        category = item['category']
        self.preferences['categories'][category] = self.preferences['categories'].get(category, 0) + 1
        
        # Update brand preference
        brand = item['brand']
        self.preferences['brands'][brand] = self.preferences['brands'].get(brand, 0) + 1
        
        # Record interaction
        if interaction_type == 'click':
            self.preferences['click_history'].append(item['product_id'])
        elif interaction_type == 'purchase':
            self.preferences['purchase_history'].append(item['product_id'])
    
    def set_budget(self, budget):
        self.budget = budget
        self.preferences['price_range'] = [0, budget]
    
    def get_top_preferences(self, pref_type, n=3):
        preferences = sorted(self.preferences[pref_type].items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in preferences[:n]]

# %% [markdown]
"""
## 3. Recommendation Engine
Create an AI-powered recommendation system that suggests outfits
"""
# %%
class FashionRecommender:
    def __init__(self, product_df):
        self.product_df = product_df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.style_clusters = self._create_style_clusters()
        
    def _create_style_clusters(self):
        """Cluster products by style using their descriptions"""
        tfidf_matrix = self.vectorizer.fit_transform(self.product_df['description'])
        kmeans = KMeans(n_clusters=7, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        self.product_df['style_cluster'] = clusters
        return clusters
    
    def _get_similar_items(self, reference_item, n=5):
        """Find similar items based on description"""
        tfidf_matrix = self.vectorizer.transform(self.product_df['description'])
        ref_idx = self.product_df[self.product_df['product_id'] == reference_item['product_id']].index[0]
        sim_scores = cosine_similarity(tfidf_matrix[ref_idx], tfidf_matrix)
        similar_indices = sim_scores.argsort()[0][-n-1:-1][::-1]
        return self.product_df.iloc[similar_indices].to_dict('records')
    
    def recommend_outfits(self, user_profile, num_outfits=3, items_per_outfit=3):
        """Recommend complete outfits based on user preferences"""
        # Get user preferences
        top_styles = user_profile.get_top_preferences('styles')
        top_colors = user_profile.get_top_preferences('colors')
        top_categories = user_profile.get_top_preferences('categories')
        budget = user_profile.budget
        
        # Filter products by preferences
        filtered_products = self.product_df[
            (self.product_df['style'].isin(top_styles)) |
            (self.product_df['color'].isin(top_colors)) |
            (self.product_df['category'].isin(top_categories))
        ]
        
        # Further filter by budget (total outfit cost should be <= budget)
        filtered_products = filtered_products[filtered_products['discounted_price'] <= budget]
        
        # If no preferences yet, use popular items
        if len(filtered_products) < 10:
            filtered_products = self.product_df[self.product_df['rating'] >= 4.5]
        
        # Generate outfits
        outfits = []
        for _ in range(num_outfits):
            outfit = []
            remaining_budget = budget
            
            # Start with a key item (top/dress)
            key_categories = ['Dress', 'Shirt', 'Jacket']
            key_items = filtered_products[filtered_products['category'].isin(key_categories)]
            if len(key_items) > 0:
                key_item = key_items.sample(1).iloc[0].to_dict()
                outfit.append(key_item)
                remaining_budget -= key_item['discounted_price']
            
            # Add complementary items
            comp_categories = ['Pants', 'Skirt'] if key_item['category'] in ['Shirt'] else ['Shoes', 'Accessories']
            comp_items = filtered_products[
                filtered_products['category'].isin(comp_categories) &
                (filtered_products['discounted_price'] <= remaining_budget)
            ]
            
            if len(comp_items) > 0:
                comp_item = comp_items.sample(1).iloc[0].to_dict()
                outfit.append(comp_item)
                remaining_budget -= comp_item['discounted_price']
            
            # Add accessories if budget allows
            if remaining_budget > 0:
                acc_items = filtered_products[
                    filtered_products['category'].isin(['Accessories', 'Shoes']) &
                    (filtered_products['discounted_price'] <= remaining_budget)
                ]
                if len(acc_items) > 0:
                    acc_item = acc_items.sample(1).iloc[0].to_dict()
                    outfit.append(acc_item)
            
            if len(outfit) >= 2:  # At least 2 items to be considered an outfit
                outfits.append(outfit)
        
        return outfits[:num_outfits]  # Return only the requested number of outfits

# %% [markdown]
"""
## 4. UI Simulation with Jupyter Widgets
Create an interactive interface to simulate the app experience
"""
# %%
class FashionAppUI:
    def __init__(self, product_df):
        self.product_df = product_df
        self.user = UserProfile()
        self.recommender = FashionRecommender(product_df)
        self.current_products = product_df.sample(6).to_dict('records')  # Initial products to show
        
        # Create widgets
        self.create_widgets()
        self.setup_ui()
    
    def create_widgets(self):
        # Product display
        self.product_images = [widgets.Image(layout=widgets.Layout(width='200px', height='200px')) for _ in range(6)]
        self.product_names = [widgets.Label() for _ in range(6)]
        self.product_prices = [widgets.Label() for _ in range(6)]
        self.product_buttons = [widgets.Button(description='View Details') for _ in range(6)]
        
        # Recommendation display
        self.outfit_images = [widgets.Image(layout=widgets.Layout(width='150px', height='150px')) for _ in range(3)]
        self.outfit_descriptions = [widgets.Label() for _ in range(3)]
        self.outfit_prices = [widgets.Label() for _ in range(3)]
        self.outfit_buttons = [widgets.Button(description='Shop This Look') for _ in range(3)]
        
        # User controls
        self.budget_slider = widgets.IntSlider(
            value=500,
            min=50,
            max=2000,
            step=50,
            description='Budget:',
            continuous_update=False
        )
        self.style_dropdown = widgets.Dropdown(
            options=['Casual', 'Formal', 'Bohemian', 'Streetwear', 'Vintage', 'Sporty', 'Business'],
            value='Casual',
            description='Preferred Style:'
        )
        self.refresh_button = widgets.Button(description='Refresh Recommendations')
        self.preferences_button = widgets.Button(description='Update Preferences')
        
        # Output area
        self.output = widgets.Output()
    
    def setup_ui(self):
        # Set up button click handlers
        for i, button in enumerate(self.product_buttons):
            button.on_click(lambda b, idx=i: self.show_product_details(idx))
        
        for i, button in enumerate(self.outfit_buttons):
            button.on_click(lambda b, idx=i: self.show_outfit_details(idx))
        
        self.budget_slider.observe(self.update_budget, names='value')
        self.refresh_button.on_click(self.refresh_recommendations)
        self.preferences_button.on_click(self.update_user_preferences)
        
        # Initial UI update
        self.update_product_display()
        self.generate_recommendations()
    
    def update_product_display(self):
        for i in range(6):
            if i < len(self.current_products):
                product = self.current_products[i]
                # In a real app, we'd use actual product images
                self.product_images[i].value = self.generate_product_image(product)
                self.product_names[i].value = product['name']
                self.product_prices[i].value = f"${product['discounted_price']:.2f}"
                self.product_buttons[i].disabled = False
            else:
                self.product_images[i].value = b''
                self.product_names[i].value = ''
                self.product_prices[i].value = ''
                self.product_buttons[i].disabled = True
    
    def generate_product_image(self, product):
        """Generate a placeholder image based on product attributes"""
        # In a real app, this would fetch actual product images
        # Here we create a simple placeholder with product info
        plt.figure(figsize=(2, 2))
        plt.text(0.5, 0.5, f"{product['category']}\n{product['color']}\n${product['price']:.2f}", 
                ha='center', va='center')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()
    
    def update_budget(self, change):
        self.user.set_budget(change['new'])
        self.generate_recommendations()
    
    def update_user_preferences(self, button):
        # In a real app, this would be more comprehensive
        self.user.preferences['styles'][self.style_dropdown.value] = 5
        self.generate_recommendations()
    
    def refresh_recommendations(self, button):
        self.current_products = self.product_df.sample(6).to_dict('records')
        self.update_product_display()
        self.generate_recommendations()
    
    def generate_recommendations(self):
        outfits = self.recommender.recommend_outfits(self.user)
        
        with self.output:
            self.output.clear_output()
            
            # Display product grid
            product_grid = widgets.GridBox(
                children=[item for sublist in zip(
                    self.product_images, 
                    self.product_names, 
                    self.product_prices, 
                    self.product_buttons
                ) for item in sublist],
                layout=widgets.Layout(
                    grid_template_columns='repeat(3, 250px)',
                    grid_gap='20px'
                )
            )
            display(product_grid)
            
            # Display recommendations header
            display(widgets.HTML("<h2>Recommended Outfits For You</h2>"))
            
            # Display outfit recommendations
            for i, outfit in enumerate(outfits[:3]):
                if i >= len(self.outfit_images):
                    break
                    
                total_price = sum(item['discounted_price'] for item in outfit)
                items_desc = " + ".join([f"{item['category']} (${item['discounted_price']:.2f})" for item in outfit])
                
                # Generate a combined outfit image
                self.outfit_images[i].value = self.generate_outfit_image(outfit)
                self.outfit_descriptions[i].value = f"Outfit {i+1}: {items_desc}"
                self.outfit_prices[i].value = f"Total: ${total_price:.2f}"
                
                outfit_box = widgets.HBox([
                    self.outfit_images[i],
                    widgets.VBox([
                        self.outfit_descriptions[i],
                        self.outfit_prices[i],
                        self.outfit_buttons[i]
                    ])
                ])
                display(outfit_box)
    
    def generate_outfit_image(self, outfit):
        """Generate a placeholder outfit image"""
        fig, axes = plt.subplots(1, len(outfit), figsize=(len(outfit)*2, 2))
        if len(outfit) == 1:
            axes = [axes]
            
        for ax, item in zip(axes, outfit):
            ax.text(0.5, 0.5, item['category'][:3], ha='center', va='center')
            ax.set_title(f"${item['discounted_price']:.2f}")
            ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()
    
    def show_product_details(self, index):
        product = self.current_products[index]
        self.user.update_preferences(product, 'click')
        
        with self.output:
            self.output.clear_output()
            display(widgets.HTML(
                f"<h2>{product['name']}</h2>"
                f"<p><strong>Brand:</strong> {product['brand']}</p>"
                f"<p><strong>Price:</strong> ${product['discounted_price']:.2f} "
                f"(<s>${product['price']:.2f}</s>)</p>"
                f"<p><strong>Style:</strong> {product['style']}</p>"
                f"<p><strong>Color:</strong> {product['color']}</p>"
                f"<p><strong>Material:</strong> {product['material']}</p>"
                f"<p><strong>Description:</strong> {product['description']}</p>"
            ))
            
            # Show similar items
            similar_items = self.recommender._get_similar_items(product)
            display(widgets.HTML("<h3>Similar Items</h3>"))
            
            similar_images = []
            for item in similar_items[:3]:
                img = widgets.Image(value=self.generate_product_image(item), width=100, height=100)
                similar_images.append(img)
            
            display(widgets.HBox(similar_images))
    
    def show_outfit_details(self, index):
        outfits = self.recommender.recommend_outfits(self.user)
        if index >= len(outfits):
            return
            
        outfit = outfits[index]
        total_price = sum(item['discounted_price'] for item in outfit)
        
        with self.output:
            self.output.clear_output()
            display(widgets.HTML("<h2>Complete Outfit Details</h2>"))
            
            for item in outfit:
                display(widgets.HTML(
                    f"<h3>{item['name']}</h3>"
                    f"<p><strong>Brand:</strong> {item['brand']}</p>"
                    f"<p><strong>Price:</strong> ${item['discounted_price']:.2f}</p>"
                    f"<p><strong>Style:</strong> {item['style']}</p>"
                ))
            
            display(widgets.HTML(f"<h3>Total Outfit Price: ${total_price:.2f}</h3>"))
            
            # Add to cart button (simulated)
            add_to_cart = widgets.Button(description="Add All to Cart")
            add_to_cart.on_click(lambda b: self.add_to_cart(outfit))
            display(add_to_cart)
    
    def add_to_cart(self, items):
        for item in items:
            self.user.update_preferences(item, 'purchase')
        
        with self.output:
            self.output.clear_output()
            display(widgets.HTML(
                "<h2 style='color:green'>Items added to your cart!</h2>"
                "<p>We'll use your purchase to improve future recommendations.</p>"
            ))
    
    def show(self):
        # Main app layout
        controls = widgets.HBox([
            self.budget_slider,
            self.style_dropdown,
            self.preferences_button,
            self.refresh_button
        ])
        
        app = widgets.VBox([
            widgets.HTML("<h1>FashionAI Stylist & Marketplace</h1>"),
            controls,
            self.output
        ])
        
        display(app)
        self.generate_recommendations()

# %%
# Run the app
app = FashionAppUI(product_df)
app.show()

# %% [markdown]
"""
## Key Features Implemented:

1. **Product Catalog Management**:
   - Simulated database of fashion products with attributes
   - Support for multiple brands and categories

2. **User Preference Learning**:
   - Tracks user clicks, purchases, and preferences
   - Learns preferred styles, colors, categories, and brands
   - Budget awareness

3. **AI-Powered Recommendations**:
   - Content-based filtering using product descriptions
   - Outfit combination logic
   - Budget-aware recommendations
   - Similar item suggestions

4. **Interactive UI**:
   - Product browsing
   - Outfit recommendations
   - Preference adjustment
   - Budget control

## Next Steps for Full Implementation:

1. **Real Data Integration**:
   - Connect to actual product databases or e-commerce APIs
   - Use real product images

2. **Enhanced Recommendation Algorithms**:
   - Collaborative filtering with user behavior data
   - Deep learning for visual similarity
   - Seasonal trend analysis

3. **Advanced Features**:
   - Virtual try-on with AR
   - Social sharing of outfits
   - Integration with designers' lookbooks
   - Multi-user accounts and profiles

4. **Deployment**:
   - Convert to a web/mobile app using Flask/Django or React Native
   - Cloud deployment with scalable backend
"""