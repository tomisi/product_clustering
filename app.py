from flask import Flask, render_template
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure FLASK_DEBUG from environment variable
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Load the CSV data
df = pd.read_csv("amazon_product_data.csv")

# Vectorize product names
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Product Name'])

# Perform clustering
kmeans = KMeans(n_clusters=4)  # You can adjust the number of clusters
df['Cluster'] = kmeans.fit_predict(X)

@app.route('/')
def index():
    # Group products by cluster
    grouped_products = df.groupby('Cluster').apply(lambda x: x[['Product Name', 'Price', 'Rating', 'Availability']].to_dict(orient='records')).to_dict()

    # Pass the clustered products to the template
    return render_template('index.html', products=grouped_products)

if __name__ == '__main__':
    app.run()