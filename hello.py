import pandas as pd

# Load the CSV file
# file_path = '/'  # Replace with your file path
data = pd.read_csv(r"C:\Users\KARRI LAKSHMI YAMINI\OneDrive - IIT Delhi\Desktop\yamini\bigBasketProducts.csv")

# Handling missing values
# Filling missing text data with an empty string and numerical data with zero
data.fillna({'product': '', 'category': '', 'sub_category': '', 'brand': '', 'description': '', 
             'sale_price': 0, 'market_price': 0, 'rating': 0}, inplace=True)

# Standardizing text data
# Convert text data to lower case and remove leading/trailing white spaces
text_columns = ['product', 'category', 'sub_category', 'brand', 'description']
for col in text_columns:
    data[col] = data[col].str.lower().str.strip()

# Removing duplicate entries
data.drop_duplicates(inplace=True)
# Optional: Save the cleaned data back to a new CSV file
# data.to_csv('cleaned_data.csv', index=False)

# Display the first few rows of the cleaned data
print(data.head())
# Combine relevant text columns (e.g., product and description)
data['combined_text'] = data['product'] + " " + data['description']
print(data.head())
from sentence_transformers import SentenceTransformer



# Initialize a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Generate embeddings
embeddings = model.encode(data['combined_text'].tolist(), show_progress_bar=True)

# embeddings is now a NumPy array with the vector representation of your text
from qdrant_client import QdrantClient
from qdrant_client.http import models

clien = QdrantClient("localhost", port=6333)

clien.create_collection(
    collection_name="my_coll",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
batch_size = 1000  # Adjust the batch size to keep the payload within the limit

for i in range(0, len(embeddings), batch_size):
    batch_embeddings = embeddings[i:i + batch_size]
    clien.upsert(
        
        collection_name="my_coll",
        points=models.Batch(
            ids=list(range(i + 1, i + 1 + len(batch_embeddings))),
            vectors=batch_embeddings
        )
    )
print("done")