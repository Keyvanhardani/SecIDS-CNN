import pandas as pd
from secids_cnn import SecIDSModel

# Step 1: Initialize the model
model = SecIDSModel()

# Step 2: Load network traffic data (replace 'path/to/your/data.csv' with the actual path)
data = pd.read_csv('path/to/your/data.csv')

# Step 3: Make predictions
predictions = model.predict(data)

# Output results
print("Intrusion Detection Results:", predictions)
