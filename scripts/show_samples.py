"""Show sample reviews from the dataset."""

import pandas as pd

# Load train data
df = pd.read_csv('data/raw/train/train_seed42.csv')

print("=" * 80)
print("Sample Amazon Reviews")
print("=" * 80)
print()

# Show 10 samples
for i in range(10):
    label = "Positive" if df.iloc[i]['label'] == 1 else "Negative"
    text = df.iloc[i]['text']
    
    # Truncate if too long
    if len(text) > 200:
        text = text[:200] + "..."
    
    print(f"{i+1}. [{label}]")
    print(f"   {text}")
    print()
