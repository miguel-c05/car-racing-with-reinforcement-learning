"""Convert comparison results CSV to NumPy array and display statistics."""

import numpy as np
import pandas as pd

# Load CSV
df = pd.read_csv("comparison_results.csv")

# Convert to numpy array
data = df.to_numpy()

print("NumPy Array Shape:", data.shape)
print("\nFirst 10 rows:")
print(data[:10])

# Extract scores as numeric array
scores = df['Score'].to_numpy()

print("\n" + "="*60)
print("Statistics:")
print("="*60)
print(f"Total episodes: {len(scores)}")
print(f"Mean score: {scores.mean():.2f}")
print(f"Std deviation: {scores.std():.2f}")
print(f"Min score: {scores.min():.2f}")
print(f"Max score: {scores.max():.2f}")

# Group by model
print("\n" + "="*60)
print("Per-Model Statistics:")
print("="*60)
summary = df.groupby("Model Name")["Score"].agg(['mean', 'std', 'min', 'max', 'count'])
summary = summary.sort_values('mean', ascending=False)
print(summary)

# Save numpy array
np.save("comparison_results.npy", data)
print(f"\n✓ NumPy array saved to: comparison_results.npy")

# Save scores only
np.save("comparison_scores.npy", scores)
print(f"✓ Scores array saved to: comparison_scores.npy")
