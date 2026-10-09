import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Iris dataset
df = pd.read_csv('Iris.csv')

# Drop 'Id' column if present as it's just an index
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# 2. Display summary statistics
print("--- Dataset Information ---")
print(df.info())

print("\n--- Summary Statistics (Overall) ---")
print(df.describe())

print("\n--- Summary Statistics (Grouped by Species) ---")
print(df.groupby('Species').mean())

# 3. Data Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Pairplot / Feature Distributions across Species
sns.boxplot(ax=axes[0, 0], x='Species', y='PetalLengthCm', data=df)
axes[0, 0].set_title('Petal Length Distribution by Species')

sns.boxplot(ax=axes[0, 1], x='Species', y='SepalLengthCm', data=df)
axes[0, 1].set_title('Sepal Length Distribution by Species')

# Plot 2: Scatter plot of Petal Length vs Petal Width
sns.scatterplot(ax=axes[1, 0], x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df, s=70)
axes[1, 0].set_title('Petal Length vs Petal Width')

# Plot 3: Feature Correlation Heatmap
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()
