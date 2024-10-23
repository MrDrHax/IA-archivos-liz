import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'A': ['foo', 'bar', 'baz', 'qux', 'quux'],
    'B': ['one', 'one', 'two', 'three', 'four'],
    'C': ['x', 'y', 'z', 'w', 'v'],
    'D': [2, 3, 4, 5, 6]
})

# Define the range for column 'D'
lower_bound = 2
upper_bound = 4

# Create a boolean mask for the range
mask = (df['D'] >= lower_bound) & (df['D'] < upper_bound)

# Apply the mask to the DataFrame and select specific columns
df_subset = df.loc[mask, ['A', 'B']]

# Get the list of columns
columns = df_subset.columns.tolist()

print(f'Columns: {columns}')
print(df_subset)
