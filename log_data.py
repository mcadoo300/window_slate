import pandas as pd

# Read the Excel file into a DataFrame
#df = pd.read_excel('evaluation_pscores.xlsx')
#df = pd.read_excel('behavior_pscores.xlsx')
#df = pd.read_excel('evaluation_logits.xlsx')
# Display the DataFrame
"""
df = pd.read_excel(f'window_additive_0.5/3_250_random_error.xlsx')
print(df)
df2 = pd.read_excel(f'window_additive_0.5/3_250_optimal_error.xlsx')
print(df2)
result = pd.merge(df, df2.iloc[:,1], left_index=True, right_index=True,suffixes=('', '-optimal'))
df2 = pd.read_excel(f'window_additive_0.5/3_250_anti_error.xlsx')
print(df2)
result = pd.merge(result, df2.iloc[:,1], left_index=True, right_index=True,suffixes=('', '-anti'))
print(result)
print(result.shape)"""

result = pd.read_excel(f'window_additive_0.5/7_500_random_error.xlsx')
print(result)
result = pd.read_excel(f'window_additive_0.5/7_500_optimal_error.xlsx')
print(result)
result = pd.read_excel(f'window_additive_0.5/7_500_anti_error.xlsx')
print(result)
"""for i in [500,1000,2000,4000]:
    print(f"num rounds: {i}")
    df = pd.read_excel(f'window_additive_0.75/4_{i}_anti_error.xlsx')
    result = pd.merge(result, df.iloc[:,1], left_index=True, right_index=True,suffixes=('', f'-{i}'))
result.to_excel('window_additive_0.75/4_anti_output.xlsx',index=False)"""