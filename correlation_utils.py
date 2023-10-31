import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


# Binary target, continuous features correlation
correlated_cols = []
for col in df:
    corr, p_value = scipy.stats.pointbiserialr(df.target, df[col])
    if p_value<0.05:
        print(f'{col}: corr = {corr}, p_value = {p_value}')
        correlated_cols.append(col)
        
correlated_cols.append('target')



# Drop correlated features
correlation_matrix = df.corr().round(2).abs()

corr_pairs = []
for i in range(len(correlation_matrix)):
    for j in range(i+1, len(correlation_matrix)):
        if correlation_matrix.iloc[i,j] >= 0.9:
            corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

cols_to_drop = []
for pair in corr_pairs:
    var1, var2 = pair[0], pair[1]
    if (var1!='target') and (var2!='target'):
        if correlation_matrix[var1]['target'] > correlation_matrix[var2]['target']:
            cols_to_drop.append(var2)
        else:
            cols_to_drop.append(var1)