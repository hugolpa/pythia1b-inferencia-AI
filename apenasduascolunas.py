import pandas as pd

df = pd.read_csv('medquad_data_copy.csv')

# Mantém apenas as colunas Question e Answer
df = df[['Question', 'Answer']]

# Preenche valores vazios com NaN
df = df.fillna(value=pd.np.nan)

# Reseta o índice
df = df.reset_index(drop=True) 

# Salva o dataframe padronizado
df.to_csv('data_padronizada.csv', index=False)