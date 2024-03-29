from os import remove
import pandas as pd
import numpy as np

def main():
    # Faz a leitura do arquivo
    names = ['Code-Number','Clump-Thickness','Cell-Size','Cell-Shape','Marginal-Adhesion','Single-Epithelial-Cell-Size','Bare-Nuclei','Bland-Chromatin','Normal-Nucleoli','Mitoses','Class'] 
    features = ['Clump-Thickness','Cell-Size','Cell-Shape','Marginal-Adhesion','Single-Epithelial-Cell-Size','Bare-Nuclei','Bland-Chromatin','Normal-Nucleoli','Mitoses','Class']
    output_file = 'Datasets/breast-cancer-output.data'
    input_file = 'Datasets/breast-cancer-wisconsin.data'
    df = pd.read_csv(input_file, # Nome do arquivo com dados
                     names = names, # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?') # Define que ? será considerado valores ausentes
    
    df_original = df.copy()
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 35 LINHAS\n")
    print(df.head(42))
    print("\n")        

    # Imprime informações sobre dos dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")
    
    # Imprime uma analise descritiva sobre dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")
    
    # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")
    

    
    columns_missing_value = df.columns[df.isnull().any()]
    print(columns_missing_value)
    method = 'mode' # number or median or mean or mode
    
    #for c in columns_missing_value:
     #   UptateMissingvalue(df, c)
    
    df = df.dropna()
     # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES APÓS REMOÇÃO\n")
    print(df.isnull().sum())
    print("\n")
    columns_missing_value = df.columns[df.isnull().any()]
    print(columns_missing_value)


    
    
    # print('Total valores ausentes: ' + str(df['Density'].isnull().sum()))
    print(df.describe())
    print("\n")
    print(df.head(42))
    print(df_original.head(42))
    print("\n")
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=False, index=False)  
    

def UptateMissingvalue(df, column, method="remove", number=0):
    if method == 'number':
        # Substituindo valores ausentes por um número
        df[column].fillna(1, inplace=True)
        #substitui valores de linhas especificas

       #df.loc[23,'Density']=6
       #df.loc[292,'Density']=6

    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        #median = df['Density'].median()
    #    df[column].fillna(median, inplace=True)
   # elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)
    elif method == 'remove':
        # Remover os valores faltantes
        remove = df[column].dropna(axis =0)  
        df[column].fillna(remove, inplace=True)


if __name__ == "__main__":
    main()
