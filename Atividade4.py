import pandas as pd
import numpy as np
import statistics
import math
from scipy import stats
import numpy
from collections import Counter

def main():
    # Faz a leitura do arquivo
    input_file = 'Datasets/breast-cancer-wisconsin.data'
    names = ['Code-Number','Clump-Thickness','Cell-Size','Cell-Shape','Marginal-Adhesion','Single-Epithelial-Cell-Size','Bare-Nuclei','Bland-Chromatin','Normal-Nucleoli','Mitoses','Class']
    features = ['Bare-Nuclei']
    target = 'Severity'
   data= df.target
  

    
    
    """ pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

data = df.Bare_Nuclei """

mediana = statistics.median(data)
#mediaHarmonica = statistics.harmonic_mean(df.Bare-Nuclei)
#desvioP = statistics.pstdev(df.Bare-Nuclei)
#variancia = statistics.pvariance(df.Bare-Nuclei)


print("Média aritmética: ", mediana)
#print("Média harmônica: ", mediaHarmonica)
#print("Média desvio padrao: ", desvioP)
#print("Média variancia: ", variancia)

if __name__ == "__main__":
    main()

""" def main():
    # Faz a leitura do arquivo
    names = ['Bare-Nuclei'] 
    features = ['Bare-Nuclei']
    output_file = 'Atividade4.data'
    input_file = 'Datasets/breast-cancer-output.data'
    df = pd.read_csv(input_file, # Nome do arquivo com dados
                     names = names, # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                    na_values='?') # Define que ? será considerado valores ausentes

    


if __name__ == "__main__":
    main() """