import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Define o stilo para ggplot
plt.style.use("ggplot")
# Define as configurações dos plots
# Cada plot terá o mesmo tamanho de figuras (10,5)
fig, (ax1) = plt.subplots(1, figsize=(10,5))

# Dados para cada subplot
ax1.bar([1,2,3,4,5],[4,3,2,4,7])
#ax2.barh([0.5,1,2.5],[0,1,2])

ax1.set(title="Gráfico de Barras Verticais", xlabel="Classes", ylabel=" Frequencia")
#ax2.set(title="Gráfico de Barras Horizontais", xlabel="Eixo x", ylabel="Eixo y")

plt.show()