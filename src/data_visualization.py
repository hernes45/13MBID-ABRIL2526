# Importación de librerías y supresión de advertencias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualizar_datos(fuente: str = 'data/raw/bank-additional-full.csv',
                     salida: str = 'docs/figures/'):
    ''' Genera una serie de gráficos sobre los datos y los exporta
    Args:
        fuente: Ruta al archivo de datos
        salida: ruta al directorio donde se vuelcan las imagenes
    '''

    # Crear el directorio de salida si no existe
    Path(salida).mkdir(parents=True, exist_ok=True)

    #Leer los datos
    df = pd.read_csv(fuente, sep=';')
    
    #Sacamos la distribución de la variable objetivo
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo (suscripción al depósito)")
    plt.xlabel("¿Suscribió un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.savefig(f"{salida}/grafico_1.png")
    plt.close()
    
    #Grafica nivel educacional
    col = 'education'
    plt.figure(figsize=(6, 4))
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f"{salida}/grafico_education.png")
    plt.close()

    #Grafica estado civil
    col = 'marital'
    plt.figure(figsize=(6, 4))
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f"{salida}/grafico_marital.png")
    plt.close()

    #Matriz de correlacion
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones')
    plt.savefig(f"{salida}/grafico_correlation_matrix.png")
    plt.close()
  
    

if __name__ == "__main__":
    visualizar_datos()