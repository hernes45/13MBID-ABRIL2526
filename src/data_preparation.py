import pandas as pd
import numpy as np

INPUT_CSV= 'data/raw/bank-additional-full.csv'
OUTPUT_CSV= 'data/processed/bank-processed.csv'

def preprocess_data(input_path= INPUT_CSV, output_path = OUTPUT_CSV):
    df = pd.read_csv(input_path, sep=';')
    
    # Se ajustan los nombres de las columnas para que no contengan puntos
    df.columns = df.columns.str.replace(".", "_")

    # Transformar los valores 'unknown' en NaN
    df.replace('unknown', np.nan, inplace=True)

    # Por la poca variedad en los datos y gran cantidad de nulos (20%), se elimina la columna "default"
    df.drop(columns=["default"], inplace=True)

    # Se hace un filtro para eliminar las filas que tienen valores nulos
    df.dropna(inplace=True)

    # Se hace un filtro para eliminar las filas duplicadas
    df = df.drop_duplicates()

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data()