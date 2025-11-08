import pandas as pd
import numpy as np

INPUT_CSV= 'data/raw/bank-additional-full.csv'
OUTPUT_CSV= 'data/processed/bank-processed.csv'

def preprocess_data(input_path= INPUT_CSV, output_path = OUTPUT_CSV):
    #Load the dataset
    df = pd.read_csv(input_path, sep=';')
    
    # Adaptar nombres de columnas
    df.columns = df.columns.str.replace(".", "_")

    # Transformar los valores 'unknown' en NaN
    df.replace('unknown', np.nan, inplace=True)

    # Se agrega el campo contacted_before
    df['contacted_before'] = np.where(df['pdays'] == 999, 'no', 'yes')

    # Se elimina la columna 'default' ya que tiene muchos valores desconocidos
    df.drop(columns=["default", "pdays"], inplace=True)

    # Se hace un filtro para eliminar las filas que tienen valores nulos
    df.dropna(inplace=True)
    
    # Se hace un filtro para eliminar las filas duplicadas
    df.drop_duplicates(inplace=True)

    # Mapear la columna objetivo 'y' a valores binarios
    map = {'yes': 1, 'no': 0}
    df['y'] = df['y'].map(map)


    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data()