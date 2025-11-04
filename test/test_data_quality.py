import pandas as pd
import numpy as np
from pandera.pandas import DataFrameSchema, Column
import pytest

@pytest.fixture
def datos_banco():
    return pd.read_csv("data/raw/bank-additional-full.csv", sep=";")

def test_esquema(datos_banco):
    df = datos_banco
    #defini el esquema esperado
    esquema = DataFrameSchema({
        "age": Column(int, nullable=False),
        "job":  Column(str, nullable=False),
        "marital":  Column(str, nullable=False),
        "education":  Column(str, nullable=False),
        "default":  Column(str, nullable=True),
        "housing":  Column(str, nullable=False),
        "loan":  Column(str, nullable=False),
        "contact":  Column(str, nullable=False),
        "month": Column(str, nullable=False),
        "day_of_week": Column(str, nullable=False),
        "duration": Column(int, nullable=False),
        "campaign": Column(int, nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, nullable=False),
        "poutcome": Column(str, nullable=False),
        "emp.var.rate": Column(float, nullable=False),
        "cons.price.idx": Column(float, nullable=False),
        "cons.conf.idx": Column(float, nullable=False),
        "euribor3m": Column(float, nullable=False),
        "nr.employed": Column(float, nullable=False),
        "y": Column(str, nullable=False),
    })


    # Validar la estructura del DataFrame
    esquema.validate(df)

def test_basico(datos_banco):
    df = datos_banco

    assert not df.empty
    assert df.isnull().sum().sum() == 0 #El dataframe contiene valores nulos
    assert df.shape[1] == 21 # debe tener 21 columnas segun la documentacion


if __name__ == "__main__":
    try:
        test_esquema(datos_banco())
        test_basico(datos_banco())

        print("¡¡Todos los test pasaron!!")
        with open("docs/test_results/test_results.txt", "w") as f:
            f.write(f"Todos los tests pasaro exitosamentes\n")
    except AssertionError as e:
        print(f"Test fallido: {e}")
        with open("docs/test_results/test_results.txt", "w") as f:
            f.write(f"Test fallido: {e}\n")