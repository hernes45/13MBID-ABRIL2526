import pandas as pd
import numpy as np
from pandera.pandas import DataFrameSchema, Column
import pytest

@pytest.fixture
def datos_banco():
    return pd.read_csv("data/raw/bank-additional-full.csv", sep=";")

def test_great_expectations(datos_banco):
    df = datos_banco
    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count":0, "total_count":0}
    }

    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1

        result = {
            "expectation": expectation_name,
            "success": condition,
            "message": message
        }

        if condition:
            results["statistics"]["success_count"] += 1
        else:
            results["success"] = False

        results["expectations"].append(result)
    
    #Validaciones
    add_expectation(
        "age_range", 
        df["age"].between(18, 100).all(),
        "La columna 'age' no esta entre el rango esperado[18-100]")
    
    add_expectation(
        "target_values", 
        df["y"].isin(["yes", "no"]).all(),
        "La columna 'y' contiene valores no validos")
    
    meses_validos = ["jan", "feb", "mar", "apr", "may", "jun", "jul",
                     "aug", "sep", "oct", "nov", "dec"]
    add_expectation(
        "month_valid_values",
        df["month"].isin(meses_validos).all(),
        "La columna 'month' contiene valores no válidos."
    )

    numeric_cols = [
        "age", "duration", "campaign", "pdays", "previous",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed"
    ]
    add_expectation(
        "numeric_types",
        all(np.issubdtype(df[col].dtype, np.number) for col in numeric_cols),
        "Alguna columna numérica no tiene tipo numérico."
    )

    marital_validos = ["single", "married", "divorced"]
    add_expectation(
        "marital_valid_values",
        df["marital"].isin(marital_validos).all(),
        f"La columna 'marital' contiene valores fuera de {marital_validos}."
    )
