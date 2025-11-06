"""
Script para entrenar un modelo de clasificacion para usar la mejor tecnica
"""
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler 
from sklearn.utils import resample

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression

def load_data(path):
    df = pd.read_csv(path)

    X = df.drop('y_yes', axis=1)
    y = df['y_yes']


    return train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

def create_preprocessor(X_train):
    # Se separan las columnas numéricas
    numerical_columns=X_train.select_dtypes(exclude='object').columns
    categorical_columns=X_train.select_dtypes(include='object').columns

    X_train = X_train.copy()
    int_columns = X_train.select_dtypes(include='int').columns
    for col in int_columns:
        X_train[col] = X_train[col].astype('float')
    
    numerical_columns=X_train.select_dtypes(exclude='object').columns
    # Pipeline para valores numéricos
    num_pipeline = Pipeline(steps=[
        ('RobustScaler', RobustScaler())
    ])

    # Pipeline para valores categóricos
    cat_pipeline = Pipeline(steps=[
        ('OneHotEncoder', OneHotEncoder(drop='first',sparse_output=False))
    ])

    # Se configuran los preprocesadores
    preprocessor_full = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_columns),
        ('cat_pipeline', cat_pipeline, categorical_columns)
    ]).set_output(transform='pandas')

    return preprocessor_full, X_train

def balance_data(X, y, random_state=42):
    # Combinar los datos preprocesados con las etiquetas
    train_data = X.copy()
    train_data['target'] = y.reset_index(drop=True)

    # Separar por clase
    class_0 = train_data[train_data['target'] == 0]
    class_1 = train_data[train_data['target'] == 1]

    # Encontrar la clase minoritaria
    min_count = min(len(class_0), len(class_1))

    # Submuestreo balanceado - tomar una muestra igual al tamaño de la clase minoritaria
    class_0_balanced = resample(class_0, n_samples=min_count, random_state=random_state)
    class_1_balanced = resample(class_1, n_samples=min_count, random_state=random_state)

    # Combinar las clases balanceadas
    balanced_data = pd.concat([class_0_balanced, class_1_balanced])

    # Separar características y objetivo
    x_train_resampled = balanced_data.drop('target', axis=1)
    y_train_resampled = balanced_data['target']

    return x_train_resampled, y_train_resampled