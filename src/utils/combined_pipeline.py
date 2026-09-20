from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def combined_transform(numerical_columns:list[str],categorical_columns: list[str]):
    """ This function provides the combined pipeline for numerical and  categorical features"""

    # numeric pipeline: imputer-> log transform-> standardScaler
    numeric_pipeline= Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('log_transform',FunctionTransformer(np.log1p)),
            ('scaling',StandardScaler())
        ]
    )

    # categorical_pipeline: imputer -> onehot encoding
    categorical_pipeline= Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # applying above pipelines to given columns
    combined_processor= ColumnTransformer(
        transformers=[
            ('numeric',numeric_pipeline,numerical_columns),
            ('categorical',categorical_pipeline,categorical_columns)
        ]
    )

    return combined_processor

def model_with_combined_processor(combined_processor, model)->Pipeline:
    # model pipeline: processor -> model
    model_pipeline= Pipeline(
        steps=[
            ('combined_transform',combined_processor),
            ('model',model)
        ]
    )
    return model_pipeline

if __name__=="__main__":
    
    numeric_features = ['subscriber_count','channel_view_count','duration_seconds']
    categorical_features = ["category"]
    combined_processor_pipeline= combined_transform(numeric_features,categorical_features)
    model_pipeline = model_with_combined_processor(combined_processor_pipeline,RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1))