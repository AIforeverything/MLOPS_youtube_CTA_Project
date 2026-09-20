import pickle
def load_model(model_path:str):
    """
    Function to load the model.
    parameters:
    inputs:
    model_path:str
    outputs:
    model.pkl
    """
    try:
        with open(model_path,'rb') as f:
            model= pickle.load(f)
        return model
    except Exception as e:
        raise e
