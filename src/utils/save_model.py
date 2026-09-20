import pickle
def save_model(model_save_path:str,model_pipeline)->None:
    """
    Function to load the model.
    parameters:
    inputs:
    model_save_path:str
    outputs:
    None
    """
    try: 
        with open(model_save_path,'wb') as model_dump:
            pickle.dump(model_pipeline,model_dump)
    except Exception as e:
        raise e 