import yaml

def yaml_loader(file_path:str):
    with open(file_path,'r') as f:
        data_yaml= yaml.safe_load(f)
    return data_yaml

if __name__=="__main__":
    yaml_loader('./params.yaml')