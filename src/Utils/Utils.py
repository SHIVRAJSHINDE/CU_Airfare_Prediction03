import yaml
import importlib
def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data

def get_class(model):
    model_class = globals()[model.split('.')[-1]]  # Get the class from the model name
    model = model_class()  # Instantiate the model
    return model

def get_class_Scaler(model_path):
    # Split the module path and class name
    module_name, class_name = model_path.rsplit('.', 1)
    print(module_name, class_name)
    # Dynamically import the module
    module = importlib.import_module(module_name)
    print(module)

    # Get the class from the module
    model_class = getattr(module, class_name)
    print(model_class)
    # Instantiate the class
    return model_class()