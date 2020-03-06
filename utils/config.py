import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as cfg:
            self.yaml_file = cfg.read()
            self.config = yaml.load(self.yaml_file, Loader=yaml.FullLoader)

    def __getattr__(self, name):
        return self.config[name]