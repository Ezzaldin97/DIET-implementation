import yaml

class Config:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)