#!pip install json5

import json5

def parse_config_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json5.load(file)

    print(f"parse file: {filepath}, config: {data}")

    for key, value in data.items():
        globals()[key] = value

parse_config_file("config.json5")

if __name__ == "__main__":
    print(f"name: {name}")