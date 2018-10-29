from typing import Union, Dict
from pathlib import Path
import json
import yaml

JSON_FORMAT = '.json'
YML_FORMAT = '.yml'


def load(config_path: Union[str, Path]) -> Dict:
    '''
    load config file.

    :param config_path: path to config file.
    :return dict of configurations.
    '''
    if isinstance(config_path, str):
        config_path = Path(config_path)
    assert config_path.exists()

    with config_path.open() as f:
        if config_path.suffix == JSON_FORMAT:
            return json.load(f)
        elif config_path.suffix == YML_FORMAT:
            return yaml.load(f)
        else:
            raise Exception('config_loader: config format is unknown.')

def save(args, save_dir: Union[str, Path], out_name: Union[str, Path]='arguments.yml') -> None:
    '''
    save model parameters based on out_name's suffix.

    :param save_dir: save directory.
    :param out_name: output name to save.
    '''

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if isinstance(out_name, str):
        out_name = Path(out_name)

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    arguments = {}
    for (key, value) in vars(args).items():
        arguments[key] = value

    save_path = save_dir / out_name
    with save_path.open('w') as f:
        if out_name.suffix == JSON_FORMAT:
            json.dump(arguments, f)
        elif out_name.suffix == YML_FORMAT:
            yaml.dump(arguments, default_flow_style=False)
        else:
            raise Exception('config_loader: config format is unknown.')
