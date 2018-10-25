from pathlib import Path
from typing import Union, List, Generator, Dict
import json
import yaml

JSON_FORMAT = '.json'
YAML_FORMAT = ['.yml', '.yaml']


def generate_all_files(
    base_dir: Union[str, Path],
    allowed_suffix: List[str] = None,
) -> Generator[Path, None, None]:
    '''
    get all files recursively. it yield values one by one.
    this function is for preventing high memory consumption
    when many sub directories are there.

    :param base_dir: base directory to get files recursively.
    :param allowed_suffix: suffix to be allowed.
    '''
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    assert base_dir.exists()
    assert allowed_suffix is None or isinstance(allowed_suffix, list)

    for p in base_dir.glob('*'):
        if p.is_dir():
            generate_all_files(p)
        else:
            if allowed_suffix is None:
                yield p
            else:
                if p.suffix in allowed_suffix:
                    yield p

def get_all_files(
        base_dir: Union[str, Path],
        allowed_suffix: List[str],
) -> List[Path]:
    '''
    get all files recursively.
    this function consumes much RAM when the number of
    sub directories is large.
    recommend to use 'get_all_files_generator' when you
    don't need get all files all at once.

    :param base_dir: base directory to get files recursively.
    :param allowed_suffix: suffix to be allowd.
    '''
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    assert base_dir.exists()
    assert allowed_suffix is None or isinstance(allowed_suffix, list)

    file_path_list: List[Path] = []

    for p in base_dir.glob('*'):
        if p.is_dir():
            subvideo_path_list = get_all_files(p, allowed_suffix)
            for sp in subvideo_path_list:
                file_path_list.append(sp)
        else:
            if p.suffix in allowed_suffix:
                file_path_list.append(p)

    return file_path_list

def load_config(path: Union[str, Path]) -> Dict:
    '''
    config loader for json and yaml format.

    :param path: path to load.
    :return: parsed config file.
    '''
    if isinstance(path, str):
        path = Path(path)
    assert path.exists()

    with path.open() as f:
        if path.suffix == JSON_FORMAT:
            return json.load(f)
        elif path.suffix in YAML_FORMAT:
            return yaml.load(f)
        else:
            raise Exception('file_loader: config format is unknown.')
