import json
import pathlib


def write_ground_truth(path, ground_truth):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    return path


def read_ground_truth(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
