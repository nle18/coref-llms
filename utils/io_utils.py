"""Most common input-output utility functions (txt, json, jsonl, csv, tsv)

Containing read, write, and some conversions
"""

from typing import List

import json
import csv


def read_txt(filepath: str, encoding: str = "utf-8") -> List:
    data = []
    with open(filepath, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def write_txt(data: list, filepath: str, encoding="utf-8") -> None:
    with open(filepath, "w", encoding=encoding) as f:
        for line in data:
            f.write(line + "\n")


def read_json(filepath: str, encoding: str = "utf-8") -> dict:
    with open(filepath, "r", encoding=encoding) as f:
        return json.load(f)


def write_json(d: dict, filepath: str, encoding="utf-8") -> None:
    with open(filepath, "w", encoding=encoding) as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


def read_jsonl(filepath: str, encoding: str = "utf-8") -> List[dict]:
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            data.append(example)
    return data


def write_jsonl(data: list, filepath: str, encoding="utf-8") -> None:
    with open(filepath, "w", encoding=encoding) as f:
        for example in data:
            f.write(json.dumps(example) + "\n")


def read_sv(filepath: str, sep: str = ",", encoding: str = "utf-8") -> List[dict]:
    """Read in seperated-value format (eg csv, tsv) into a list of dictionaries,
    where each dict represents a row

    Args:
        sep (`str`): token to seperate value, either `,` or `\t` (default `,`)
    """
    assert sep in [",", "\t"]
    data = []
    with open(filepath, mode="r", newline="", encoding=encoding) as file:
        reader = csv.DictReader(file, delimiter=sep)
        for row in reader:
            data.append(row)

    return data


def write_sv(
    data: list, filepath: str, fieldnames: List[str], sep: str = ",", encoding="utf-8"
) -> None:
    """Read in seperated-value format (eg csv, tsv) into a list of dictionaries,
    where each dict represents a row

    Args:
        fieldnames (List[str]): list of headers
    """
    assert sep in [",", "\t"]
    with open(filepath, "w", encoding=encoding) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=sep)
        writer.writeheader()
        writer.writerows(data)
