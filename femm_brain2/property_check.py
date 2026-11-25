"""Read and compare property records saved by native magnetic FEMM."""
import json
import math
from pathlib import Path
import re

SECTIONS = {
    "materials": ("BlockProps", "Block", "BlockName"),
    "boundaries": ("BdryProps", "Bdry", "BdryName"),
    "points": ("PointProps", "Point", "PointName"),
    "circuits": ("CircuitProps", "Circuit", "CircuitName"),
}


def read_properties(path):
    text = Path(path).read_text()
    result = {}
    for section, (heading, tag, name_key) in SECTIONS.items():
        count = re.search(r"(?m)^\[" + heading + r"\]\s*=\s*(\d+)\s*$", text)
        blocks = re.findall(r"(?ms)^\s*<Begin" + tag + r">\s*\n(.*?)^\s*<End" + tag + r">\s*$", text)
        if not count or len(blocks) != int(count.group(1)):
            raise ValueError("Invalid property count: " + section)
        records = []
        for block in blocks:
            record = {}
            lines = block.splitlines()
            for index, line in enumerate(lines):
                field = re.fullmatch(r"\s*<([^>]+)>\s*=\s*(.*?)\s*", line)
                if not field:
                    continue
                key, value = field.groups()
                if key in record:
                    raise ValueError("Duplicate property field: " + key)
                value = json.loads(value) if value.startswith('"') else float(value)
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Nonfinite property field: " + key)
                record[key] = value
                if key == "BHPoints":
                    if value < 0 or not value.is_integer():
                        raise ValueError("Invalid B-H point count")
                    rows = [[float(v) for v in row.split()] for row in lines[index+1:index+1+int(value)]]
                    if len(rows) != int(value) or any(len(row) != 2 or not all(math.isfinite(v) for v in row) for row in rows):
                        raise ValueError("Invalid B-H point data")
                    record["BHdata"] = rows
            if not isinstance(record.get(name_key), str):
                raise ValueError("Missing property name: " + section)
            records.append(record)
        if len({r[name_key] for r in records}) != len(records):
            raise ValueError("Duplicate property names: " + section)
        result[section] = records
    return result


def _equal(actual, expected, tolerance):
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _equal(a, b, tolerance) for a, b in zip(actual, expected))
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and abs(actual-expected) <= tolerance
    return actual == expected


def evaluate_properties(path, expected, tolerance):
    observed = read_properties(path)
    errors = []
    for section, records in expected.items():
        actual = observed[section]
        name_key = SECTIONS[section][2]
        if len(actual) != len(records):
            errors.append(f"{section}: expected {len(records)}, observed {len(actual)}")
        by_name = {r[name_key]: r for r in actual}
        for record in records:
            name = record[name_key]
            if name not in by_name:
                errors.append(section + ": missing " + name)
                continue
            for key, value in record.items():
                if key not in by_name[name] or not _equal(by_name[name][key], value, tolerance):
                    errors.append(f"{section}/{name}/{key}: expected {value}, observed {by_name[name].get(key)}")
    return observed, errors
