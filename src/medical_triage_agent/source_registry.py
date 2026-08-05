from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRecord:
    id: str
    name: str
    url: str
    license: str
    languages: tuple[str, ...]
    intended_use: tuple[str, ...]
    status: str
    notes: str

    @property
    def ingestible(self) -> bool:
        return self.status == "verified" and self.license not in {"", "to_verify", "unknown"}


def load_source_registry(path: Path = Path("docs/data-sources.md")) -> dict[str, SourceRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        raise ValueError(f"{path} does not contain a source table")

    header = _cells(table_lines[0])
    expected = ["id", "name", "url", "license", "languages", "intended_use", "status", "notes"]
    if header != expected:
        raise ValueError(f"{path} source table columns must be: {', '.join(expected)}")

    records: dict[str, SourceRecord] = {}
    for line in table_lines[2:]:
        cells = _cells(line)
        if len(cells) != len(expected):
            raise ValueError(f"invalid source table row: {line}")
        record = SourceRecord(
            id=cells[0],
            name=cells[1],
            url=cells[2],
            license=cells[3],
            languages=_split_csv(cells[4]),
            intended_use=_split_csv(cells[5]),
            status=cells[6],
            notes=cells[7],
        )
        if record.id in records:
            raise ValueError(f"duplicate source id: {record.id}")
        records[record.id] = record
    return records


def validate_source_ids(source_ids: list[str], registry: dict[str, SourceRecord]) -> None:
    for source_id in source_ids:
        if source_id not in registry:
            raise ValueError(f"unknown source id: {source_id}")
        if not registry[source_id].ingestible:
            raise ValueError(f"source is not ingestible: {source_id}")


def validate_source_for_use(
    source_id: str,
    intended_use: str,
    registry: dict[str, SourceRecord],
) -> SourceRecord:
    validate_source_ids([source_id], registry)
    source = registry[source_id]
    if intended_use not in source.intended_use:
        raise ValueError(f"source {source_id} is not registered for {intended_use}")
    return source


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())
