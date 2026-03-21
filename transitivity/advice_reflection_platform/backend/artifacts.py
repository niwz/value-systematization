from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from .schemas import RunRecord, ScenarioRunBundle


class ArtifactStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.runs_raw_dir = self.base_dir / "runs" / "raw"
        self.runs_summaries_dir = self.base_dir / "runs" / "summaries"
        self.exports_dir = self.base_dir / "exports"
        self.db_path = self.base_dir / "runs" / "metadata.sqlite"
        for path in (self.runs_raw_dir, self.runs_summaries_dir, self.exports_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    presentation_order TEXT NOT NULL,
                    repeat_idx INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    canonical_choice TEXT,
                    parse_provenance TEXT NOT NULL,
                    within_response_revision INTEGER NOT NULL,
                    thinking INTEGER NOT NULL DEFAULT 0,
                    raw_path TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            if "thinking" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN thinking INTEGER NOT NULL DEFAULT 0")

    def write_bundle(self, bundle: ScenarioRunBundle) -> tuple[Path, Path]:
        slug = f"{bundle.scenario.scenario_id}_{bundle.baseline.run_id[:8]}"
        raw_path = self.runs_raw_dir / f"{slug}.jsonl"
        rows = bundle.to_rows()
        with open(raw_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        summary_path = self.write_summary(rows, f"{slug}.csv")
        self._insert_rows(bundle_rows=[bundle.baseline, *( [bundle.reflection] if bundle.reflection else [] )], raw_path=raw_path)
        return raw_path, summary_path

    def write_records(self, records: list[RunRecord], filename_prefix: str) -> tuple[Path, Path]:
        raw_path = self.runs_raw_dir / f"{filename_prefix}.jsonl"
        rows = [record.to_flat_dict() for record in records]
        with open(raw_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        summary_path = self.write_summary(rows, f"{filename_prefix}.csv")
        self._insert_rows(bundle_rows=records, raw_path=raw_path)
        return raw_path, summary_path

    def write_summary(self, rows: list[dict[str, object]], filename: str) -> Path:
        summary_path = self.runs_summaries_dir / filename
        if not rows:
            summary_path.write_text("", encoding="utf-8")
            return summary_path
        with open(summary_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return summary_path

    def _insert_rows(self, bundle_rows: list[RunRecord], raw_path: Path) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO runs (
                    run_id,
                    scenario_id,
                    model_name,
                    condition,
                    presentation_order,
                    repeat_idx,
                    timestamp,
                    canonical_choice,
                    parse_provenance,
                    within_response_revision,
                    thinking,
                    raw_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.run_id,
                        record.scenario_id,
                        record.model_name,
                        record.condition,
                        record.presentation_order,
                        record.repeat_idx,
                        record.timestamp,
                        record.canonical_choice,
                        record.parsed.parse_provenance,
                        int(record.parsed.within_response_revision),
                        int(record.thinking),
                        str(raw_path),
                    )
                    for record in bundle_rows
                ],
            )
