from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from advice_reflection_platform.backend.artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_init_db_adds_new_columns_to_existing_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            runs_dir = base_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            db_path = runs_dir / "metadata.sqlite"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE runs (
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
                        raw_path TEXT NOT NULL
                    )
                    """
                )
            ArtifactStore(base_dir)
            with sqlite3.connect(db_path) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            self.assertIn("run_mode", columns)
            self.assertIn("advice_text", columns)
            self.assertIn("parser_model_name", columns)
            self.assertIn("mixed_or_conditional", columns)
            self.assertIn("parser_secondary_fit", columns)
            self.assertIn("parser_why_not_clean_fit", columns)


if __name__ == "__main__":
    unittest.main()
