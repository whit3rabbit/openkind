"""Execute the documented Arrow-to-Jev conversion with repeated Score labels."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


class ArrowDocsTests(unittest.TestCase):
    def test_python_example_preserves_score_positions_and_answer_shapes(self):
        page = (Path(__file__).resolve().parents[2] / "docs/ARROW.md").read_text()
        example = page.split("```python\n", 1)[1].split("```", 1)[0]
        ast.parse(example)
        row = {
            "refund": 0.7,
            "department": {
                "choice": 1,
                "confidence": 0.6,
                "probabilities": [0.1, 0.8, 0.1],
            },
            "urgency": {
                "score": 1.2,
                "confidence": 0.4,
                "probabilities": [0.2, 0.3, 0.5],
            },
        }
        table = SimpleNamespace(
            column_names=list(row),
            schema=[
                SimpleNamespace(metadata={b"jev.type": b"noul"}),
                SimpleNamespace(metadata={
                    b"jev.type": b"choice",
                    b"labels": b'["__none__", "returns", "shipping"]',
                }),
                SimpleNamespace(metadata={
                    b"jev.type": b"score",
                    b"legend": b'["Can wait", "Can wait", "Today"]',
                }),
            ],
            to_pylist=lambda: [row],
        )
        # Exercise the published loop without network or optional Python packages.
        conversion = example[example.index("for row in table.to_pylist():"):]
        exec(conversion, {"json": json, "table": table, "print": lambda *args: None})
        self.assertEqual(row["refund"], {"noul": 0.7})
        self.assertEqual(row["department"]["choice"], "returns")
        self.assertEqual(row["urgency"]["legend"], {
            "0": "Can wait", "1": "Can wait", "2": "Today",
        })
        self.assertEqual(row["urgency"]["probabilities"], {
            "0": 0.2, "1": 0.3, "2": 0.5,
        })


if __name__ == "__main__":
    unittest.main()
