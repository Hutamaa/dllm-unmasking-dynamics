"""CPU public-table validation; no raw responses or model access."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from analyze_phase7 import load, paired_tests, paired_markdown, read_csv


class PublicTables(unittest.TestCase):
    def test_published_matched_counts_and_statistics(self):
        groups, summaries, strata = load(Path('reports/phase7'), Path('research/benchmark_selection.json'))
        self.assertEqual(len(groups), 17)
        counts = {r['method']: r['correct'] for r in summaries}
        self.assertEqual([counts[f'left_to_right P{p}'] for p in (1, 4, 8)], [43, 37, 5])
        self.assertEqual([counts[f'confidence P{p}'] for p in (1, 4, 8)], [26, 27, 26])
        rows = paired_tests(Path('reports/phase7'), groups, strata, 42, 200)
        self.assertEqual(rows, paired_tests(Path('reports/phase7'), groups, strata, 42, 200))
        contrast = next(r for r in rows if r['a'] == 'left_to_right P4' and r['b'] == 'left_to_right P8')
        self.assertEqual((contrast['recovered'], contrast['regressed']), (1, 33))
        self.assertAlmostEqual(contrast['mcnemar_exact_p'], 4.0745362639427185e-9)

    def test_canonical_bootstrap_provenance_and_report(self):
        source = Path('reports/phase7')
        metadata = json.loads((source / 'bootstrap_provenance.json').read_text())
        self.assertEqual(metadata['seed'], 42)
        self.assertEqual(metadata['bootstrap_repetitions'], 10000)
        self.assertEqual(metadata['script_sha256'], hashlib.sha256(Path('analyze_phase7.py').read_bytes()).hexdigest())
        for name, digest in metadata['input_sha256'].items():
            path = Path('research/benchmark_selection.json') if name == 'benchmark_selection.json' else source / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        groups, _, strata = load(source, Path('research/benchmark_selection.json'))
        regenerated = paired_tests(source, groups, strata, 42, 10000)
        saved = read_csv(source / 'paired_reanalysis.csv')
        self.assertEqual(len(saved), len(regenerated))
        for a, b in zip(saved, regenerated):
            for field, value in b.items():
                self.assertEqual(a[field], str(value))
        report = (source / 'PHASE7_REPORT.md').read_text()
        block = report.split('<!-- BEGIN REGENERATED PAIRED TABLE -->')[1].split('<!-- END REGENERATED PAIRED TABLE -->')[0].strip()
        self.assertEqual(block, paired_markdown(regenerated))

    def test_rejects_duplicate_mismatched_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            for name in ('method_summary.csv', 'selected_paired_tests.csv'):
                shutil.copyfile(Path('reports/phase7') / name, source / name)
            original = (Path('reports/phase7') / 'unified_per_question.csv').read_text()
            lines = original.splitlines(keepends=True)
            for text in (original + lines[1], ''.join(lines[:-1])):
                (source / 'unified_per_question.csv').write_text(text)
                with self.assertRaises(ValueError):
                    load(source, Path('research/benchmark_selection.json'))
            rows = list(csv.DictReader(lines))
            rows[0]['stratum'] = 'invalid'
            with (source / 'unified_per_question.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ValueError):
                load(source, Path('research/benchmark_selection.json'))


if __name__ == '__main__':
    unittest.main()
