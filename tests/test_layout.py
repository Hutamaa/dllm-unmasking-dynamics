"""Structural equivalence, provenance, packaging and CLI checks; no model use."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dllm_unmasking.layout import historical_bytes, historical_digest, repository_root, source_matches, source_path, source_records
from dllm_unmasking.phase2_data import sha


class LayoutTests(unittest.TestCase):
    def test_exact_pre_move_source_reconstruction(self):
        for legacy, record in source_records().items():
            self.assertEqual(historical_digest(record['path']), record['sha256'], legacy)
            self.assertEqual(source_path(legacy), repository_root() / record['path'])

    def test_computation_edits_cannot_match_historical_source(self):
        path = source_path('phase2_samplers.py')
        original = path.read_bytes()
        changed = original.replace(b'seed=42', b'seed=43', 1)
        self.assertNotEqual(changed, original)
        with self.assertRaises(ValueError):
            historical_bytes(path, changed)
        self.assertFalse(source_matches(path, '0' * 64))

    def test_new_hashes_remain_actual_file_hashes(self):
        path = source_path('phase2_data.py')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        legacy = source_records()['phase2_data.py']['sha256']
        self.assertNotEqual(actual, legacy)
        self.assertEqual(sha('phase2_data.py'), actual)
        self.assertTrue(source_matches('phase2_data.py', actual))
        self.assertTrue(source_matches('phase2_data.py', legacy))

    def test_installed_cli_help_outside_checkout(self):
        modules = [f'experiments.phase{i}.run' for i in range(1, 7)] + [
            'dllm_unmasking.analyze_phase15', 'dllm_unmasking.analyze_phase25',
            'dllm_unmasking.analyze_phase6', 'dllm_unmasking.analyze_phase7']
        with tempfile.TemporaryDirectory() as tmp:
            for module in modules:
                result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, (module, result.stderr))
                self.assertIn('usage:', result.stdout)
