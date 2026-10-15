"""Run fixture tests from a fresh clone; --with-artifacts requires private inputs."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# These tests explicitly consume excluded research artifacts; never hide other failures.
ARTIFACT_TESTS = {
    'test_phase2.SelectionAndJournalTests.test_actual_selection',
    'test_phase3.Tests.test_exact_phase2_selection',
    'test_phase4.Tests.test_actual_baseline_config',
    'test_phase4.Tests.test_pilot_extends_full_without_duplicate_keys',
    'test_phase5.Tests.test_actual_baseline_selection',
    'test_phase6.Tests.test_actual_60_question_matching',
    'test_phase6.Tests.test_preservation_of_earlier_sources_and_baseline_artifacts',
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-artifacts', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the repository root')
    suite = unittest.TestSuite()
    for test in flatten(unittest.defaultTestLoader.discover('tests', pattern='test_*.py')):
        if not args.with_artifacts and test.id() in ARTIFACT_TESTS:
            # Skip a fresh instance, without mutating historical test classes.
            name = test.id()
            def needs_artifacts(name=name):
                raise unittest.SkipTest(f'{name}: use --with-artifacts and documented baseline inputs')
            suite.addTest(unittest.FunctionTestCase(needs_artifacts, description=name))
        else:
            suite.addTest(test)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
