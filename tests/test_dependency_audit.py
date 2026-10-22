import pytest

from scripts.audit_dependencies import check_coverage, requirement_lines


def test_cpu_torch_version_is_audited_as_upstream_version():
    lines = requirement_lines({'torch': '2.13.0+cpu', 'numpy': '2.5.1'})
    assert 'torch==2.13.0\n' in lines
    assert 'numpy==2.5.1\n' in lines


def test_audit_fails_if_a_dependency_was_silently_skipped():
    with pytest.raises(ValueError, match='torch'):
        check_coverage({'dependencies': [{'name': 'torch', 'skip_reason': 'not found on PyPI'}]})


def test_vulnerabilities_are_counted_in_a_complete_audit():
    report = {'dependencies': [{'name': 'safe', 'vulns': []},
                               {'name': 'unsafe', 'vulns': [{'id': 'example'}]}]}
    assert check_coverage(report) == 1
