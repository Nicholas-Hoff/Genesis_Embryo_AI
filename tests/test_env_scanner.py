import env_scanner


def test_scan_environment_has_expected_attributes():
    info = env_scanner.scan_environment()
    attrs = [
        "os",
        "release",
        "version",
        "machine",
        "hostname",
        "is_container",
    ]
    for attr in attrs:
        assert hasattr(info, attr), f"Missing attribute: {attr}"

    assert isinstance(info.is_container, bool)
