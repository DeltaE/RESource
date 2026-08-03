from RESource.cli import build_parser
from RESource.coders import default_coders_cfg_file_path, load_api_key


def test_cli_parses_minimal_run():
    args = build_parser().parse_args(["config/example.yaml"])

    assert args.config == "config/example.yaml"
    assert args.year == 2024


def test_missing_coders_credentials_are_optional(tmp_path):
    assert load_api_key(tmp_path / "missing.yaml") == (None, None)


def test_coders_credentials_use_dedicated_local_directory():
    assert default_coders_cfg_file_path == "credentials/coders_api.yaml"


def test_coders_credentials_support_current_list_template(tmp_path):
    credential_file = tmp_path / "coders_api.yaml"
    credential_file.write_text("api_keys:\n  - test-key\n", encoding="utf-8")

    assert load_api_key(credential_file) == ("test-key", "key_1")
