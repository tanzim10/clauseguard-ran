import json

from clauseguard.cli.prepare_scenarios import main


def arguments(s):
    return ["--scenarios", str(s["scenarios"]), "--acquisition-dir", str(s["acquired"]),
            "--output-dir", str(s["output"]), "--trials", "128"]


def test_scored_command_keeps_blocked_diagnostics(release_sources, capsys):
    s = release_sources
    assert main(arguments(s)) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["quality"]["membership_counts"] == {"excluded": 39}
    assert (s["output"] / "release.json").exists()
    assert not report["quality"]["tasks"]["detection"]["kpi"]["ready"]


def test_exploratory_command_is_explicit_and_succeeds_on_coverage(release_sources, capsys):
    assert main([*arguments(release_sources), "--mode", "exploratory"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["quality"]["mode"] == "exploratory"
    assert not report["quality"]["tasks"]["detection"]["kpi"]["ready"]
    assert report["quality"]["tasks"]["detection"]["kpi"]["coverage_ready"]


def test_evidence_and_release_commands_are_separate(release_sources, capsys):
    s = release_sources
    reviews_path = s["output"].parent / "recording-reviews.json"
    reviews_path.write_text(json.dumps(s["reviews"]))
    assert main([*arguments(s), "--stage", "evidence",
                 "--recording-reviews", str(reviews_path)]) == 0
    assert not (s["output"] / "splits.json").exists()
    assert main([*arguments(s), "--stage", "release"]) == 0
    assert (s["output"] / "membership.csv").exists()
    assert main([*arguments(s), "--stage", "release"]) == 1  # Protect existing release.
