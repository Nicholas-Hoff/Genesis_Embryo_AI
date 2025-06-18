from skill_synthesizer import SkillSynthesizer


def test_synthesize_skill_creates_file(tmp_path):
    synth = SkillSynthesizer(module_dir=tmp_path)
    path = synth.synthesize_skill("foo", "doc")
    assert path.exists()
    content = path.read_text()
    assert "def foo" in content
    assert '"""doc"""' in content
