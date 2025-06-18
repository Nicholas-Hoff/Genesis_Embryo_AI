from feedback_hooks import FeedbackHooks


def test_feedback_hooks_poll(tmp_path):
    f1 = tmp_path / "log1.txt"
    f2 = tmp_path / "log2.txt"
    f1.write_text("ok\nerror happened\nall good\n")
    f2.write_text("FAIL occurred\nallow\naccess denied\n")

    hooks = FeedbackHooks([f1, f2])

    count1 = hooks.poll()
    assert count1 == 3
    pos1_first = hooks.cursors[f1]
    pos2_first = hooks.cursors[f2]

    # No new lines -> poll again should yield zero negatives
    count2 = hooks.poll()
    assert count2 == 0
    assert hooks.cursors[f1] == pos1_first
    assert hooks.cursors[f2] == pos2_first
