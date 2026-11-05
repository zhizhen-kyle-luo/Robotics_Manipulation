import torch

import ppo_runs


def save(path, iteration, global_step):
    torch.save(dict(agent={}, optimizer={}, iteration=iteration, global_step=global_step), path)


def test_latest_checkpoint_prefers_the_highest_iteration(tmp_path):
    for i in [1, 26, 51]:
        save(tmp_path / f"ckpt_{i}.pt", i, (i - 1) * ppo_runs.BATCH_SIZE)
    assert ppo_runs.latest_checkpoint(tmp_path).name == "ckpt_51.pt"
    # a finished 10M run continues into a longer run from its final checkpoint
    save(tmp_path / "final_ckpt.pt", 306, 305 * ppo_runs.BATCH_SIZE)
    assert ppo_runs.latest_checkpoint(tmp_path).name == "final_ckpt.pt"


def test_latest_checkpoint_skips_a_write_cut_off_by_a_disconnect(tmp_path):
    save(tmp_path / "ckpt_26.pt", 26, 0)
    (tmp_path / "ckpt_51.pt").write_bytes(b"\x80\x02partial")
    assert ppo_runs.latest_checkpoint(tmp_path).name == "ckpt_26.pt"


def test_finished_compares_the_final_step_with_the_requested_total(tmp_path):
    assert not ppo_runs.finished(tmp_path, 10_000_000)
    # ppo.py runs total // batch iterations, so a 10M run stops at 305 batches
    save(tmp_path / "final_ckpt.pt", 306, 305 * ppo_runs.BATCH_SIZE)
    assert ppo_runs.finished(tmp_path, 10_000_000)
    assert not ppo_runs.finished(tmp_path, 75_000_000)


def test_command_resumes_with_the_same_flags(tmp_path):
    cmd = ppo_runs.ppo_command("run", ["--seed", "2"], 75_000_000, checkpoint=tmp_path / "ckpt_51.pt")
    assert cmd[1:] == ["ppo.py", "--exp-name", "run", "--total-timesteps", "75000000", "--no-capture-video",
                       "--seed", "2", "--checkpoint", str(tmp_path / "ckpt_51.pt")]
