"""Long-session chunks retain early frames and explicit DOF order."""

import json

import numpy as np

from sim2sense_fall.humans.recording import ChunkRecorder


def test_chunk_boundaries_and_final_partial_preserve_every_frame(tmp_path):
    recorder = ChunkRecorder(tmp_path, ("z_joint", "a_joint"), 3)
    for frame in range(8):
        recorder.append(
            {
                "time_s": float(frame),
                "mode": "stand",
                "joints": np.array([frame, -frame]),
                "contacts": [],
                "contact_detail": [],
                "floor_contact_slips_m_s": [],
            }
        )
        recorder.flush()
    recorder.flush(final=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total_frames"] == 8
    assert [row["frames"] for row in manifest["chunks"]] == [3, 3, 2]
    arrays = [np.load(tmp_path / row["control"]) for row in manifest["chunks"]]
    assert all(list(a["dof_names"]) == ["z_joint", "a_joint"] for a in arrays)
    assert np.array_equal(np.concatenate([a["time_s"] for a in arrays]), np.arange(8))
