"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import tyro
import os
import numpy as np
import tqdm

NUM_EPS = 20
REPO_NAME = f"XCZhang/libero_object_20demos"  # Name of the output dataset, also used for the Hugging Face Hub

RAW_DATASET_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/env_repos/LIBERO/libero/datasets/libero_object_no_noops"

def main(data_dir: str=RAW_DATASET_DIR, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=50,
        image_writer_processes=10,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for file in tqdm.tqdm(os.listdir(data_dir)):
        if not file.endswith(".hdf5"):
            continue
        file_path = os.path.join(data_dir, file)
        print(f"Processing file: {file_path}")
        # compute language instruction
        raw_file_string = os.path.basename(file_path).split('/')[-1]
        words = raw_file_string[:-10].split("_")
        command = ''
        for w in words:
            if "SCENE" in w:
                command = ''
                continue
            command = command + w + ' '
        command = command[:-1]
        print(f"Computed command: {command}")
        with h5py.File(file_path, "r") as f:
            demos = f["data"].keys()
            for episode in tqdm.tqdm(list(demos)[:NUM_EPS]):
                print(f"Processing episode: {episode}")
                demo = f["data"][episode]
                for step_idx in range(len(demo["states"])):
                    dataset.add_frame(
                        {
                            "image": demo["obs"]["agentview_rgb"][step_idx][::-1],
                            "wrist_image": demo["obs"]["eye_in_hand_rgb"][step_idx][::-1],
                            "state": np.asarray(np.concatenate((demo["obs"]["ee_states"][step_idx], demo["obs"]["gripper_states"][step_idx]), axis=-1), dtype=np.float32),
                            "actions": np.asarray(demo["actions"][step_idx], dtype=np.float32),
                            "task": command,
                        }
                    )
                dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)