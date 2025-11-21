"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize depth observations along with image observations
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --render_depth_names agentview_depth \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import random

from pathlib import Path
import multiprocessing as mp
import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros
from robosuite.environments.manipulation.empty_env import SingleArmEmptyEnv

import libero.libero.utils.utils as libero_utils
from PIL import Image
from robosuite.utils import camera_utils
from libero.libero.envs import *
from libero.libero import get_libero_path

# Ensure safe multiprocessing and reduce thread oversubscription early
try:
    mp.set_start_method("spawn", force=True)  # avoid fork with OpenGL / imageio backends
except RuntimeError:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass


class VideoWriterManager:
    """Context manager to handle imageio video writers using temporary files
    and atomic rename on close. Ensures corresponding pose and video files
    are finalized together and cleans up stale .tmp files.
    """
    def __init__(self, video_dir, camera_names, seg_idx, res=128, fps=20):
        self.video_dir = video_dir
        self.camera_names = camera_names
        self.seg_idx = seg_idx
        self.res = res
        self.fps = fps

        self.video_writers = {}
        self.pose_video_writers = {}

        self.video_tmp_paths = {}
        self.video_final_paths = {}
        self.pose_tmp_paths = {}
        self.pose_final_paths = {}

    def __enter__(self):
        # Open writers for cameras whose final files don't already exist.
        for camera_name in self.camera_names:
            camera_video_final = os.path.join(self.video_dir, f"{camera_name}_seg{self.seg_idx}.mp4")
            base, ext = os.path.splitext(camera_video_final)
            camera_video_tmp = f"{base}.tmp{ext}"  # ensure plugin sees .mp4 extension
            # Skip if final already exists (finals are only created on successful close)
            if os.path.exists(camera_video_final):
                continue
            # Remove stale tmp if present
            if os.path.exists(camera_video_tmp):
                os.remove(camera_video_tmp)
            # create writer to tmp file
            self.video_writers[camera_name] = imageio.get_writer(camera_video_tmp, fps=self.fps)
            self.video_tmp_paths[camera_name] = camera_video_tmp
            self.video_final_paths[camera_name] = camera_video_final

            if 'robot' not in camera_name:
                pose_video_final = os.path.join(self.video_dir, f"{camera_name}_seg{self.seg_idx}_pose.mp4")
                pbase, pext = os.path.splitext(pose_video_final)
                pose_video_tmp = f"{pbase}.tmp{pext}"
                # Skip if final already exists
                if os.path.exists(pose_video_final):
                    continue
                if os.path.exists(pose_video_tmp):
                    os.remove(pose_video_tmp)
                self.pose_video_writers[camera_name] = imageio.get_writer(pose_video_tmp, fps=self.fps)
                self.pose_tmp_paths[camera_name] = pose_video_tmp
                self.pose_final_paths[camera_name] = pose_video_final

        return self

    def __exit__(self, exc_type, exc, tb):
        # Finalize per-camera: close writers then atomically rename their tmp files
        cams_to_finalize = set()
        cams_to_finalize.update(self.video_tmp_paths.keys())
        cams_to_finalize.update(self.pose_tmp_paths.keys())

        for cam_name in cams_to_finalize:
            vw = self.video_writers.get(cam_name)
            pw = self.pose_video_writers.get(cam_name)
            if vw is not None:
                vw.close()
            if pw is not None:
                pw.close()

            v_tmp = self.video_tmp_paths.get(cam_name)
            v_final = self.video_final_paths.get(cam_name)
            p_tmp = self.pose_tmp_paths.get(cam_name)
            p_final = self.pose_final_paths.get(cam_name)

            # Rename video then pose
            if v_tmp and v_final and os.path.exists(v_tmp):
                os.replace(v_tmp, v_final)
            if p_tmp and p_final and os.path.exists(p_tmp):
                os.replace(p_tmp, p_final)

        # Cleanup any leftover tmp files
        for tmp_path in list(self.video_tmp_paths.values()) + list(self.pose_tmp_paths.values()):
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

def noise_fn(states: np.ndarray, actions: np.ndarray, noise: float, env, gripper_only=False, min_len: int=40):
    if noise == 0:
        return states, actions
    traj_len = actions.shape[0]
    if min_len is None:
        min_len = traj_len
    noising_start=np.random.choice(traj_len - min_len) if traj_len > min_len else 0 # at least 8 frames (40 steps)
    if actions.shape[-1] == 14:
        gripper_index = [6, 13]
    elif actions.shape[-1] == 7:
        gripper_index = [6,]
    else:
        gripper_index = []
    if len(gripper_index) > 0:
        actions[noising_start:, gripper_index] = actions[noising_start:, gripper_index] * ((np.random.rand(traj_len - noising_start, len(gripper_index)) > noise).astype(np.float32) * 2 - 1)
    gaussian_noise = np.random.normal(0, noise, actions.shape) if not gripper_only else np.zeros_like(actions)
    actions[noising_start:] += gaussian_noise[noising_start:]
    action_low, action_high = env.action_spec
    actions = np.clip(actions, action_low, action_high)
    actions = actions[noising_start:]
    states = states[noising_start:]
    return states, actions

def split_trajectory(args, states: np.ndarray, actions: np.ndarray):
    traj_len = args.traj_len
    if traj_len is None or traj_len == -1:
        return [(states, actions)]
    splits = []
    start = 0
    while start < states.shape[0]:
        end = min(start + traj_len, states.shape[0])
        if (args.eval and (end - start) < traj_len) or (not args.eval and (end - start) < 40):
            break ## only retain full traj_len segments for eval
        splits.append((states[start:end], actions[start:end]))
        start += random.randint(20, 40) if not args.eval else (traj_len // 2) # 40 steps overlap, 8 frames
    return splits

def playback_trajectory_with_env(
    env, 
    empty_env,
    initial_state, 
    states, 
    actions=None, 
    render=False, 
    video_writers=None, 
    pose_video_writers=None,
    video_skip=5, 
    action_chunk=None,
    camera_names=None,
    first=False,
    res=128,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = (video_writers is not None)
    video_count = 0
    assert not (render and write_video)
    if action_chunk is None:
        action_chunk = video_skip

    empty_env.copy_env_model(env)
    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    env.sim.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    empty_env.copy_robot_state(env)

    # update camera_names for available cameras
    camera_names = list(set(env.sim.model.camera_names).intersection(empty_env.sim.model.camera_names).intersection(set(camera_names)).intersection(set(video_writers.keys())))
    if len(camera_names) == 0:
        return
    
    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in range(traj_len):
        if action_playback:
            obs, reward, done, info = env.step(actions[i])
            empty_env.step(actions[i])
            # if i < traj_len - 1:
            #     # check whether the actions deterministically lead to the same recorded states
            #     state_playback = env.get_state()["states"]
            #     if not np.all(np.equal(states[i + 1], state_playback)):
            #         err = np.linalg.norm(states[i + 1] - state_playback)
            #         print("warning: playback diverged by {} at step {}".format(err, i))
        else:
            env.sim.set_state_from_flattened(states[i])
            empty_env.copy_robot_state(env)

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                for cam_name in camera_names:
                    video_img = obs[cam_name + "_image"][::-1]
                    video_writers[cam_name].append_data(video_img.copy())
                    if 'robot' not in cam_name:
                        cam_transform = empty_env.get_camera_transform(camera_name=cam_name, camera_height=res, camera_width=res)
                        pose_image = empty_env.plot_pose(cam_transform, height=res, width=res)
                        pose_video_writers[cam_name].append_data(pose_image.copy())
            video_count += 1
            if video_count % action_chunk == 0:
                empty_env.copy_robot_state(env)

        if first:
            break


def playback_dataset(args):

    f = h5py.File(args.dataset, "r")

    # some arg checking
    if args.video_path is None:
        # args.video_path = os.path.dirname(args.dataset)
        # Find the relative path of args.dataset to ./dataset
        video_dir = args.dataset_dir + f"_std_{args.noise}" + f"_{args.res}" + f"_chunk{args.action_chunk}" + (f"_gripper" if args.gripper_only else "") + (f"_len{args.traj_len}" if args.traj_len is not None else "") \
            + (f"_eval" if args.eval else "")
        rel_dataset_path = os.path.relpath(args.dataset, args.dataset_dir)
        rel_dataset_path = os.path.splitext(rel_dataset_path)[0]
        print(f"Relative dataset path: {rel_dataset_path}")
        args.video_path = os.path.join(video_dir, rel_dataset_path)
        os.makedirs(args.video_path, exist_ok=True)
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both


    env_name = f["data"].attrs["env_name"]

    env_args = f["data"].attrs["env_args"]
    env_kwargs = json.loads(env_args)["env_kwargs"]

    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_info["domain_name"]
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info["language_instruction"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    bddl_file_name = f["data"].attrs["bddl_file_name"]
    libero_utils.update_env_kwargs(
            env_kwargs,
            bddl_file_name=bddl_file_name,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            camera_depths=False,
            camera_names=args.render_image_names,
            reward_shaping=True,
            control_freq=20,
            camera_heights=args.res,
            camera_widths=args.res,
            camera_segmentations=None,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_name,
        camera_names=args.render_image_names,
        camera_heights=args.res,
        camera_widths=args.res,
        ignore_done=True,
        hard_reset=False,
    ).env
    
    empty_env_kwargs = env_kwargs.copy()
    empty_env_kwargs['robots'] = [type(robot.robot_model).__name__ for robot in env.robots]
    empty_env_kwargs['hard_reset'] = False
    empty_env_kwargs['use_camera_obs'] = False
    empty_env_kwargs['has_renderer'] = False
    empty_env_kwargs['has_offscreen_renderer'] = False
    empty_env = SingleArmEmptyEnv(**empty_env_kwargs)


    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    
    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[:args.n]
    

    for ind in range(len(demos)):
        ep = demos[ind]
        video_dir = os.path.join(args.video_path, ep)
        os.makedirs(video_dir, exist_ok=True)

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        # supply actions if using open-loop action playback
        # actions = None
        # if args.use_actions:
        # always use actions
        actions = f["data/{}/actions".format(ep)][()]
        # Dump states and actions for this episode
        np.save(os.path.join(video_dir, "states.npy"), states)
        print("ep len:", states.shape[0])
        if actions is not None:
            np.save(os.path.join(video_dir, "actions.npy"), actions)
        with open(os.path.join(video_dir, "args.json"), "w") as f_args:
            json.dump(vars(args), f_args, indent=4)

        for i, (seg_states, seg_actions) in enumerate(split_trajectory(args, states, actions)):
            # Use the VideoWriterManager to encapsulate writer lifecycle and atomic moves
            with VideoWriterManager(video_dir, args.render_image_names, i, res=args.res, fps=20) as vwm:
                video_writers = vwm.video_writers
                pose_video_writers = vwm.pose_video_writers

                camera_names = list(set(env.sim.model.camera_names).intersection(empty_env.sim.model.camera_names).intersection(set(args.render_image_names)).intersection(set(video_writers.keys())))
                if len(camera_names) == 0:
                    # context exit will finalize/cleanup writers
                    continue

                noised_states, noised_actions = noise_fn(seg_states, seg_actions, args.noise, env, gripper_only=args.gripper_only, min_len=(80 if args.traj_len is None else None))

                print(f"Playing back seg{i} of episode: {ep} of env {rel_dataset_path} with length {len(noised_states)}", flush=True)

                initial_state = noised_states[0]
                
                playback_trajectory_with_env(
                    env=env, 
                    empty_env=empty_env,
                    initial_state=initial_state, 
                    states=noised_states, actions=noised_actions, 
                    render=args.render, 
                    video_writers=video_writers, 
                    pose_video_writers=pose_video_writers,
                    video_skip=args.video_skip,
                    camera_names=camera_names,
                    first=args.first,
                    res=args.res,
                    action_chunk=args.action_chunk,
                )
                # context manager __exit__ will finalize and cleanup
    f.close()
    del env
    del empty_env
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=['agentview', 'frontview', 'sideview', 'birdview'],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # depth observations to use for writing to video
    parser.add_argument(
        "--render_depth_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) depth observation(s) to use for rendering to video"
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="(optional) noise level to add to actions"
    )
    parser.add_argument(
        "--res",
        type=int,
        default=128,
        help="The resolution of created videos"
    )
    parser.add_argument(
        "--action-chunk",
        type=int,
        default=None,
        help="(optional) number of steps between copying robot state from env to empty_env during"
    )
    parser.add_argument(
        "--gripper-only",
        action='store_true',
        help="if true, only add noise to gripper actions"
    )
    parser.add_argument(
        "--traj-len",
        type=int,
        default=None,
        help="(optional) if provided, only playback random traj_len steps of each trajectory"
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        help="if true, generate eval videos, maximizing length"
    )

    args = parser.parse_args()
    args.dataset_dir = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/DiffRL/libero_dataset/LIBERO/libero/datasets/libero_90"
    if args.action_chunk is None:
        args.action_chunk = args.video_skip
    if not os.path.exists(args.dataset):
        import multiprocessing as mp
        tasks = []
        for file in os.listdir(args.dataset_dir):
            if file.endswith(".hdf5") and (args.dataset in file.lower() or args.dataset == "all"):
                from copy import deepcopy
                cur_args = deepcopy(args)
                cur_args.dataset = os.path.join(args.dataset_dir, file)
                cur_args.video_path = None
                tasks.append(cur_args)

        processes = []
        for task_args in tasks:
            p = mp.Process(target=playback_dataset, args=(task_args,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        playback_dataset(args)
