import os
import h5py
import json
import random
import numpy as np
import tqdm
from argparse import ArgumentParser
from libero.libero.envs.env_wrapper import ControlEnv

def get_goal_object_attrs(env):
    goal_states = env.env.parsed_problem['goal_state']

    object_attrs_dict = dict()

    for goal_state in goal_states:
        if len(goal_state) == 2:
            predicate_fn_name = goal_state[0]
            object_name = goal_state[1]
            object_attrs_dict[object_name] = predicate_fn_name
        elif len(goal_state) == 3:
            predicate_fn_name = goal_state[0]
            object_name_1 = goal_state[1]
            object_name_2 = goal_state[2]
            object_attrs_dict[object_name_1] = "grasp"
    
    return object_attrs_dict

# object_attrs_dict = get_goal_object_attrs(env)

def get_object_instructions(object_attrs_dict):
    object_instructions_dict = {}
    for object_name, attr in object_attrs_dict.items():
        if "region" in object_name:
            words = object_name.split("_")
            assert words[-1] == "region"
            region_name = words[-2]
            actual_name = words[:-3][-1]
            assert words[-3] in ['1', '2', '3', '4', '5']
            instruction = attr + " the " + region_name + " region of the " + actual_name
        else:
            words = object_name.split("_")
            assert words[-1] in ['1', '2', '3', '4', '5']
            actual_name = words[:-1][-1]
            if attr == "turnon":
                attr = "turn on"
            instruction = attr + " the " + actual_name
        object_instructions_dict[object_name] = instruction
        # print(f"- {instruction}.")

    return object_instructions_dict

# get_object_instructions(object_attrs_dict)


def record_object_states(env, states, object_attrs_dict=None, object_instructions_dict=None):
    from libero.libero.envs.predicates import eval_predicate_fn
    from collections import defaultdict, OrderedDict
    object_states_dict = defaultdict(list)
    if object_attrs_dict is None:
        object_attrs_dict = get_goal_object_attrs(env)
    if object_instructions_dict is None:
        object_instructions_dict = get_object_instructions(object_attrs_dict)
    for t in range(states.shape[0]):
        obs = env.set_init_state(states[t])
        for object_name in object_attrs_dict.keys():
            attr = object_attrs_dict[object_name]
            object_state = env.env.object_states_dict[object_name]
            if attr == "grasp":
                grasp = env.env._check_grasp(env.env.robots[0].gripper, env.env.objects_dict[object_name])
                object_states_dict[object_name].append(grasp)
                # print(f'Object {object_name} position: {pos}')
            else:
                object_states_dict[object_name].append(eval_predicate_fn(attr, object_state))
    
    object_segments_dict = OrderedDict()

    for object_name, object_state_list in object_states_dict.items():
        attr = object_attrs_dict[object_name]
        if attr == "grasp":
            object_states_dict[object_name] = np.array(object_state_list, dtype=bool)
            change_idxs = np.where(object_states_dict[object_name][1:] != object_states_dict[object_name][:-1])[0] + 1
            if len(change_idxs) == 0:
                object_attrs_dict.pop(object_name)
                import warnings
                warnings.warn(f"Object {object_name} does not change state in the demo.", RuntimeWarning)
                continue
            start_idx = change_idxs[0]
            end_idx = change_idxs[1] if len(change_idxs) > 1 else states.shape[0]-1
            object_segments_dict[object_name] = (start_idx, end_idx)
            assert all(object_states_dict[object_name][start_idx:end_idx])
        else:
            object_states_dict[object_name] = np.array(object_state_list, dtype=bool)
            if all(object_states_dict[object_name]):
                object_attrs_dict.pop(object_name)
                continue
            change_idxs = np.where(object_states_dict[object_name][1:] != object_states_dict[object_name][:-1])[0] + 1
            if len(change_idxs) == 0:
                raise ValueError(f"Object {object_name} does not change state in the demo.")
            start_idx = change_idxs[0]
            object_segments_dict[object_name] = (start_idx, )

    object_segments_dict = OrderedDict(sorted(object_segments_dict.items(), key=lambda x: x[1][0]))

    cur_step = 0
    followup_length = 16
    
    resulting_segments_dict = {}
    for i, (object, segment) in enumerate(object_segments_dict.items()):
        if len(segment) == 2:
            segment_start = cur_step
            segment_end = min(segment[0] + followup_length, segment[1])
            cur_step = max(segment_end, segment[1] + followup_length)
        else:
            segment_start = cur_step
            segment_end = min(segment[0] + followup_length, states.shape[0]-1)
            cur_step = segment_end
        resulting_segments_dict[object] = (segment_start, segment_end)

    results_dict = {}
    for object_name in object_attrs_dict.keys():
        results_dict[object_name] = {
            'instruction': object_instructions_dict[object_name],
            'segment': resulting_segments_dict[object_name]
        }

    return results_dict

def visualize(demo, segment=None, file_name=None):
    file_path = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/scratch_dir"
    save_path = os.path.join(file_path, "segement_vis", file_name.replace(" ", "_"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images = demo['obs']['agentview_rgb']
    if segment is None:
        segment = (0, len(images))
    images = np.array(images)[segment[0]:segment[1], ::-1]
    import imageio
    imageio.mimwrite(save_path + '.mp4', images, fps=10)


def copy_slice(src_group, dst_group, slc):
    """
    Recursively copy groups & datasets from `src_group` into `dst_group`.
    Applies the slice `slc` to every dataset.
    """
    for key in src_group.keys():
        item = src_group[key]

        if isinstance(item, h5py.Dataset):  # dataset → slice and write
            data = item[slc]
            dst_group.create_dataset(key, data=data, compression="gzip")

        elif isinstance(item, h5py.Group):  # group → recurse
            new_grp = dst_group.create_group(key)
            copy_slice(item, new_grp, slc)


def main(args):
    
    with h5py.File(args.dataset, "r") as f:
        env_args = f["data"].attrs["env_args"]
        env_kwargs = json.loads(env_args)["env_kwargs"]
        problem_info = json.loads(f["data"].attrs["problem_info"])
        language_instruction = problem_info["language_instruction"]
        bddl_file_name = f["data"].attrs["bddl_file_name"]
        print("Language Instruction:", language_instruction)
    

    env = ControlEnv(
        bddl_file_name=bddl_file_name,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )

    object_attrs_dict = get_goal_object_attrs(env)
    object_instructions_dict = get_object_instructions(object_attrs_dict)
    print("Object Instructions:", object_instructions_dict)

    with h5py.File(args.dataset, "a") as f:
        demos = list(f['data'].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        dst_grp = f.require_group("segment_data")
        
        # maybe reduce the number of demonstrations to playback
        if args.n is not None:
            random.shuffle(demos)
            demos = demos[:args.n]

        for ep in tqdm.tqdm(demos):
            states = f[f"data/{ep}/states"][()]
            results_dict = record_object_states(env, states, object_attrs_dict, object_instructions_dict)
            for i, (object_name, result) in enumerate(results_dict.items()):
                segment = result['segment']
                instruction = result['instruction']
                slc = slice(segment[0], segment[1])
                ep_name = f"{ep}_{i}"
                if ep_name in dst_grp:
                    continue
                ep_grp = dst_grp.create_group(ep_name)
                copy_slice(f[f"data/{ep}"], ep_grp, slc)
                ep_grp.attrs['language_instruction'] = instruction

        if args.vis:
            for ep_name in list(f['segment_data'].keys()):
                instruction = f['segment_data'][ep_name].attrs['language_instruction']
                visualize(f['segment_data'][ep_name], None, ep_name + "_" + instruction)
                


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the dataset hdf5 file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of demonstrations to process. If not set, process all.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Whether to visualize the segments.",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dataset_dir = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/DiffRL/libero_dataset/LIBERO/libero/datasets/libero_90"
    if not os.path.exists(args.dataset):
        import multiprocessing as mp
        tasks = []
        for file in os.listdir(dataset_dir):
            if file.endswith(".hdf5") and (args.dataset in file.lower() or args.dataset == "all"):
                from copy import deepcopy
                cur_args = deepcopy(args)
                cur_args.dataset = os.path.join(dataset_dir, file)
                tasks.append(cur_args)

        processes = []
        for task_args in tasks:
            p = mp.Process(target=main, args=(task_args,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        # for task_args in tasks:
        #     main(task_args)
    else:
        main(args)

                
    