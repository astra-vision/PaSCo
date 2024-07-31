# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from waffleiron import Segmenter
from torch.utils.data import DataLoader
from datasets import NuScenesSemSeg, Collate


if __name__ == "__main__":

    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint")
    parser.add_argument("--path_dataset", type=str, help="Path to SemanticKITTI dataset")
    parser.add_argument("--result_folder", type=str, help="Path to where result folder")
    parser.add_argument("--num_votes", type=int, default=1, help="Number of test time augmentations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--phase", required=True, help="val or test")
    args = parser.parse_args()
    assert args.num_votes % args.batch_size == 0
    args.result_folder = os.path.join(args.result_folder, "lidarseg", args.phase)
    os.makedirs(args.result_folder, exist_ok=True)

    # --- Load config file
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- Dataloader
    dataset = NuScenesSemSeg(
        rootdir=args.path_dataset,
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors=config["embedding"]["neighbors"],
        dim_proj=config["waffleiron"]["dim_proj"],
        grids_shape=config["waffleiron"]["grids_size"],
        fov_xyz=config["waffleiron"]["fov_xyz"],
        phase=args.phase,
        tta=(args.num_votes > 1),
    )
    if args.num_votes > 1:
        new_list = []
        for f in dataset.list_frames:
            for v in range(args.num_votes):
                new_list.append(f)
        dataset.list_frames = new_list
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
    )
    args.num_votes = args.num_votes // args.batch_size

    # --- Build network
    net = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
    )
    net = net.cuda()

    # --- Load weights
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    try:
        net.load_state_dict(ckpt["net"])
    except:
        # If model was trained using DataParallel or DistributedDataParallel
        state_dict = {}
        for key in ckpt["net"].keys():
            state_dict[key[len("module."):]] = ckpt["net"][key]
        net.load_state_dict(state_dict)
    net = net.eval()

    # --- Evaluation
    id_vote = 0
    for it, batch in enumerate(tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")):

        # Reset vote
        if id_vote == 0:
            vote = None

        # Network inputs
        feat = batch["feat"].cuda(non_blocking=True)
        labels = batch["labels_orig"].cuda(non_blocking=True)
        batch["upsample"] = [
            up.cuda(non_blocking=True) for up in batch["upsample"]
        ]
        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # Get prediction
        with torch.autocast("cuda", enabled=True):
            with torch.inference_mode():
                # Get prediction
                out = net(*net_inputs)
                for b in range(out.shape[0]):
                    temp = out[b, :, batch["upsample"][b]].T
                    if vote is None:
                        vote = torch.softmax(temp, dim=1)
                    else:
                        vote += torch.softmax(temp, dim=1)
        id_vote += 1

        # Save prediction
        if id_vote == args.num_votes:
            # Get label
            pred_label = vote.max(1)[1] + 1 # Shift by 1 because of ignore_label at index 0
            # Save result
            bin_file_path = os.path.join(args.result_folder, batch["filename"][0] + "_lidarseg.bin")
            np.array(pred_label.cpu().numpy()).astype(np.uint8).tofile(bin_file_path)
            # Reset count of votes
            id_vote = 0