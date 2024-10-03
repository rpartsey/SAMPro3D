"""
Script for the stage of 3D Prompt Proposal in the paper

Author: Mutian Xu (mutianxu@link.cuhk.edu.cn) and Xingyilang Yin
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
import os
import cv2
import argparse
import torch
import numpy as np
import open3d as o3d
import pointops
from utils.main_utils import *
from utils.sam_utils import *
from segment_anything import sam_model_registry, SamPredictor
from tqdm import trange


def create_output_folders(args):
    # Create folder to save SAM outputs:
    create_folder(args.sam_output_path)
    # Create subfolder for saving different output types:
    create_folder(f"{args.sam_output_path}/points_npy")
    create_folder(f"{args.sam_output_path}/iou_preds_npy")
    create_folder(f"{args.sam_output_path}/masks_npy")
    create_folder(f"{args.sam_output_path}/corre_3d_ins_npy")


def prompt_init(xyz, rgb, voxel_size, device):
    # Here we only use voxelization to decide the number of fps-sampled points, \
    # since voxel_size is more controllable. We use fps later for prompt initialization
    idx_sort, num_pt = voxelize(xyz, voxel_size, mode=1)
    print("the number of initial 3D prompts:", len(num_pt))
    xyz = torch.from_numpy(xyz).cuda().contiguous()
    o, n_o = len(xyz), len(num_pt)
    o, n_o = torch.cuda.IntTensor([o]), torch.cuda.IntTensor([n_o])
    idx = pointops.farthest_point_sampling(xyz, o, n_o)
    fps_points = xyz[idx.long(), :]
    fps_points = torch.from_numpy(fps_points.cpu().numpy()).to(device=device)
    rgb = rgb / 256.
    rgb = torch.from_numpy(rgb).cuda().contiguous()
    fps_colors = rgb[idx.long(), :]
    
    return fps_points, fps_colors
    

def save_init_prompt(xyz, rgb, args):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    point_cloud.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
    prompt_ply_file = f"{args.prompt_path}/{args.scene_name}.ply"
    o3d.io.write_point_cloud(prompt_ply_file, point_cloud)
    
    
def process_batch(
    predictor,
    points: torch.Tensor,
    point_labels: torch.Tensor,
    ins_idxs: torch.Tensor,
    im_size: Tuple[int, ...],
) -> MaskData:
    transformed_points = predictor.transform.apply_coords_torch(points, im_size)
    in_points = torch.as_tensor(transformed_points, device=predictor.device)
    in_labels = torch.from_numpy(point_labels).to(device=predictor.device)  
    # torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

    masks, iou_preds, _ = predictor.predict_torch(
        in_points[None, :, :],
        in_labels[None, :],
        multimask_output=False,
        return_logits=True,
    )
    
    # Serialize predictions and store in MaskData  
    data_original = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=points, 
        corre_3d_ins=ins_idxs 
    )

    return data_original
    

def sam_seg(predictor, frame_id_init, frame_id_end, init_prompt, point_labels, args):
    for frame_id in trange(frame_id_init, frame_id_end):
        # Load the intrinsic
        depth_intrinsic = torch.tensor(
            np.loadtxt(f"{args.data_path}/intrinsics.txt"),
            dtype=torch.float64
        ).to(device=predictor.device)

        # Load the depth, and pose
        depth = torch.from_numpy(
            cv2.imread(f"{args.data_path}/{args.scene_name}/depth/{frame_id}.png", -1).astype(np.float64)
        ).to(device=predictor.device) 

        pose = torch.tensor(
            np.loadtxt(f"{args.data_path}/{args.scene_name}/pose/{frame_id}.txt"),
            dtype=torch.float64
        ).to(device=predictor.device)
        
        if str(pose[0, 0].item()) == '-inf': # skip frame with '-inf' pose
            print(f'skip frame {frame_id}')
            continue

        # 3D-2D projection
        input_point_pos, corre_ins_idx = transform_pt_depth_scannet_torch(
            points=init_prompt, 
            depth_intrinsic=depth_intrinsic, 
            depth=depth, 
            pose=pose, 
            device=predictor.device
        )  # [valid, 2], [valid]
        if corre_ins_idx.shape[0] == 0 or point_labels[corre_ins_idx.cpu().numpy()].sum() == 0:
            print(f'skip frame {frame_id}')
            continue
        # if input_point_pos.shape[0] == 0 or input_point_pos.shape[1] == 0:
        #     print(f'skip frame {frame_id}')
        #     continue
        
        
        if os.path.exists(f"{args.data_path}/{args.scene_name}/features"):
            # if features are pre-computed, load them
            features = np.load(os.path.join(args.data_path, args.scene_name, 'features', str(frame_id) + '.npy'))
            features = torch.from_numpy(features).to(device=predictor.device)
            predictor.features = features
            predictor.original_size = (480, 640)
            predictor.input_size = (768, 1024)
            predictor.is_image_set = True
        else:
            # otherwise, compute features
            image = cv2.imread(os.path.join(args.data_path, args.scene_name, 'color', str(frame_id) + '.jpg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

        # predictor.set_image(image)
        # SAM segmetaion on image
        # data_original = MaskData()
        # for (points, ins_idxs) in batch_iterator(64, input_point_pos, corre_ins_idx):
            
        #     ins_idxs = torch.arange(1).to(device) # override the ins_idxs to 0 for all points
        #     batch_data_original = process_batch(predictor, points, ins_idxs, image_size)
        #     data_original.cat(batch_data_original)
        #     del batch_data_original

        ins_idxs = torch.arange(1).to(device) # override the ins_idxs to 0 for all points
        data_original = process_batch(predictor, input_point_pos, (point_labels[corre_ins_idx.cpu().numpy()] == 1).astype(np.int32), ins_idxs, predictor.original_size)
        
        predictor.reset_image()
        data_original.to_numpy()

        save_file_name = str(frame_id) + ".npy"
        np.save(f"{args.sam_output_path}/points_npy/{save_file_name}", data_original["points"])
        np.save(f"{args.sam_output_path}/masks_npy/{save_file_name}", data_original["masks"])  
        np.save(f"{args.sam_output_path}/iou_preds_npy/{save_file_name}", data_original["iou_preds"])  
        np.save(f"{args.sam_output_path}/corre_3d_ins_npy/{save_file_name}", data_original["corre_3d_ins"])


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate 3d prompt proposal on ScanNet.")
    # for voxelization to decide the number of fps-sampled points:
    # parser.add_argument('--voxel_size', default=0.2, type=float, help='Size of voxels.')
    # path arguments:
    parser.add_argument('--data_path', default="dataset/scannet", type=str, help='Path to the dataset containing ScanNet 2d frames and 3d .ply files.')
    parser.add_argument('--scene_name', default="scene0030_00", type=str, help='The scene names in ScanNet.')
    parser.add_argument('--prompt_name', required=True, type=str, help='The name of the prompt.')
    # parser.add_argument('--prompt_path', default="init_prompt", type=str, help='Path to the save the sampled 3D initial prompts.')
    parser.add_argument('--experiments_path', default="experiments", type=str, help='Path to the experiments folder.')
    # sam arguments:
    parser.add_argument('--model_type', default="vit_h", type=str, help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument('--sam_checkpoint', default="sam_vit_h_4b8939.pth", type=str, help='The path to the SAM checkpoint to use for mask generation.')
    parser.add_argument("--device", default="cuda:0", type=str, help="The device to run generation on.")
    args = parser.parse_args()

    # set the output path
    args.sam_output_path = f"{args.experiments_path}/{args.scene_name}/{args.prompt_name}/sam_output"

    return args
    

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    # Initialize SAM:
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](
        checkpoint=args.sam_checkpoint
    ).to(device=device)
    predictor = SamPredictor(sam)
    
    # Load all 3D points of the input scene: 
    # scene_xyz, scene_rgb = load_ply(f"{args.data_path}/{args.scene_name}/{args.scene_name}_vh_clean_2.ply")

    # 3D prompt initialization:
    # prompt_xyz, prompt_rgb = prompt_init(scene_xyz, scene_rgb, args.voxel_size, device)
    # save the initial 3D prompts for later use:
    # create_folder(args.prompt_path)
    # save_init_prompt(prompt_xyz, prompt_rgb, args)

    prompt_xyz, prompt_rgb = load_ply(f"{args.data_path}/{args.scene_name}/{args.scene_name}_{args.prompt_name}.ply")
    prompt_xyz = torch.from_numpy(prompt_xyz).to(device=device)
    print(f"prompt_xyz.shape {prompt_xyz.shape}")

    # SAM segmentation on image frames:
    # create folder to save diffrent SAM output types for later use (note that this is the only stage to perform SAM):
    create_output_folders(args)  # we use npy files to save different output types for faster i/o and clear split
    
    # perform SAM on each 2D RGB frame:
    frame_id_init = 0
    frame_id_end = len(os.listdir(os.path.join(args.data_path, args.scene_name, 'depth'))) 
    # You can define frame_id_init and frame_id_end by yourself for segmenting partial point clouds from limited frames. Sometimes partial result is better!
    print("Start performing SAM segmentations on {} 2D frames...".format(frame_id_end))
    sam_seg(
        predictor=predictor, 
        frame_id_init=frame_id_init, 
        frame_id_end=frame_id_end, 
        init_prompt=prompt_xyz, 
        point_labels=np.isclose(np.linalg.norm(prompt_rgb - prompt_rgb[-1], axis=1), 0),
        args=args
    )
    print("Finished performing SAM segmentations!")
