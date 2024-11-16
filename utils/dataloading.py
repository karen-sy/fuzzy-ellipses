import glob
import json
import os
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm
import numpy as np
import ray.data
import trimesh
from PIL import Image
import utils
from scipy.spatial.transform import Rotation as R
import cv2

def ray_init(num_cpus: int | None = None):
    """Initialize or connect to existing Ray cluster and set commonly used
    environment variables.

    Args:
        num_cpus (Optional[int]): Number of CPUs to assign for parallel workers.
        By default, this is set based on the number of available CPUs.
    """
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

YCB_MODEL_NAMES = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]


def remove_zero_pad(img_id):
    for i, ch in enumerate(img_id):
        if ch != "0":
            return img_id[i:]


def get_ycbv_num_images(ycb_dir, scene_id):
    scene_id = str(scene_id).rjust(6, "0")
    scene_data_dir = os.path.join(
        ycb_dir, scene_id
    )  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json

    scene_rgb_images_dir = os.path.join(scene_data_dir, "rgb")
    sorted(glob.glob(scene_rgb_images_dir + "/*.png"))[-1]
    return int(
        os.path.basename(sorted(glob.glob(scene_rgb_images_dir + "/*.png"))[-1]).split(
            "."
        )[0]
    )


def get_camera_K_pose_from_dict(image_cam_data):
    cam_K = np.array(image_cam_data["cam_K"]).reshape(3, 3)
    cam_R_w2c = np.array(image_cam_data["cam_R_w2c"]).reshape(3, 3)
    cam_t_w2c = np.array(image_cam_data["cam_t_w2c"]).reshape(3, 1)
    cam_pose_w2c = np.vstack(
        [np.hstack([cam_R_w2c, cam_t_w2c]), np.array([0, 0, 0, 1])]
    )
    cam_pose = np.linalg.inv(cam_pose_w2c)
    cam_pose[:3, 3] /= 1000.0
    camera_intrinsics = np.array(
        [
            cam_K[0, 0],
            cam_K[1, 1],
            cam_K[0, 2],
            cam_K[1, 2],
        ]
    )
    return camera_intrinsics, utils.pose_matrix_to_transform(cam_pose)


def get_pose_object_info_from_dict(d):
    model_R = np.array(d["cam_R_m2c"]).reshape(3, 3)
    model_t = np.array(d["cam_t_m2c"]) / 1000.0
    obj_id = d["obj_id"] - 1
    return obj_id, np.concatenate([model_t, R.from_matrix(model_R).as_quat()])


def get_ycb_mesh(ycb_dir, id):
    mesh = trimesh.load(
        os.path.join(ycb_dir, f'../models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    )
    mesh.vertices *= 0.001
    return mesh

def get_ycb_mesh_eval(ycb_dir, id):
    mesh = trimesh.load(
        os.path.join(ycb_dir, f'../models_eval/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    )
    mesh.vertices *= 0.001
    return mesh


def fetch_ycbv_data(ycb_dir, scene_id, images_indices, fields=[]):
    scene_id = str(scene_id).rjust(6, "0")

    scene_data_dir = os.path.join(
        ycb_dir, scene_id
    )  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json

    scene_rgb_images_dir = os.path.join(scene_data_dir, "rgb")
    scene_depth_images_dir = os.path.join(scene_data_dir, "depth")
    mask_visib_dir = os.path.join(scene_data_dir, "mask_visib")

    with open(os.path.join(scene_data_dir, "scene_camera.json")) as scene_cam_data_json:
        scene_cam_data = json.load(scene_cam_data_json)

    with open(os.path.join(scene_data_dir, "scene_gt.json")) as scene_imgs_gt_data_json:
        scene_imgs_gt_data = json.load(scene_imgs_gt_data_json)

    all_data = []
    for image_index in images_indices:
        img_id = str(image_index).rjust(6, "0")
        data = {"t": image_index, "img_id": img_id}

        image_cam_data = scene_cam_data[remove_zero_pad(img_id)]
        cam_depth_scale = image_cam_data["depth_scale"]

        image_cam_data_np = {k: np.array(v) for k, v in image_cam_data.items()}

        camera_intrinsics, cam_pose = get_camera_K_pose_from_dict(image_cam_data_np)
        data["camera_pose"] = cam_pose
        data["camera_intrinsics"] = camera_intrinsics
        data["camera_depth_scale"] = cam_depth_scale

        data["rgb_filename"] = os.path.join(scene_rgb_images_dir, f"{img_id}.png")
        data["depth_filename"] = os.path.join(scene_depth_images_dir, f"{img_id}.png")

        num_objects = len(scene_imgs_gt_data[remove_zero_pad(img_id)])
        mask_visib_image_paths = [
            os.path.join(mask_visib_dir, f"{img_id}_{obj_idx:06}.png")
            for obj_idx in range(num_objects)
        ]
        data["masks"] = mask_visib_image_paths

        # get GT object model ID+poses
        objects_gt_data = scene_imgs_gt_data[remove_zero_pad(img_id)]
        object_types = []
        object_poses = []
        for d in objects_gt_data:
            obj_id, obj_pose = get_pose_object_info_from_dict(d)
            object_types.append(obj_id)
            object_poses.append(obj_pose)

        data["object_types"] = np.array(object_types)
        data["object_poses"] = np.array(object_poses)
        all_data.append(data)
    return all_data


def preprocess_ycbv_data(data: dict[str, Any]) -> dict[str, Any]:
    rgb = np.array(Image.open(data["rgb_filename"]), dtype=np.uint8)
    depth = (np.array(Image.open(data["depth_filename"])) * data["camera_depth_scale"] / 1000.0)[
        ...
    ]
    data["rgb"] = rgb
    data["lab"] = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    data["depth"] = depth
    data["masks"] = np.stack([np.array(Image.open(mask)) > 0 for mask in data["masks"]])
    return data

def get_ycbv_data(
    ycb_dir: Path | str, scene_id: int, images_indices: Iterable[int], fields=[]
):
    all_data = []
    scene_info = fetch_ycbv_data(ycb_dir, scene_id, images_indices, fields)
    for data in tqdm(scene_info):
        data = preprocess_ycbv_data(data)
        all_data.append(data)
    return all_data


def get_ycbv_data_ray(
    ycb_dir: Path | str, scene_id: int, images_indices: Iterable[int], fields=[]
) -> ray.data.Dataset:
    """Construct a [Ray Dataset](https://docs.ray.io/en/latest/data/api/dataset.html)
    for the given YCBV test scene, which lazily load and process the images only
    when the corresponding frame is accessed."""
    # initizlie ray (if we haven't already) and set the correct number of CPUs
    scene_info = fetch_ycbv_data(ycb_dir, scene_id, images_indices, fields)
    scene_data = ray.data.from_items(scene_info).map(preprocess_ycbv_data)
    return scene_data