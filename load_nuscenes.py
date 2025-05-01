from nuscenes.nuscenes import NuScenes
import numpy as np
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from nuscenes.utils.data_classes import Quaternion  # Import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points#, Box
from colormap import ade_palette
# from nuscenes.map_expansion.map_api import NuScenesMap


# pkl_path = "/home/ge32buc/new/nuprompt/nuprompt_infos_val.pkl"

# import pickle
# with open(pkl_path,"rb") as f:
#     data = pickle.load(f)

# print(data)
import os

class NuScenesData():
    def __init__(self,dataset_type):
        print(os.getcwd())
        if("home" in os.getcwd()):
            self.dataset_root = '/home/datasets/nuscenes/raw/'
        elif("mnt" in os.getcwd()):
            self.dataset_root = '/mnt/nuscenes/raw/'
        # self.dataset_root = '/mnt/nuscenes/'
        self.nusc = NuScenes(version=dataset_type, dataroot=self.dataset_root, verbose=True)#trainval#mini

    def get_closest_pixel_with_point(self,points,detected_2d_points):
            distances_min = np.linalg.norm(points - detected_2d_points[0], axis=1)
            closest_index_min = np.argmin(distances_min)
            closest_pixel_with_point_min = points[closest_index_min]
            closest_point_min = self.point_cloud.points.T[closest_index_min]

            distances_max = np.linalg.norm(points - detected_2d_points[1], axis=1)
            closest_index_max = np.argmin(distances_max)
            closest_pixel_with_point_max = points[closest_index_max]
            closest_point_max = self.point_cloud.points.T[closest_index_max]

            return (closest_pixel_with_point_min,closest_pixel_with_point_max), (closest_point_min,closest_point_max)

    def transform_lidar_camera(self,lidar_token,cam_token,nusc,dataset_root):
        cam_front_data = nusc.get('sample_data', cam_token,)
        cam_front_image = Image.open(dataset_root + cam_front_data['filename'])

        # Access the LiDAR (LIDAR_TOP) data
        lidar_top_data = nusc.get('sample_data', lidar_token)
        lidar_top_file = dataset_root + lidar_top_data['filename']

        point_cloud = LidarPointCloud.from_file(lidar_top_file)

        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
        point_cloud.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        point_cloud.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
        point_cloud.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        point_cloud.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        point_cloud.translate(-np.array(poserecord['translation']))
        point_cloud.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        point_cloud.translate(-np.array(cs_record['translation']))
        point_cloud.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        ego_pose = nusc.get('ego_pose', cam_front_data["ego_pose_token"])
        # ego_pose_ = nusc.get('ego_pose', lidar_top_data["ego_pose_token"])
        

        projected_points = view_points(point_cloud.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True).T

        return cam_front_image, point_cloud, projected_points, ego_pose

    def get_relative_rotation(self,rotation_ego,rotation_object):
        q = rotation_ego
        rotation1_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])

        w1, x1, y1, z1 = rotation1_conjugate
        w2, x2, y2, z2 = rotation_object
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
        
    def get_annotation_data(self,sample_token,ego_pose,nusc):
        my_annotation_metadata =  nusc.get('sample_annotation', sample_token)
        # translation = np.array(my_annotation_metadata["translation"]) - np.array(ego_pose["translation"])
        # rotation = self.get_relative_rotation(ego_pose["rotation"], my_annotation_metadata["rotation"])

        
        size = my_annotation_metadata["size"]
        rotation = my_annotation_metadata["rotation"]
        translation = my_annotation_metadata["translation"]
        category_name = my_annotation_metadata["category_name"]

        # Create a 3D bounding box for the object (size and rotation in the world frame)
        # box = Box(center=translation, size=size, orientation=rotation)

        # Step 1: Convert the 3D bounding box corners to camera coordinates
        # Get the corners of the bounding box in world coordinates
        # corners_3d = box.corners()  # This returns a 3x8 matrix with the 8 corners of the box
        # print(corners_3d)
        # print("C"+5)

        return translation, rotation, size, category_name

    def get_nuscenes_video(self,next_token,last_token,data,nusc,dataset_root,nusc_map,patch_radius,traffic_light,traffic_sign):

        sample = nusc.get('sample', next_token)
        

        front_image, point_cloud, projected_points, ego_pose = self.transform_lidar_camera(sample['data']['LIDAR_TOP'],
                                                                            sample['data']['CAM_FRONT'],
                                                                            nusc,dataset_root)
        # traffic lights
        ego_translation = ego_pose["translation"]
        box_coords = (
            ego_translation[0] - patch_radius,
            ego_translation[1] - patch_radius,
            ego_translation[0] + patch_radius,
            ego_translation[1] + patch_radius,
            )
        layer_names = ["traffic_light"]
        records_in_patch = nusc_map.get_records_in_patch(box_coords, layer_names, "intersect")
        annotation_data = []
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = nusc_map.get(layer_name, token)
                annotation_data.append([[record["pose"]["tx"],record["pose"]["ty"],record["pose"]["tz"]],
                                        [record["pose"]["rx"],record["pose"]["ry"],record["pose"]["rz"]],
                                        [0,0,0],"traffic_light"])
        for anno_token in sample["anns"]:
            annotation_data.append(self.get_annotation_data(anno_token,ego_pose,nusc))
            # print(annotation_data[-1][-1])
            if("light" in annotation_data[-1][-1]):
                traffic_light = True
            if("sign" in annotation_data[-1][-1]):
                traffic_sign = True

        data.append((front_image,point_cloud,projected_points,ego_pose,annotation_data))

        if(sample["next"] == last_token):
            return data,traffic_light,traffic_sign
        
        return self.get_nuscenes_video(sample["next"],last_token,data,nusc,dataset_root,nusc_map,patch_radius,traffic_light,traffic_sign)

    def get_nuscenes_data(self,scene_index):
        # Initialize the NuScenes object
        
        traffic_light = False
        traffic_sign = False
        try:
            scene = self.nusc.scene[scene_index]
        except IndexError:
            print(f"Error: Index {scene_index} is out of range for nuscenes scenes.")

        # print(scene)
        log_token = scene["log_token"]
        patch_radius = 50
        log_record = self.nusc.get("log", log_token)
        log_location = log_record["location"]
        
        nusc_map = NuScenesMap(dataroot=self.dataset_root, map_name=log_location)
        data = []
        data,traffic_light,traffic_sign = self.get_nuscenes_video(scene["first_sample_token"],scene["last_sample_token"],data,self.nusc,self.dataset_root,nusc_map,patch_radius,traffic_light,traffic_sign)
        
        
        if(traffic_sign == True and traffic_light == True):
            print(scene_index)
            print("C"+5)

        return data, scene["token"]



if __name__ == "__main__":
    # dataset = NuScenesData("v1.0-mini")
    dataset = NuScenesData("v1.0-trainval")
    token = "1d4db80d13f342aba4881b38099bc4b7"
    try:
        print("scene")
        print(dataset.nusc.get("scene", token))
    except Exception as e:
        print(e)
    try:
        print("sample")
        print(dataset.nusc.get('sample', token))
    except Exception as e:
        print(e)

    
    for i in range(122):
        data = dataset.get_nuscenes_data(i)
    # print(data)

