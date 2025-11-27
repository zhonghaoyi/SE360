import torch
import os
import numpy as np
import random
from collections import defaultdict
from utils.pano import Equirectangular, random_sample_camera, horizon_sample_camera, icosahedron_sample_camera, cubemap_sample_camera, eight_pers_sample_camera
import lightning as L
import cv2
from glob import glob
from einops import rearrange
from abc import abstractmethod
from PIL import Image
from external.Perspective_and_Equirectangular import mp2e
import json

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


class PanoDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.random_clip = 0
        self.data_dir = config['data_dir']
        self.inpaint_data_dir = config['inpaint_data_dir']
        self.predict_data_dir = config['predict_data_dir']
        self.s3d_data_dir = config['s3d_data_dir']
        self.s3d_inpaint_data_dir = config['s3d_inpaint_data_dir']
        self.result_dir = config.get('result_dir', None)
        self.config = config
        self.use_fixed_pers_prompt = config['use_fixed_pers_prompt']
        self.use_cubemap_prompt = config['use_cubemap_prompt']
        self.only_pano = config['only_pano']
        self.use_ref = config['use_ref']

        self.data = self.load_split(mode)

        if mode == 'predict':
            self.data = sum([[d.copy() for i in range(self.config['repeat_predict'])] for d in self.data], [])
            if self.config['repeat_predict'] > 1:
                for i, d in enumerate(self.data):
                    d['repeat_id'] = i % self.config['repeat_predict']

        if not self.config['gt_as_result'] and self.result_dir is not None:
            results = self.scan_results(self.result_dir)
            assert results, f"No results found in {self.result_dir}, forgot to set environment variable WANDB_RUN_ID?"
            
            results_set = set(results)
            new_data = [d for d in self.data if (d['scene_id'], d['view_id']) in results_set]
            if len(new_data) != len(self.data):
                print(f"WARNING: {len(self.data)-len(new_data)} views are missing in results folder {self.result_dir} for {self.mode} set.")
                self.data = list(new_data)
                self.data.sort()

    @abstractmethod  #sub class must implement this method
    def load_split(self):
        pass

    @abstractmethod
    def scan_results(self):
        pass

    def __len__(self):
        return len(self.data)
    
    def load_prompt_without_img_add(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        return prompt
    def load_prompt_without_img_remove(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        prompt = prompt.replace('add', 'remove', 1)
        return prompt
    def load_prompt_add(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        prompt = prompt.strip()
        if prompt.endswith('.'):
            prompt = prompt[:-1]  # Remove trailing period
        prompt = prompt + ' in this image: <img><|image_1|></img>'
        return prompt
    def load_prompt_with_ref(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        prompt = prompt.strip()
        if prompt.endswith('.'):
            prompt = prompt[:-1]  # Remove trailing period
        prompt = 'According to the reference image <img><|image_2|></img>, ' + prompt + ' in this image: <img><|image_1|></img>'
        return prompt
    def load_prompt_remove(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        prompt = prompt.strip()
        if prompt.endswith('.'):
            prompt = prompt[:-1]  # Remove trailing period
        prompt = prompt + ' in this image: <img><|image_1|></img>'
        prompt = prompt.replace('add', 'remove', 1)
        return prompt
    def load_test_pers_prompt(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        prompt = prompt.strip()
        return prompt
    
    @abstractmethod
    def get_data(self, idx):
        pass

    def __getitem__(self, idx):
        data = self.get_data(idx)
        self.random_clip = random.random()

        

        # generate camera poses
        if self.config['cam_sampler'] == 'horizon':
            theta, phi = horizon_sample_camera(8)
            if self.mode == 'train':
                cam_rot = random.random() * 360
                theta = (theta + cam_rot) % 360
                if 'prompt' in data:
                    shift_idx = round(cam_rot / 45)
                    data['prompt'] = data['prompt'][shift_idx:] + data['prompt'][:shift_idx]
        elif self.config['cam_sampler'] == 'icosahedron':
            if self.mode == 'train':
                if self.use_fixed_pers_prompt==True and self.use_cubemap_prompt==False:
                    theta, phi = icosahedron_sample_camera()#random_sample_camera(20) #20
                elif self.use_fixed_pers_prompt==True and self.use_cubemap_prompt==True:
                    theta, phi = cubemap_sample_camera()
                else:
                    theta, phi = random_sample_camera(20)
            elif self.mode == 'val' and self.use_cubemap_prompt==True:
                theta, phi = cubemap_sample_camera()
            else:
                theta, phi = cubemap_sample_camera()#icosahedron_sample_camera()
        else:
            raise NotImplementedError
        theta, phi = np.rad2deg(theta), np.rad2deg(phi) # Used to convert radians to degrees

        Ks, Rs = [], []
        for t, p in zip(theta, phi):
            K, R = get_K_R(self.config['fov'], t, p,
                           self.config['pers_resolution'], self.config['pers_resolution'])
            Ks.append(K)
            Rs.append(R)
        K = np.stack(Ks).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)

        cameras = {
            'height': np.full_like(theta, self.config['pers_resolution'], dtype=int),
            'width': np.full_like(theta, self.config['pers_resolution'], dtype=int),
            'FoV': np.full_like(theta, self.config['fov'], dtype=int),
            'theta': theta,
            'phi': phi,
            'R': R,
            'K': K,
        }
        data['cameras'] = cameras
        data['height'] = self.config['pano_height']
        data['width'] = self.config['pano_height'] * 2

        if 'rotation_type' in data:
            if data['rotation_type'] == 'relative':
                rotation = random.random() * 360 if self.mode == 'train' and self.config['rand_rot_img'] else 0
            elif data['rotation_type'] == 'absolute':
                rotation = random.random() * 10 if self.mode == 'train' and self.config['rand_rot_img'] else 0
        else:
            rotation = 0
        # flip = self.config['rand_flip'] and self.mode == 'train' and random.random() < 0.5
        flip = False

    
        def process_equi(equi, normalize, mode='train', function='add'):
            imgs = []
            refs = []
            images = []
            test_pers = []
            # Extract parameters - only process ref when function is 'add'
            if mode == 'train' and function == 'add' and self.use_ref==True:
                json_data = json.load(open(data['json_path']))
                transform_params = json_data.get("perspective_transform_params", {})
                center_u_deg = transform_params.get("center_u_deg")
                center_v_deg = transform_params.get("center_v_deg") 
                hfov_deg = transform_params.get("final_hfov_deg")
                vfov_deg = transform_params.get("final_vfov_deg")
                
                ref_resolution = (self.config['refs_resolution'], self.config['refs_resolution'])
                
                ref = equi.to_perspective((hfov_deg, vfov_deg), center_u_deg, center_v_deg, ref_resolution)
                
                # # Save original ref image
                # debug_dir = "debug_ref_images"
                # os.makedirs(debug_dir, exist_ok=True)
                
                # # Save original ref
                # ref_original = ref.copy()
                # ref_original_uint8 = np.clip(ref_original, 0, 255).astype(np.uint8)
                # cv2.imwrite(os.path.join(debug_dir, f"ref_original_{data.get('scene_id', 'unknown')}_{data.get('view_id', 'unknown')}.png"), 
                #            cv2.cvtColor(ref_original_uint8, cv2.COLOR_RGB2BGR))
                
                # If there's pano_mask, get perspective mask
                mask_img = cv2.imread(data['pano_mask_path'], cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.resize(mask_img, (data['width'], data['height']), interpolation=cv2.INTER_NEAREST)
                
                # Binarize mask (ensure only 0 and 1)
                mask_img = (mask_img > 128).astype(np.float32)

                # Get numpy array from pano_mask
                mask_array = mask_img.squeeze()  # Remove .numpy() call
                if hasattr(mask_array, 'numpy'):  # If it's a tensor, convert to numpy
                    mask_array = mask_array.numpy()
                if mask_array.ndim == 3:
                    mask_array = mask_array[0]  # Take the first channel
                
                # Create Equirectangular object for mask and apply same rotation and flip
                mask_equi = Equirectangular(mask_array[:, :, np.newaxis])
                # mask_equi.rotate(rotation)
                # mask_equi.flip(flip)
                
                # Project mask to perspective view using same parameters, maintaining same resolution as ref
                ref_mask = mask_equi.to_perspective((hfov_deg, vfov_deg), center_u_deg, center_v_deg, ref_resolution)
                
                # Ensure mask is binary
                if ref_mask.ndim == 3:
                    ref_mask = ref_mask[:, :, 0]
                ref_mask = (ref_mask > 0.5).astype(np.uint8)
                
                # Set mask to all 1s with certain probability, i.e., don't remove background
                if self.mode == 'train' and random.random() < self.config.get('refs_keep_background_prob', 0.3):
                    ref_mask = np.ones_like(ref_mask, dtype=np.uint8)
                
                # # Save ref_mask######
                # cv2.imwrite(os.path.join(debug_dir, f"ref_mask_{data.get('scene_id', 'unknown')}_{data.get('view_id', 'unknown')}.png"), 
                #            ref_mask * 255)
                
                # Get object pixels in ref
                ref = ref * ref_mask[:, :, np.newaxis]  # Extract pixels in mask region
                data['ref_mask'] = ref_mask  # Save perspective mask
                
                # # Save ref after applying mask #############
                # ref_masked_uint8 = np.clip(ref, 0, 255).astype(np.uint8)
                # cv2.imwrite(os.path.join(debug_dir, f"ref_masked_{data.get('scene_id', 'unknown')}_{data.get('view_id', 'unknown')}.png"), 
                #            cv2.cvtColor(ref_masked_uint8, cv2.COLOR_RGB2BGR))
                
                # Apply data augmentation to ref image
                ref = self.apply_refs_augmentation(ref, mode=self.mode)
                
            elif (mode == 'predict' or mode == 'test' or mode == 'val') and function == 'add' and self.use_ref==True:
                ref = None
                if data['ref_img_path'] is not None and len(data['ref_img_path']) > 0:
                    try:
                        ref_image = Image.open(data['ref_img_path'])
                    except:
                        print(f"convert jpg to png")
                        ref_image = Image.open(data['ref_img_path'].replace('.jpg', '.png'))
                        
                    # Directly resize to standard resolution
                    if hasattr(self, 'config') and 'refs_resolution' in self.config:
                        target_size = self.config['refs_resolution']
                        ref_image = ref_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    
                    ref = np.array(ref_image)

            else:
                ref = None

            if ref is not None and function == 'add':
                if ref.shape[:2] != (self.config['refs_resolution'], self.config['refs_resolution']):
                    ref = cv2.resize(ref, (self.config['refs_resolution'], self.config['refs_resolution']), 
                                   interpolation=cv2.INTER_LINEAR)
                refs.append(ref)
                refs = np.stack(refs)
                refs = (refs.astype(np.float32)/127.5)-1
                refs = rearrange(refs, 'b h w c -> b c h w')

            else:
                empty_ref = np.zeros((1, 3, self.config['refs_resolution'], self.config['refs_resolution']), dtype=np.float32)
                refs = empty_ref
            if self.use_fixed_pers_prompt==True and self.use_cubemap_prompt==True and self.only_pano==False:
                assert self.only_pano==False, "only_pano must be False when use_fixed_pers_prompt and use_cubemap_prompt are True"
                equi.rotate(rotation)
                equi.flip(flip)
                
                for t, p in zip(theta, phi):
                    img = equi.to_perspective((self.config['fov'], self.config['fov']), t, p, (self.config['pers_resolution'], self.config['pers_resolution']))
                    imgs.append(img)
                images = np.stack(imgs)
                if self.result_dir is None and normalize:
                    images = (images.astype(np.float32)/127.5)-1

                images = rearrange(images, 'b h w c -> b c h w')

            elif self.only_pano==True and self.use_fixed_pers_prompt==False and self.use_cubemap_prompt==False:
                equi.rotate(rotation)
                equi.flip(flip)

            pano = cv2.resize(equi.equirectangular, (data['width'], data['height']), interpolation=cv2.INTER_AREA)
            pano = pano.reshape(data['height'], data['width'], equi.equirectangular.shape[-1])
            if self.result_dir is None and normalize:
                pano = (pano.astype(np.float32)/127.5 - 1)
            pano = rearrange(pano, 'h w c -> 1 c h w')
            
            return pano, images, refs
        

        def process_mask_equi(mask_path, normalize):
 
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (data['width'], data['height']), interpolation=cv2.INTER_NEAREST)
            
            # Binarize mask (ensure only 0 and 1)
            mask_img = (mask_img > 128).astype(np.float32)
            
            # Create Equirectangular object for generating perspective view
            equi = Equirectangular(mask_img[:, :, None])  # Add channel dimension
            equi.rotate(rotation)
            equi.flip(flip)
            pano_mask = equi.equirectangular

            # Convert to expected format
            pano_mask = pano_mask[None,  :, :, :].copy()  # [1, 1, h, w]
            pano_mask = rearrange(pano_mask, 'b h w c -> b c h w')
            
            return pano_mask
        
        # load images
        if 'pano_path' in data and 'remove_pano_path' in data:

            equirectangular = Equirectangular.from_file(data['pano_path'])
            data['pano'],data['images'],data['refs'] = process_equi(equirectangular, True, mode=self.mode, function=data['function']) #, data['images']
            equirectangular = Equirectangular.from_file(data['remove_pano_path'])
            data['remove_pano'],data['remove_images'], _= process_equi(equirectangular, True, mode=self.mode, function=data['function']) #, data['remove_images'] 
            data['pano_mask'] = process_mask_equi(data['pano_mask_path'], True)



        #flip the perspective image prompt
        if flip:
            data['prompt'] = data['prompt'][::-1]

        # load pano prompt
        if 'pano_prompt' not in data:
            if data['function'] == 'remove':
                data['pano_prompt'] = self.load_prompt_remove(data['pano_prompt_path'])
                data['pano_prompt_without_img'] = self.load_prompt_without_img_remove(data['pano_prompt_path'])
                data['pano_prompt_with_ref'] = 'no'
            else:
                data['pano_prompt'] = self.load_prompt_add(data['pano_prompt_path'])
                data['pano_prompt_without_img'] = self.load_prompt_without_img_add(data['pano_prompt_path'])
                data['pano_prompt_with_ref'] = self.load_prompt_with_ref(data['pano_prompt_path'])
                if self.use_ref and self.mode != 'train':
                    data['pano_prompt_with_ref'] = self.load_prompt_with_ref(data['ref_pano_prompt_path'])

        if 'pano_simple_prompt' not in data and self.mode != 'predict' and self.mode != 'test':
            if data['function'] == 'remove':
                data['pano_simple_prompt'] = self.load_prompt_remove(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_without_img'] = self.load_prompt_without_img_remove(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_ref'] = 'no'
            else:
                data['pano_simple_prompt'] = self.load_prompt_add(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_without_img'] = self.load_prompt_without_img_add(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_ref'] = self.load_prompt_with_ref(data['pano_simple_prompt_path'])
                if self.use_ref and self.mode != 'train':
                    # Temporarily replace simple
                    data['pano_simple_prompt_with_ref'] = self.load_prompt_with_ref(data['ref_pano_prompt_path'])

        if self.mode == 'test':
            data['test_pers_prompt'] = self.load_test_pers_prompt(data['test_pers_prompt_path'])
        # load forward prompt
        # if 'forward_prompt' not in data:
        #     data['forward_prompt'] = self.load_prompt(data['forward_prompt_path'])
        # # load reverse prompt
        # if 'reverse_prompt' not in data:
        #     data['reverse_prompt'] = self.load_prompt(data['reverse_prompt_path'])
        # # load edited prompt
        # if 'edited_prompt' not in data:
        #     data['edited_prompt'] = self.load_prompt(data['edited_prompt_path'])
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['simple_prompt_ratio']:
            data['pano_prompt'] = data['pano_simple_prompt']
            if data['function'] == 'add':
                data['pano_prompt_with_ref'] = data['pano_simple_prompt_with_ref']
        # unconditioned training
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['conditioning_dropout_prob']:
            data['pano_prompt'] = ''
            # if 'prompt' in data:
            #     data['prompt'] = [''] * len(data['prompt'])
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['uncond_ratio_pers']:
            data['prompt'] = [''] * len(data['prompt'])
        # load results
        if self.config['gt_as_result']:
            data['pano_pred'] = data['pano']
            data['images_pred'] = data['images']
            # data['pano_pred'] = rearrange(data['pano_pred'], '1 c h w -> 1 h w c')
        elif self.result_dir is not None:
            if 'scene_id' in data:
                scene_id = data['scene_id']
                
                # Find edited_prompt*.txt files
                instruction = data['instruction']
                index = instruction.split('.')[0][-1]
                test_pers_prompt_files = data['test_pers_prompt']

            else:
                data['pano_prompt'] = data['pano_prompt']  
            images_pred = []
            for i in range(len(data['images'])):
                image_pred_path = os.path.join(os.path.dirname(data['pano_pred_path']), f"{i}.png")
                if not os.path.exists(image_pred_path):
                    break
                image_pred = Image.open(image_pred_path).convert('RGB')
                image_pred = np.array(image_pred)
                image_pred = cv2.resize(image_pred, (self.config['pers_resolution'], self.config['pers_resolution']))
                images_pred.append(image_pred)
            if images_pred:
                images_pred = np.stack(images_pred)
                data['images_pred'] = rearrange(images_pred, 'b h w c -> b c h w')

            if os.path.exists(data['pano_pred_path']):
                equirectangular = Equirectangular.from_file(data['pano_pred_path'])
                pano = cv2.resize(equirectangular.equirectangular, (data['width'], data['height']))
                pano = pano.reshape(data['height'], data['width'], equirectangular.equirectangular.shape[-1])
                data['pano_pred'] = rearrange(pano, 'h w c -> 1 c h w')
                if data['json_path'] is not None:
                    json_data = json.load(open(data['json_path']))
                    transform_params = json_data.get("perspective_transform_params", {})
                    center_u_deg = transform_params.get("final_center_u_deg")
                    center_v_deg = transform_params.get("final_center_v_deg") 
                    hfov_deg = transform_params.get("final_hfov_deg")
                    vfov_deg = transform_params.get("final_vfov_deg")
                    test_pers_resolution = (self.config['pers_resolution'], self.config['pers_resolution'])
                    test_pers = equirectangular.to_perspective((hfov_deg, vfov_deg), center_u_deg, center_v_deg, test_pers_resolution)
                    # test_pers = (test_pers.astype(np.float32)/127.5)-1
                    test_pers = rearrange(test_pers, 'h w c -> 1 c h w')
                    data['test_pers'] = test_pers
                if self.test_function == 'remove':
                    ori_equirectangular = Equirectangular.from_file(data['remove_pano_path'])
                else:
                    ori_equirectangular = Equirectangular.from_file(data['pano_path'])
                ori_pano = ori_equirectangular.equirectangular
                ori_pano = cv2.resize(ori_pano, (data['width'], data['height']))
                ori_pano = ori_pano.reshape(data['height'], data['width'], ori_pano.shape[-1])
                data['pano'] = rearrange(ori_pano, 'h w c -> 1 c h w')
                if data['json_path'] is not None:
                    json_data = json.load(open(data['json_path']))
                    transform_params = json_data.get("perspective_transform_params", {})
                    center_u_deg = transform_params.get("final_center_u_deg")
                    center_v_deg = transform_params.get("final_center_v_deg") 
                    hfov_deg = transform_params.get("final_hfov_deg")
                    vfov_deg = transform_params.get("final_vfov_deg")
                    test_pers_resolution = (self.config['pers_resolution'], self.config['pers_resolution'])
                    test_pers = ori_equirectangular.to_perspective((hfov_deg, vfov_deg), center_u_deg, center_v_deg, test_pers_resolution)
                    # test_pers = (test_pers.astype(np.float32)/127.5)-1
                    test_pers = rearrange(test_pers, 'h w c -> 1 c h w')
                    data['ori_test_pers'] = test_pers
            elif 'images_pred' in data:
                # merge images for MVDiffusion results
                pano = mp2e(
                    images_pred, cameras['FoV'], cameras['theta'], cameras['phi'],
                    (data['height'], data['width']))
                data['pano_pred'] = rearrange(pano, 'h w c -> 1 c h w')


        return data

    def apply_refs_augmentation(self, ref_img, mode='train'):
        if mode != 'train' or not self.config.get('refs_augmentation', True):
            return ref_img
        
        augmented_ref = ref_img.copy()
        
        # 1. random flip
        if random.random() < self.config.get('refs_flip_prob', 0.5):
            augmented_ref = np.flip(augmented_ref, axis=1).copy()
        
        # 2. random rotation (-15° to +15°)
        if random.random() < self.config.get('refs_rotation_prob', 0.7):
            angle = random.uniform(-15, 15)
            h, w = augmented_ref.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_ref = cv2.warpAffine(augmented_ref, M, (w, h), 
                                         borderMode=cv2.BORDER_REFLECT_101)
        
        # 3. random affine transformation (slight perspective transformation)
        if random.random() < self.config.get('refs_affine_prob', 0.3):
            h, w = augmented_ref.shape[:2]
            pts1 = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float32)
            # add random offset (maximum offset is 3% of image size, reduce offset to avoid excessive deformation)
            offset = min(w, h) * 0.03
            # ensure the points after offset are still reasonable
            offsets = np.random.uniform(-offset, offset, (4, 2)).astype(np.float32)
            pts2 = pts1 + offsets
            
            # ensure the points after transformation are within reasonable range
            pts2[:, 0] = np.clip(pts2[:, 0], -w*0.1, w*1.1)
            pts2[:, 1] = np.clip(pts2[:, 1], -h*0.1, h*1.1)
            
            try:
                M = cv2.getPerspectiveTransform(pts1, pts2)
                augmented_ref = cv2.warpPerspective(augmented_ref, M, (w, h),
                                                  borderMode=cv2.BORDER_REFLECT_101)
            except cv2.error:
                # if perspective transformation fails, skip this augmentation
                print("perspective transformation failed")
                pass
        
        # 4. random scale and crop
        if random.random() < self.config.get('refs_scale_prob', 0.5):
            h, w = augmented_ref.shape[:2]
            scale = random.uniform(0.9, 1.1)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # scale image
            augmented_ref = cv2.resize(augmented_ref, (new_w, new_h), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # if the scaled size is larger than the original size, perform center crop
            if new_h > h or new_w > w:
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                augmented_ref = augmented_ref[start_y:start_y+h, start_x:start_x+w]
            # if the scaled size is smaller than the original size, perform padding
            elif new_h < h or new_w < w:
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                augmented_ref = np.pad(augmented_ref, 
                                     ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)),
                                     mode='reflect')
        
        # 5. random color enhancement
        if random.random() < self.config.get('refs_color_prob', 0.6):
            # brightness adjustment
            brightness_factor = random.uniform(0.8, 1.2)
            augmented_ref = np.clip(augmented_ref * brightness_factor, 0, 255)
            
                # contrast adjustment
            if random.random() < 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                mean_val = np.mean(augmented_ref)
                augmented_ref = np.clip((augmented_ref - mean_val) * contrast_factor + mean_val, 0, 255)
            
            # saturation adjustment
            if random.random() < 0.5 and len(augmented_ref.shape) == 3:
                saturation_factor = random.uniform(0.8, 1.2)
                # convert to HSV space and adjust saturation
                try:
                    hsv = cv2.cvtColor(augmented_ref.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
                    augmented_ref = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
                except cv2.error:
                    # if color space conversion fails, skip saturation adjustment
                    pass
        
        # 6. random gaussian noise
        if random.random() < self.config.get('refs_noise_prob', 0.3):
            noise_std = random.uniform(1, 5)
            noise = np.random.normal(0, noise_std, augmented_ref.shape)
            augmented_ref = np.clip(augmented_ref + noise, 0, 255)
        
        # 7. random gaussian blur
        if random.random() < self.config.get('refs_blur_prob', 0.2):
            kernel_size = random.choice([3, 5])
            augmented_ref = cv2.GaussianBlur(augmented_ref, (kernel_size, kernel_size), 0)
        
        # 8. random sharpening
        if random.random() < self.config.get('refs_sharpen_prob', 0.2):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            sharpened = cv2.filter2D(augmented_ref, -1, kernel)
            # mix original image and sharpened image
            sharpen_factor = random.uniform(0.1, 0.3)
            augmented_ref = cv2.addWeighted(augmented_ref, 1-sharpen_factor, sharpened, sharpen_factor, 0)
        
        # 9. random gamma correction
        if random.random() < self.config.get('refs_gamma_prob', 0.3):
            gamma = random.uniform(0.7, 1.3)
            # normalize image to [0,1], apply gamma correction, then restore to [0,255]
            augmented_ref = augmented_ref / 255.0
            augmented_ref = np.power(np.clip(augmented_ref, 0, 1), gamma)
            augmented_ref = augmented_ref * 255.0
            augmented_ref = np.clip(augmented_ref, 0, 255)
        
        # 10. random block crop
        if random.random() < self.config.get('refs_block_crop_prob', 0.4):
            h, w = augmented_ref.shape[:2]
            # randomly generate 1-5 blocks
            num_blocks = random.randint(1, 5)
            
            for _ in range(num_blocks):
                # block size is 5%-15% of image size
                block_size_ratio = random.uniform(0.05, 0.15)
                block_h = int(h * block_size_ratio)
                block_w = int(w * block_size_ratio)
                
                # ensure the block size is at least 5x5 pixels
                block_h = max(5, block_h)
                block_w = max(5, block_w)
                
                # randomly select block position
                start_y = random.randint(0, max(0, h - block_h))
                start_x = random.randint(0, max(0, w - block_w))
                
                # fill block area with black
                augmented_ref[start_y:start_y+block_h, start_x:start_x+block_w] = 0
        
        return augmented_ref.astype(np.uint8)


class PanoDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = None,
            fov: int = 95,
            cam_sampler: str = 'icosahedron',  # 'horizon', 'icosahedron'
            refs_resolution: int = 256, # the size of reference image
            pers_resolution: int = 256,
            pano_height: int = 512, 
            # uncond_ratio: float = 0.2,
            conditioning_dropout_prob: float = 0.01,
            simple_prompt_ratio: float = 0.3,
            uncond_ratio_pers: float = 0,
            train_batch_size: int = 1,
            val_batch_size: int = 1,
            num_workers: int = 1,
            result_dir: str = None,
            rand_rot_img: bool = True,
            rand_flip: bool = False,
            gt_as_result: bool = False,
            repeat_predict: int = 10,
            only_pano: bool = True,
            use_fixed_pers_prompt: bool = False,
            use_cubemap_prompt: bool = False,
            test_function: str = 'add', #remove
            
            use_ref: bool = False,
            refs_augmentation: bool = True,      # whether to enable refs data augmentation
            refs_flip_prob: float = 0,        # horizontal flip probability
            refs_rotation_prob: float = 0.5,    # rotation probability
            refs_affine_prob: float = 0.2,      # affine transformation probability
            refs_scale_prob: float = 0.5,       # scale probability
            refs_color_prob: float = 0.2,       # color enhancement probability
            refs_noise_prob: float = 0.1,       # noise addition probability
            refs_blur_prob: float = 0.1,        # blur probability
            refs_sharpen_prob: float = 0.1,     # sharpening probability
            refs_gamma_prob: float = 0.1,       # gamma correction probability
            refs_block_crop_prob: float = 0.3,  # random block crop probability
            refs_keep_background_prob: float = 0.3,  # keep background probability (mask all 1)
            ):
        super().__init__()
        self.save_hyperparameters()
        self.result_dir = result_dir
        self.dataset_cls = PanoDataset

    def setup(self, stage=None):
        if self.result_dir is not None:
            self.hparams['result_dir'] = self.result_dir
        
        if stage in ('fit', None):
            self.train_dataset = self.dataset_cls(self.hparams, mode='train')

        if stage in ('fit', 'validate', None):
            self.val_dataset = self.dataset_cls(self.hparams, mode='val')

        if stage in ('test', None):
            self.test_dataset = self.dataset_cls(self.hparams, mode='test')

        if stage in ('predict', None):
            self.predict_dataset = self.dataset_cls(self.hparams, mode='predict')


        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.train_batch_size,
            shuffle=True, num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams.val_batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.hparams.val_batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
