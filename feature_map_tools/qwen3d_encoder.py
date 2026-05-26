import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
# import open3d as o3d
from torch_scatter import scatter_mean
from PIL import Image

class Qwen3DEncoder(nn.Module):
    """
    A 3D encoder that uses a pre-initialized Qwen vision-language model to extract features 
    from multiple images and projects them into 3D space using camera parameters and depth information.
    """
    
    def __init__(
        self, 
        model, 
        processor,
        voxel_size: float = 0.05,
        feature_dim: int = 2048,
        min_points_per_voxel: int = 1,
        device: Optional[str] = None
    ):
        """
        Initialize the Qwen3DEncoder with pre-initialized model and processor.
        
        Args:
            model: Pre-initialized Qwen VL model (e.g., Qwen2.5-VL-3B-Instruct)
            processor: Pre-initialized Qwen processor
            voxel_size: Size of voxels for point cloud discretization
            feature_dim: Dimension of the feature vectors
            min_points_per_voxel: Minimum number of points required in a voxel
            device: Device to run the model on (if None, uses model's device)
        """
        super().__init__()
        
        # run visual encoder in fp16
        self.processor = processor
        self.device = device or next(model.parameters()).device
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        self.min_points_per_voxel = min_points_per_voxel
        
        # Access the vision encoder component
        self.visual = model.to(device).to(torch.bfloat16)
        
        # Ensure the model is in eval mode for feature extraction
        self.visual.eval()
        
    def extract_features(
        self, 
        images: Union[torch.Tensor, List[Image.Image]],
        text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract features from images using the Qwen visual encoder.
        
        Args:
            images: Batch of images [B, C, H, W] or list of PIL images
            text: Optional text prompt to condition the feature extraction
            
        Returns:
            features: Extracted features [B, N, C]
        """
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            images_list = []
            for i in range(images.shape[0]):
                images_list.append(images[i].permute(1, 2, 0))
            
            # doing PIL image here. Nokos is writing on torch.tensor processing
            # Nikos's version will save us a lot of time
            image_inputs = self.processor.image_processor(
                images=images_list,
                do_rescale=True,
                return_tensors="pt"
            ).to(self.device)
            
            # print(f">>> Image processing time: {time.time() - time_1:.2f} seconds")
            
            # Get the processor-calculated grid_thw
            processed_images = image_inputs['pixel_values'].to(torch.bfloat16)
            processor_grid_thw = image_inputs['image_grid_thw']
            # Use the model's visual component to extract features
            
            vision_outputs = self.visual(
                processed_images, 
                grid_thw=processor_grid_thw
            )
            # print(f">>> Vision feature extraction time: {end_time - start_time:.2f} seconds with batch size {len(images_list)}")
            assert vision_outputs.dtype != torch.float16, "Vision outputs should never be in float16"
            assert vision_outputs.dtype == torch.bfloat16, "Vision outputs should be in bfloat16 for now"
                
        return vision_outputs, processor_grid_thw
    
    def forward(
        self, 
        images: torch.Tensor,
        # depths: torch.Tensor,
        # intrinsics: torch.Tensor,
        # extrinsics: torch.Tensor,
        text: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of images with associated depth maps and camera parameters.
        
        Args:
            images: Batch of images [B, C, H, W]
            depths: Batch of depth maps [B, H, W]
            intrinsics: Camera intrinsic matrices [B, 3, 3]
            extrinsics: Camera extrinsic matrices [B, 4, 4]
            text: Optional text prompt
            
        Returns:
            points_3d: Voxelized point cloud [N, 3]
            features_3d: Features for each point [N, C]
        """        
        features, grid_thw = self.extract_features(images, text) # qwen.visual
        return features, grid_thw

    def process_scene(
        self,
        images: List[torch.Tensor],
        text: Optional[str] = None,
        batch_size: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a scene with multiple images taken from different viewpoints.
        
        Args:
            images: List of images [N, C, H, W]
            depths: List of depth maps [N, H, W]
            intrinsics: List of camera intrinsic matrices [N, 3, 3]
            extrinsics: List of camera extrinsic matrices [N, 4, 4]
            text: Optional text prompt
            batch_size: Batch size for processing
            
        Returns:
            points_3d: Voxelized point cloud [M, 3]
            features_3d: Features for each point [M, C]
        """
        # all_points = []
        all_features = []
        all_grid_thw = []
        
        num_images = len(images) # 15, 200
        for i in range(0, num_images, batch_size):
            
            
            end_idx = min(i + batch_size, num_images)
            batch_images = torch.stack(images[i:end_idx]).to(self.device)
            features, grid_thw = self.forward(
                batch_images, 
                text
            )
            if features.shape[0] > 0:
                all_features.append(features)
                all_grid_thw.append(grid_thw)
        
        if not all_features:
            return torch.zeros(0, self.feature_dim, device=self.device)
        combined_features = torch.cat(all_features, dim=0)
        combined_grid_thw = torch.stack(all_grid_thw, dim=0)
        
        return combined_features, combined_grid_thw