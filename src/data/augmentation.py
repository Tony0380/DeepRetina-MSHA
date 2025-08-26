"""
Medical-specific augmentations for retinal images
Implements appropriate medical transformations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


class MedicalAugmentation:
    """
    Medical-specific augmentations for retinal images
    Preserves important diagnostic features
    """
    
    def __init__(
        self,
        image_size: int = 512,
        severity: str = 'medium'  # 'light', 'medium', 'heavy'
    ):
        self.image_size = image_size
        self.severity = severity
        self.severity_params = self._get_severity_params(severity)
    
    def _get_severity_params(self, severity: str) -> Dict:
        """
        Augmentation parameters based on severity level
        """
        params = {
            'light': {
                'rotation_limit': 10,
                'brightness_limit': 0.1,
                'contrast_limit': 0.1,
                'saturation_limit': 0.1,
                'hue_shift_limit': 5,
                'blur_limit': 3,
                'noise_var_limit': (10, 25)
            },
            'medium': {
                'rotation_limit': 15,
                'brightness_limit': 0.2,
                'contrast_limit': 0.2,
                'saturation_limit': 0.2,
                'hue_shift_limit': 10,
                'blur_limit': 5,
                'noise_var_limit': (10, 40)
            },
            'heavy': {
                'rotation_limit': 20,
                'brightness_limit': 0.3,
                'contrast_limit': 0.3,
                'saturation_limit': 0.3,
                'hue_shift_limit': 15,
                'blur_limit': 7,
                'noise_var_limit': (15, 50)
            }
        }
        return params[severity]
    
    def get_training_augmentation(self) -> A.Compose:
        """
        Augmentation per training - più aggressive ma medicalmente appropriate
        """
        params = self.severity_params
        
        return A.Compose([
            # Geometric transformations (conservative per preservare anatomia)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=params['rotation_limit'], 
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            
            # Optical distortions (simula variazioni nell'acquisizione)
            A.OpticalDistortion(
                distort_limit=0.05,
                shift_limit=0.05,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.05,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            
            # Color augmentations (simula variazioni di illuminazione)
            A.ColorJitter(
                brightness=params['brightness_limit'],
                contrast=params['contrast_limit'],
                saturation=params['saturation_limit'],
                hue=params['hue_shift_limit'] / 360.0,
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=params['brightness_limit'],
                contrast_limit=params['contrast_limit'],
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=params['hue_shift_limit'],
                sat_shift_limit=params['saturation_limit'] * 100,
                val_shift_limit=params['brightness_limit'] * 100,
                p=0.5
            ),
            
            # Gamma correction (simula variazioni di esposizione)
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            
            # Channel operations
            A.ChannelShuffle(p=0.1),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20, 
                b_shift_limit=20,
                p=0.3
            ),
            
            # Blur and noise (simula qualità dell'immagine)
            A.OneOf([
                A.MotionBlur(blur_limit=params['blur_limit'], p=0.4),
                A.MedianBlur(blur_limit=params['blur_limit'], p=0.3),
                A.GaussianBlur(blur_limit=params['blur_limit'], p=0.3),
            ], p=0.3),
            
            A.GaussNoise(
                var_limit=params['noise_var_limit'],
                mean=0,
                p=0.3
            ),
            
            # Advanced augmentations per robustezza
            A.OneOf([
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.3
                ),
                A.Perspective(
                    scale=(0.05, 0.1),
                    keep_size=True,
                    p=0.2
                )
            ], p=0.2),
            
            # Cutout per robustezza (occlusion)
            A.CoarseDropout(
                max_holes=8,
                max_height=self.image_size // 16,
                max_width=self.image_size // 16,
                min_holes=1,
                min_height=self.image_size // 32,
                min_width=self.image_size // 32,
                fill_value=0,
                p=0.3
            ),
            
            # Normalize e convert to tensor
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def get_validation_augmentation(self) -> A.Compose:
        """
        Augmentation per validation - solo normalizzazione
        """
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def get_test_time_augmentation(self) -> List[A.Compose]:
        """
        Test Time Augmentation per migliorare le predizioni
        """
        base_transform = [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
        
        tta_transforms = [
            # Original
            A.Compose(base_transform),
            
            # Horizontal flip
            A.Compose([
                A.HorizontalFlip(p=1.0)
            ] + base_transform),
            
            # Vertical flip  
            A.Compose([
                A.VerticalFlip(p=1.0)
            ] + base_transform),
            
            # Rotation
            A.Compose([
                A.Rotate(limit=5, p=1.0)
            ] + base_transform),
            
            # Brightness adjustment
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=1.0
                )
            ] + base_transform),
        ]
        
        return tta_transforms


class RetinalSpecificAugmentation:
    """
    Retinal pathology-specific augmentations
    """
    
    @staticmethod
    def simulate_microaneurysms(image: np.ndarray, p: float = 0.3, **kwargs) -> np.ndarray:
        """
        Simulates microaneurysms (small red dots)
        """
        if np.random.random() > p:
            return image
        
        h, w = image.shape[:2]
        num_points = np.random.randint(1, 5)
        
        for _ in range(num_points):
            x = np.random.randint(w//4, 3*w//4)
            y = np.random.randint(h//4, 3*h//4)
            radius = np.random.randint(1, 3)
            
            cv2.circle(image, (x, y), radius, (255, 0, 0), -1)
        
        return image
    
    @staticmethod
    def simulate_hemorrhages(image: np.ndarray, p: float = 0.2, **kwargs) -> np.ndarray:
        """
        Simulates retinal hemorrhages
        """
        if np.random.random() > p:
            return image
        
        h, w = image.shape[:2]
        num_hemorrhages = np.random.randint(1, 3)
        
        for _ in range(num_hemorrhages):
            x = np.random.randint(w//4, 3*w//4)
            y = np.random.randint(h//4, 3*h//4)
            size_x = np.random.randint(5, 15)
            size_y = np.random.randint(3, 8)
            
            cv2.ellipse(
                image, 
                (x, y), 
                (size_x, size_y), 
                np.random.randint(0, 180), 
                0, 360, 
                (150, 0, 0), 
                -1
            )
        
        return image
    
    @staticmethod
    def enhance_vessel_contrast(image: np.ndarray, p: float = 0.4, **kwargs) -> np.ndarray:
        """
        Enhances blood vessel contrast
        """
        if np.random.random() > p:
            return image
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Enhance contrast in L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced


def get_train_transforms(
    image_size: int = 512,
    severity: str = 'medium',
    use_retinal_specific: bool = True
) -> A.Compose:
    """
    Creates augmentation pipeline for training
    """
    medical_aug = MedicalAugmentation(image_size=image_size, severity=severity)
    transforms = medical_aug.get_training_augmentation()
    
    if use_retinal_specific:
        # Add retinal-specific augmentations
        additional_transforms = [
            A.Lambda(
                image=RetinalSpecificAugmentation.simulate_microaneurysms,
                p=0.2
            ),
            A.Lambda(
                image=RetinalSpecificAugmentation.simulate_hemorrhages,
                p=0.15
            ),
            A.Lambda(
                image=RetinalSpecificAugmentation.enhance_vessel_contrast,
                p=0.3
            )
        ]
        
        # Inserisci prima della normalizzazione
        transforms_list = list(transforms.transforms)
        insert_idx = -2  # Prima di Normalize e ToTensorV2
        for i, transform in enumerate(additional_transforms):
            transforms_list.insert(insert_idx + i, transform)
        
        transforms = A.Compose(transforms_list)
    
    return transforms


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """
    Crea pipeline di augmentation per validation
    """
    medical_aug = MedicalAugmentation(image_size=image_size)
    return medical_aug.get_validation_augmentation()


def get_tta_transforms(image_size: int = 512) -> List[A.Compose]:
    """
    Crea pipeline per Test Time Augmentation
    """
    medical_aug = MedicalAugmentation(image_size=image_size)
    return medical_aug.get_test_time_augmentation()


# Funzioni di utilità per visualizzazione augmentation
def visualize_augmentations(
    image: np.ndarray,
    transform: A.Compose,
    num_samples: int = 4
) -> List[np.ndarray]:
    """
    Visualizza esempi di augmentation
    """
    augmented_images = []
    
    for _ in range(num_samples):
        augmented = transform(image=image)['image']
        
        # Denormalize per visualizzazione
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.numpy().transpose(1, 2, 0)
            
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented = augmented * std + mean
        augmented = np.clip(augmented, 0, 1)
        
        augmented_images.append((augmented * 255).astype(np.uint8))
    
    return augmented_images
