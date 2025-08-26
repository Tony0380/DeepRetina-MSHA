"""
Dataset class for EyePACS and utilities for data loading
"""

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A

from .preprocessing import RetinalPreprocessor
from .augmentation import get_train_transforms, get_val_transforms


class EyePACSDataset(Dataset):
    """
    Dataset class for EyePACS Diabetic Retinopathy dataset
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: str,
        transforms: Optional[A.Compose] = None,
        image_size: int = 512,
        use_preprocessing: bool = True,
        cache_images: bool = False
    ):
        """
        Args:
            dataframe: DataFrame with columns 'image' and 'level'
            images_dir: Directory containing the images
            transforms: Albumentations transforms
            image_size: Image dimensions
            use_preprocessing: Whether to use advanced preprocessing
            cache_images: Whether to cache images in memory (for small datasets)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms
        self.image_size = image_size
        self.use_preprocessing = use_preprocessing
        self.cache_images = cache_images
        
        # Setup preprocessor
        if use_preprocessing:
            self.preprocessor = RetinalPreprocessor(
                target_size=(image_size, image_size),
                normalize=False  # Normalization is in transforms
            )
        
        # Cache for images
        self.image_cache = {} if cache_images else None
        
        # Verify that all images exist
        self._verify_images()
        
        print(f"Dataset initialized with {len(self)} samples")
        print(f"Class distribution: {self.get_class_distribution()}")
    
    def _verify_images(self):
        """Verify that all images exist"""
        missing_images = []
        
        for idx, row in self.dataframe.iterrows():
            image_name = row['image']
            if not image_name.endswith('.png'):
                image_name += '.png'
            
            image_path = os.path.join(self.images_dir, image_name)
            if not os.path.exists(image_path):
                missing_images.append(image_name)
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images missing")
            if len(missing_images) < 10:
                print(f"Missing images: {missing_images}")
            
            # Remove rows with missing images
            valid_mask = ~self.dataframe['image'].apply(
                lambda x: (x + '.png' if not x.endswith('.png') else x) in missing_images
            )
            self.dataframe = self.dataframe[valid_mask].reset_index(drop=True)
            print(f"Dataset reduced to {len(self)} valid samples")
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a sample
        """
        row = self.dataframe.iloc[idx]
        image_name = row['image']
        label = int(row['level'])
        
        # Make sure the name has extension
        if not image_name.endswith('.png'):
            image_name += '.png'
        
        # Load image
        if self.cache_images and image_name in self.image_cache:
            image = self.image_cache[image_name].copy()
        else:
            image = self._load_image(image_name)
            if self.cache_images:
                self.image_cache[image_name] = image.copy()
        
        # Preprocessing
        if self.use_preprocessing:
            image = self.preprocessor(image)
        
        # Make sure image is in correct dimensions
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        else:
            # If no transforms, convert to tensor anyway
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_name': image_name,
            'index': idx
        }
    
    def _load_image(self, image_name: str) -> np.ndarray:
        """
        Load an image from disk
        """
        image_path = os.path.join(self.images_dir, image_name)
        
        try:
            # Load with OpenCV (faster)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to load {image_path}")
            
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Create fallback image
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        return image
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Returns class distribution"""
        return self.dataframe['level'].value_counts().sort_index().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate weights for class balancing"""
        class_counts = self.dataframe['level'].value_counts().sort_index()
        total_samples = len(self.dataframe)
        num_classes = len(class_counts)
        
        # Weights inversely proportional to frequency
        weights = total_samples / (num_classes * class_counts.values)
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> List[float]:
        """Calculate weights for each sample (for WeightedRandomSampler)"""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.dataframe['level']]
        return sample_weights


def create_train_val_split(
    labels_file: str,
    train_ratio: float = 0.8,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crea split train/validation stratificato
    
    Args:
        labels_file: Path al file CSV con labels
        train_ratio: Percentuale per training
        stratify: Se fare split stratificato
        random_state: Seed per riproducibilità
        
    Returns:
        train_df, val_df: DataFrames per train e validation
    """
    # Carica dataset
    df = pd.read_csv(labels_file)
    
    # Pulisci dati se necessario
    df = df.dropna(subset=['image', 'level'])
    df['level'] = df['level'].astype(int)
    
    print(f"Total dataset: {len(df)} samples")
    print(f"Class distribution:\n{df['level'].value_counts().sort_index()}")
    
    # Stratified split
    if stratify:
        stratify_column = df['level']
    else:
        stratify_column = None
    
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=stratify_column
    )
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Train distribution:\n{train_df['level'].value_counts().sort_index()}")
    
    print(f"\nValidation set: {len(val_df)} samples")
    print(f"Val distribution:\n{val_df['level'].value_counts().sort_index()}")
    
    return train_df, val_df


def create_data_loaders(
    labels_file: str,
    images_dir: str,
    batch_size: int = 32,
    image_size: int = 512,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: bool = True,
    cache_images: bool = False,
    augmentation_severity: str = 'medium'
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Crea DataLoaders per training e validation
    
    Returns:
        train_loader, val_loader, info_dict
    """
    # Crea split
    train_df, val_df = create_train_val_split(
        labels_file=labels_file,
        train_ratio=train_ratio
    )
    
    # Crea transforms
    train_transforms = get_train_transforms(
        image_size=image_size,
        severity=augmentation_severity
    )
    val_transforms = get_val_transforms(image_size=image_size)
    
    # Crea datasets
    train_dataset = EyePACSDataset(
        dataframe=train_df,
        images_dir=images_dir,
        transforms=train_transforms,
        image_size=image_size,
        cache_images=cache_images
    )
    
    val_dataset = EyePACSDataset(
        dataframe=val_df,
        images_dir=images_dir,
        transforms=val_transforms,
        image_size=image_size,
        cache_images=cache_images
    )
    
    # Weighted sampling per bilanciamento classi
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Crea DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Info dictionary
    info_dict = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'num_classes': 5,
        'class_weights': train_dataset.get_class_weights(),
        'train_distribution': train_dataset.get_class_distribution(),
        'val_distribution': val_dataset.get_class_distribution(),
        'image_size': image_size,
        'batch_size': batch_size
    }
    
    return train_loader, val_loader, info_dict


class TTA_Dataset(Dataset):
    """
    Dataset per Test Time Augmentation
    """
    
    def __init__(
        self,
        base_dataset: EyePACSDataset,
        tta_transforms: List[A.Compose]
    ):
        self.base_dataset = base_dataset
        self.tta_transforms = tta_transforms
        self.num_tta = len(tta_transforms)
    
    def __len__(self):
        return len(self.base_dataset) * self.num_tta
    
    def __getitem__(self, idx):
        base_idx = idx // self.num_tta
        tta_idx = idx % self.num_tta
        
        # Carica immagine originale (senza transforms)
        row = self.base_dataset.dataframe.iloc[base_idx]
        image_name = row['image']
        label = int(row['level'])
        
        if not image_name.endswith('.png'):
            image_name += '.png'
        
        image = self.base_dataset._load_image(image_name)
        
        if self.base_dataset.use_preprocessing:
            image = self.base_dataset.preprocessor(image)
        
        # Applica TTA transform specifico
        transformed = self.tta_transforms[tta_idx](image=image)
        image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_name': image_name,
            'base_index': base_idx,
            'tta_index': tta_idx
        }


def create_tta_loader(
    base_dataset: EyePACSDataset,
    tta_transforms: List[A.Compose],
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    """
    Crea DataLoader per Test Time Augmentation
    """
    tta_dataset = TTA_Dataset(base_dataset, tta_transforms)
    
    return DataLoader(
        tta_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


# Utility functions
def analyze_dataset_statistics(labels_file: str, images_dir: str) -> Dict:
    """
    Analizza statistiche complete del dataset
    """
    df = pd.read_csv(labels_file)
    
    # Basic stats
    stats = {
        'total_samples': len(df),
        'class_distribution': df['level'].value_counts().sort_index().to_dict(),
        'missing_labels': df['level'].isna().sum()
    }
    
    # Image stats (sample from first 100 images)
    sample_df = df.head(100)
    image_sizes = []
    
    for _, row in sample_df.iterrows():
        image_name = row['image']
        if not image_name.endswith('.png'):
            image_name += '.png'
        
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                image_sizes.append(img.shape[:2])
    
    if image_sizes:
        heights, widths = zip(*image_sizes)
        stats.update({
            'avg_height': np.mean(heights),
            'avg_width': np.mean(widths),
            'min_height': np.min(heights),
            'max_height': np.max(heights),
            'min_width': np.min(widths),
            'max_width': np.max(widths)
        })
    
    return stats


def verify_data_integrity(labels_file: str, images_dir: str) -> Dict:
    """
    Verifica l'integrità del dataset
    """
    df = pd.read_csv(labels_file)
    
    missing_images = []
    corrupted_images = []
    valid_images = 0
    
    for _, row in df.iterrows():
        image_name = row['image']
        if not image_name.endswith('.png'):
            image_name += '.png'
        
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            missing_images.append(image_name)
        else:
            try:
                img = cv2.imread(image_path)
                if img is None:
                    corrupted_images.append(image_name)
                else:
                    valid_images += 1
            except Exception:
                corrupted_images.append(image_name)
    
    return {
        'total_labels': len(df),
        'valid_images': valid_images,
        'missing_images': len(missing_images),
        'corrupted_images': len(corrupted_images),
        'integrity_ratio': valid_images / len(df),
        'missing_list': missing_images[:10],  # First 10
        'corrupted_list': corrupted_images[:10]  # First 10
    }
