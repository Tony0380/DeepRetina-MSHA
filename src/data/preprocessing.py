"""
Preprocessing specifico per immagini retiniche
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch
from PIL import Image, ImageEnhance


class RetinalPreprocessor:
    """
    Preprocessing avanzato per immagini retiniche
    Include normalizzazione, enhancement del contrasto e rimozione noise
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        enhance_contrast: bool = True,
        remove_noise: bool = True,
        crop_black_borders: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.remove_noise = remove_noise
        self.crop_black_borders = crop_black_borders
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applica preprocessing completo all'immagine retinica
        
        Args:
            image: Immagine input come array numpy [H, W, C]
            
        Returns:
            processed_image: Immagine preprocessata
        """
        # 1. Crop black borders se richiesto
        if self.crop_black_borders:
            image = self._crop_black_borders(image)
        
        # 2. Resize all'immagine target
        image = self._resize_image(image)
        
        # 3. Enhancement del contrasto
        if self.enhance_contrast:
            image = self._enhance_contrast(image)
        
        # 4. Rimozione noise
        if self.remove_noise:
            image = self._denoise(image)
        
        # 5. Normalizzazione
        if self.normalize:
            image = self._normalize_image(image)
        
        return image
    
    def _crop_black_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Rimuove i bordi neri dalle immagini retiniche
        """
        # Converti in grayscale per trovare la regione di interesse
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Trova contorni
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Trova il contorno più grande (dovrebbe essere la retina)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Aggiungi un piccolo padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop l'immagine
            if len(image.shape) == 3:
                return image[y:y+h, x:x+w, :]
            else:
                return image[y:y+h, x:x+w]
        
        return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize mantenendo aspect ratio con padding
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calcola scaling factor mantenendo aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crea immagine finale con padding
        if len(image.shape) == 3:
            final_image = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            final_image = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Centra l'immagine
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        
        if len(image.shape) == 3:
            final_image[start_y:start_y+new_h, start_x:start_x+new_w, :] = resized
        else:
            final_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return final_image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhancement del contrasto specifico per immagini retiniche
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            # Converti a LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Applica CLAHE al canale L
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Converti back a RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Immagine grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Rimozione noise preservando dettagli importanti
        """
        if len(image.shape) == 3:
            # Non-local means denoising per immagini a colori
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # Non-local means denoising per immagini grayscale
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizzazione dell'immagine
        """
        # Converti a float32
        image = image.astype(np.float32) / 255.0
        
        # Normalizzazione per canale (ImageNet stats adaptate per retinal images)
        if len(image.shape) == 3:
            # Stats specifiche per immagini retiniche
            mean = np.array([0.485, 0.456, 0.406])  # Valori ImageNet
            std = np.array([0.229, 0.224, 0.225])   # Valori ImageNet
            
            image = (image - mean) / std
        else:
            # Grayscale normalization
            image = (image - 0.485) / 0.229
        
        return image


class RetinalQualityAssessment:
    """
    Valutazione automatica della qualità delle immagini retiniche
    """
    
    def __init__(self, min_contrast: float = 0.1, min_sharpness: float = 0.3):
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness
    
    def assess_quality(self, image: np.ndarray) -> dict:
        """
        Valuta la qualità dell'immagine retinica
        
        Returns:
            quality_metrics: Dict con metriche di qualità
        """
        # Converti a grayscale se necessario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Contrasto (deviazione standard normalizzata)
        contrast = np.std(gray) / 255.0
        
        # 2. Sharpness (varianza del Laplaciano)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian) / (gray.shape[0] * gray.shape[1])
        
        # 3. Brightness (media)
        brightness = np.mean(gray) / 255.0
        
        # 4. Coverage (percentuale di pixel non neri)
        non_black_pixels = np.sum(gray > 10)
        total_pixels = gray.shape[0] * gray.shape[1]
        coverage = non_black_pixels / total_pixels
        
        # 5. Score complessivo
        quality_score = (
            (contrast >= self.min_contrast) * 0.3 +
            (sharpness >= self.min_sharpness) * 0.3 +
            (0.2 <= brightness <= 0.8) * 0.2 +
            (coverage >= 0.6) * 0.2
        )
        
        return {
            'contrast': contrast,
            'sharpness': sharpness,
            'brightness': brightness,
            'coverage': coverage,
            'quality_score': quality_score,
            'is_good_quality': quality_score >= 0.7
        }


def preprocess_retinal_image(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),
    return_quality: bool = False
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Funzione di utilità per preprocessing di una singola immagine
    
    Args:
        image_path: Path dell'immagine
        target_size: Dimensione target
        return_quality: Se True, ritorna anche metriche di qualità
        
    Returns:
        processed_image: Immagine preprocessata
        quality_metrics: Metriche di qualità (se richieste)
    """
    # Carica immagine
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    
    # Converti da BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Quality assessment se richiesto
    quality_metrics = None
    if return_quality:
        quality_assessor = RetinalQualityAssessment()
        quality_metrics = quality_assessor.assess_quality(image)
    
    # Preprocessing
    preprocessor = RetinalPreprocessor(target_size=target_size)
    processed_image = preprocessor(image)
    
    return processed_image, quality_metrics


def batch_preprocess_images(
    image_paths: list,
    target_size: Tuple[int, int] = (512, 512),
    quality_threshold: float = 0.7
) -> Tuple[list, list]:
    """
    Preprocessing in batch con filtering per qualità
    
    Returns:
        good_images: Lista di immagini di buona qualità
        poor_quality_paths: Lista di path con bassa qualità
    """
    good_images = []
    poor_quality_paths = []
    
    for image_path in image_paths:
        try:
            processed_image, quality_metrics = preprocess_retinal_image(
                image_path, target_size, return_quality=True
            )
            
            if quality_metrics['quality_score'] >= quality_threshold:
                good_images.append((image_path, processed_image))
            else:
                poor_quality_paths.append(image_path)
                
        except Exception as e:
            print(f"Errore nel preprocessing di {image_path}: {e}")
            poor_quality_paths.append(image_path)
    
    return good_images, poor_quality_paths
