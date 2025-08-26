"""
Main evaluator for MSHA models
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from tqdm import tqdm

from ..training.metrics import MedicalMetrics, MetricsTracker
from .clinical_metrics import ClinicalMetricsCalculator
from .uncertainty_analysis import UncertaintyAnalyzer


class ModelEvaluator:
    """
    Complete evaluator for MSHA models
    """
    
    def __init__(
        self,
        model,
        device: str = 'auto',
        num_classes: int = 5,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Modello da valutare
            device: Device per evaluation
            num_classes: Numero di classi
            class_names: Nomi delle classi
        """
        self.model = model
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Inizializza moduli di evaluation
        self.metrics_tracker = MetricsTracker(num_classes)
        self.medical_metrics = MedicalMetrics(num_classes, self.class_names)
        self.clinical_calculator = ClinicalMetricsCalculator(self.class_names)
        self.uncertainty_analyzer = UncertaintyAnalyzer()
    
    def evaluate_dataset(
        self,
        dataloader,
        use_uncertainty: bool = True,
        uncertainty_samples: int = 50,
        save_predictions: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Valuta il modello su un dataset completo
        
        Args:
            dataloader: DataLoader per evaluation
            use_uncertainty: Se calcolare uncertainty
            uncertainty_samples: Numero di campioni MC per uncertainty
            save_predictions: Se salvare le predizioni
            output_dir: Directory per salvare risultati
        
        Returns:
            results: Risultati completi della valutazione
        """
        print(f"🔬 Evaluating model on {len(dataloader)} batches...")
        
        # Reset tracker
        self.metrics_tracker.reset()
        
        # Liste per raccogliere dati
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        all_image_names = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                image_names = batch.get('image_name', [f'batch_{batch_idx}_img_{i}' for i in range(len(images))])
                
                # Forward pass standard
                outputs = self.model(images)
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = outputs['predictions']
                
                # Aggiorna tracker
                self.metrics_tracker.update(
                    predictions=predictions,
                    targets=labels,
                    probabilities=probabilities
                )
                
                # Salva per analisi dettagliate
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_image_names.extend(image_names)
                
                # Uncertainty estimation su subset
                if use_uncertainty and batch_idx < 10:  # Limita per velocità
                    if hasattr(self.model, 'monte_carlo_predict'):
                        mc_results = self.model.monte_carlo_predict(images, uncertainty_samples)
                        uncertainties = mc_results['total_uncertainty'].cpu().numpy()
                        all_uncertainties.extend(uncertainties)
        
        # Calcola metriche standard
        standard_metrics = self.metrics_tracker.compute_metrics()
        
        # Converti a numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Metriche cliniche dettagliate
        clinical_metrics = self.clinical_calculator.calculate_comprehensive_metrics(
            y_true, y_pred, y_prob
        )
        
        # Analisi uncertainty se disponibile
        uncertainty_results = {}
        if all_uncertainties:
            uncertainties = np.array(all_uncertainties[:len(y_true)])  # Truncate to match
            uncertainty_results = self.uncertainty_analyzer.analyze_uncertainty(
                uncertainties, y_pred[:len(uncertainties)], y_true[:len(uncertainties)]
            )
        
        # Combina tutti i risultati
        results = {
            'standard_metrics': standard_metrics,
            'clinical_metrics': clinical_metrics,
            'uncertainty_analysis': uncertainty_results,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist(),
                'image_names': all_image_names,
                'uncertainties': all_uncertainties if all_uncertainties else []
            },
            'evaluation_info': {
                'num_samples': len(y_true),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'device': str(self.device),
                'uncertainty_enabled': len(all_uncertainties) > 0
            }
        }
        
        # Salva risultati se richiesto
        if save_predictions and output_dir:
            self._save_evaluation_results(results, output_dir)
        
        return results
    
    def evaluate_single_image(
        self,
        image_path: str,
        transforms,
        use_uncertainty: bool = True,
        uncertainty_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Valuta una singola immagine
        
        Args:
            image_path: Path dell'immagine
            transforms: Trasformazioni da applicare
            use_uncertainty: Se calcolare uncertainty
            uncertainty_samples: Numero di campioni MC
        
        Returns:
            result: Risultato della predizione
        """
        from PIL import Image
        import cv2
        
        # Carica immagine
        if Path(image_path).suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Formato immagine non supportato: {image_path}")
        
        # Applica transforms
        transformed = transforms(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Predizione standard
            outputs = self.model(input_tensor)
            probabilities = outputs['predictions'][0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            result = {
                'predicted_class': predicted_class,
                'predicted_grade': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy(),
                'logits': outputs['logits'][0].cpu().numpy()
            }
            
            # Uncertainty se richiesta
            if use_uncertainty and hasattr(self.model, 'monte_carlo_predict'):
                mc_results = self.model.monte_carlo_predict(input_tensor, uncertainty_samples)
                result['uncertainty'] = {
                    'total': mc_results['total_uncertainty'][0].mean().item(),
                    'epistemic': mc_results['epistemic_uncertainty'][0].mean().item(),
                    'aleatoric': mc_results['aleatoric_uncertainty'][0].mean().item(),
                    'confidence': mc_results['confidence'][0].item()
                }
        
        return result
    
    def compare_with_ground_truth(
        self,
        predictions: Dict[str, Any],
        ground_truth_file: str
    ) -> Dict[str, Any]:
        """
        Confronta predizioni con ground truth
        """
        import pandas as pd
        
        # Carica ground truth
        gt_df = pd.read_csv(ground_truth_file)
        
        # Crea mapping image_name -> true_label
        gt_mapping = dict(zip(gt_df['image'], gt_df['level']))
        
        # Trova matches
        matched_predictions = []
        matched_targets = []
        matched_names = []
        
        for i, image_name in enumerate(predictions['image_names']):
            # Pulisci nome immagine
            clean_name = Path(image_name).stem
            if clean_name.endswith('_left') or clean_name.endswith('_right'):
                clean_name = clean_name
            elif not clean_name.endswith('.png'):
                clean_name = clean_name + '.png'
            
            if clean_name in gt_mapping:
                matched_predictions.append(predictions['y_pred'][i])
                matched_targets.append(gt_mapping[clean_name])
                matched_names.append(image_name)
        
        if matched_predictions:
            # Calcola metriche sui matches
            y_true = np.array(matched_targets)
            y_pred = np.array(matched_predictions)
            
            matched_metrics = self.medical_metrics.clinical_metrics(y_true, y_pred)
            
            return {
                'matched_samples': len(matched_predictions),
                'total_predictions': len(predictions['y_pred']),
                'match_rate': len(matched_predictions) / len(predictions['y_pred']),
                'metrics': matched_metrics,
                'matched_data': {
                    'y_true': y_true.tolist(),
                    'y_pred': y_pred.tolist(),
                    'image_names': matched_names
                }
            }
        else:
            return {'error': 'No matches found between predictions and ground truth'}
    
    def _save_evaluation_results(self, results: Dict[str, Any], output_dir: str):
        """
        Salva risultati di evaluation
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salva metriche JSON
        metrics_file = output_path / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'standard_metrics': results['standard_metrics'],
                'clinical_metrics': results['clinical_metrics'],
                'uncertainty_analysis': results['uncertainty_analysis'],
                'evaluation_info': results['evaluation_info']
            }, f, indent=2, default=str)
        
        # Salva predizioni CSV
        predictions_file = output_path / 'predictions.csv'
        pred_df = pd.DataFrame({
            'image_name': results['predictions']['image_names'],
            'true_label': results['predictions']['y_true'],
            'predicted_label': results['predictions']['y_pred'],
            'confidence': [prob[pred] for prob, pred in zip(
                results['predictions']['y_prob'], 
                results['predictions']['y_pred']
            )]
        })
        
        # Aggiungi uncertainty se disponibile
        if results['predictions']['uncertainties']:
            pred_df['uncertainty'] = results['predictions']['uncertainties']
        
        pred_df.to_csv(predictions_file, index=False)
        
        print(f"Results saved in: {output_path}")
        print(f"• Metrics: {metrics_file}")
        print(f"• Predictions: {predictions_file}")


def evaluate_model(
    model,
    dataloader,
    device: str = 'auto',
    num_classes: int = 5,
    class_names: Optional[List[str]] = None,
    use_uncertainty: bool = True,
    uncertainty_samples: int = 50
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model on a dataset
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device for evaluation
        num_classes: Number of classes
        class_names: Names of classes
        use_uncertainty: Whether to calculate uncertainty
        uncertainty_samples: Number of MC samples for uncertainty
    
    Returns:
        results: Complete evaluation results
    """
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        num_classes=num_classes,
        class_names=class_names
    )
    
    return evaluator.evaluate_dataset(
        dataloader=dataloader,
        use_uncertainty=use_uncertainty,
        uncertainty_samples=uncertainty_samples
    )
