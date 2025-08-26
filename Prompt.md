# DeepRetina-MSHA: Multi-Scale Hierarchical Attention Networks per Diabetic Retinopathy

Crea un progetto completo di deep learning per la classificazione di retinopatia diabetica usando il dataset EyePACS. Il progetto deve implementare un'architettura innovativa con attention gerarchica multi-scala.

## 📁 Struttura del Progetto

inventa tu la struttura del progetto, l'importante è che alla fine l'output comprenda un notebook da cui si possa utilizzare l'intero progetto

Sistema specifiche e settings in base ad una GPU di tipo 2060 Super 8 GB VRAM

## 🎯 Architettura MSHA (Multi-Scale Hierarchical Attention)

### Componenti Principali:

1. **Backbone**: EfficientNet-B4 pre-trainata
2. **Multi-Scale Feature Pyramid**: Estrazione features a 4 scale (1x, 2x, 4x, 8x)
3. **Hierarchical Attention**:
   - **Global Attention**: Attention sull'intera immagine (512x512)
   - **Regional Attention**: Attention su 4 quadranti (256x256 each)
   - **Local Attention**: Attention su patches locali (64x64)
4. **Feature Fusion**: Concatenazione pesata con pesi appresi
5. **Uncertainty Quantification**: Bayesian dropout layers
6. **Classification Head**: 5-class output (DR grades 0-4)

### Specifiche Tecniche:

- **Input Size**: 512x512x3
- **Batch Size**: 16-32 (dipende da GPU)
- **Optimizer**: AdamW con weight decay
- **Learning Rate**: Cosine annealing 1e-4 → 1e-6
- **Loss Function**: Focal Loss + Uncertainty Loss
- **Augmentation**: Advanced augmentation medicale-specifica

## 📊 Dataset Configuration

Il dataset EyePACS è nella cartella `dataset/` con questa struttura:
```
dataset/
├── images/           #  images
├── all_labels.csv        # labels

studia comunque il dataset per capire la sua conformazione, labels e funzionamento e divisione
```


## 📈 Training Configuration

### config/training_config.yaml
```yaml
model:
  name: "MSHANetwork"
  backbone: "efficientnet_b4"
  num_classes: 5
  dropout_rate: 0.3
  attention_dim: 256

training:
  batch_size: 24
  num_epochs: ?
  learning_rate: 1e-4
  weight_decay: 1e-5
  gradient_clip_val: 1.0
  
optimizer:
  name: "AdamW"
  betas: [0.9, 0.999]
  
scheduler:
  name: "CosineAnnealingWarmRestarts"
  T_0: 10
  T_mult: 2
  eta_min: 1e-6

loss:
  focal_alpha: [1, 2, 2, 3, 3]  # Pesi per classi
  focal_gamma: 2
  uncertainty_weight: 0.1

data:
  image_size: 512
  train_split: 0.8
  val_split: 0.2
  num_workers: 8
  pin_memory: true
```

## 🎯 Obiettivi Performance

Target da raggiungere:
- **Accuracy**: >90% su validation set
- **Quadratic Kappa**: >0.85
- **AUC-ROC**: >0.95 per classe majority
- **Inference Time**: <500ms per immagine
- **Memory Usage**: <8GB GPU per training

## ⚠️ Medical Disclaimer

Aggiungi sempre nei file appropriati:

```markdown
## ⚠️ Medical Disclaimer

This software is for **research purposes only** and is **not intended for clinical use**. 
It has not been approved by any regulatory authority for medical diagnosis. 
Always consult qualified healthcare professionals for medical decisions.

- 🚫 Not FDA approved
- 🚫 Not CE marked  
- 🚫 Not intended for patient care
- ✅ Research and educational use only
```

## 📄 MIT License

Crea file LICENSE con:

```
MIT License

Copyright (c) 2025 Antonio Colamartino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Inizia creando la struttura base del progetto e implementa tutti i componenti step by step. Ogni file deve essere production-ready con documentazione completa, error handling e logging appropriato.