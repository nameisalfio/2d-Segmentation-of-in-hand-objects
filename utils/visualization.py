import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import functional as F
from PIL import Image
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLUE_MASK_COLOR, MASK_ALPHA

def visualize_prediction(image, masks, alpha=MASK_ALPHA):
    """
    Visualizza un'immagine con maschere sovrapposte
    
    Args:
        image: Immagine di input (PIL Image o tensore)
        masks: Maschere predette
        alpha: Opacità delle maschere
        
    Returns:
        Immagine di visualizzazione
    """
    # Converti l'immagine in numpy se è un tensore
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            # Converti CHW in HWC
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            # Prendi la prima immagine se viene fornito un batch
            image = image[0].permute(1, 2, 0).cpu().numpy()
    
    # Converti in intervallo 0-255 e uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Crea una copia dell'immagine per la visualizzazione
    vis_image = image.copy()
    
    # Crea un overlay della maschera
    mask_overlay = np.zeros_like(vis_image)
    
    # Elabora ogni maschera
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Ottieni la maschera binaria
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        if mask.dtype != bool and mask.dtype != np.bool_:
            mask = mask > 0.5
        
        # Applica il colore alla maschera
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask] = BLUE_MASK_COLOR
        
        # Aggiungi all'overlay
        mask_overlay = np.maximum(mask_overlay, color_mask)
    
    # Unisci la maschera con l'immagine
    vis_image = cv2.addWeighted(vis_image, 1, mask_overlay, alpha, 0)
    
    return vis_image

def visualize_dataset_samples(dataset, num_samples=5, save_path=None):
    """
    Visualizza campioni dal dataset
    
    Args:
        dataset: Dataset da visualizzare
        num_samples: Numero di campioni da visualizzare
        save_path: Percorso per salvare le visualizzazioni
    """
    # Campiona indici casuali
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Crea una figura
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    # Elabora ogni campione
    for i, idx in enumerate(indices):
        # Ottieni il campione
        image, targets = dataset[idx]
        
        # Converti l'immagine in numpy
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Converti in intervallo 0-255
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        # Ottieni le maschere
        masks = targets['masks']
        
        # Converti le maschere in numpy se necessario
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = masks
        
        # Crea visualizzazione
        vis_image = visualize_prediction(image_np, masks_np)
        
        # Visualizza l'immagine originale
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Immagine Originale")
        axes[i, 0].axis('off')
        
        # Visualizza l'immagine con maschere
        axes[i, 1].imshow(vis_image)
        axes[i, 1].set_title(f"Immagine con Maschere di Oggetti")
        axes[i, 1].axis('off')
    
    # Adatta il layout
    plt.tight_layout()
    
    # Salva se viene fornito un percorso
    if save_path:
        plt.savefig(save_path)
        print(f"Visualizzazione salvata in {save_path}")
    
    # Mostra la figura
    plt.show()

def visualize_training_history(history, save_path=None):
    """
    Visualizza lo storico dell'addestramento
    
    Args:
        history: Dizionario con lo storico dell'addestramento
        save_path: Percorso per salvare la visualizzazione
    """
    # Crea una figura
    plt.figure(figsize=(10, 8))
    
    # Visualizza le metriche
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'])
    plt.title('Loss di Addestramento')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(history['val_miou'])
    plt.title('mIoU di Validazione')
    plt.xlabel('Epoca')
    plt.ylabel('mIoU')
    
    # Adatta il layout
    plt.tight_layout()
    
    # Salva se viene fornito un percorso
    if save_path:
        plt.savefig(save_path)
        print(f"Visualizzazione salvata in {save_path}")
    
    # Mostra la figura
    plt.show()

def visualize_test_predictions(model, test_dataset, indices=None, num_samples=5, save_dir=None):
    """
    Visualizza predizioni su campioni di test
    
    Args:
        model: Modello addestrato
        test_dataset: Dataset di test
        indices: Indici specifici da visualizzare (se None, vengono usati campioni casuali)
        num_samples: Numero di campioni da visualizzare
        save_dir: Directory per salvare le visualizzazioni
    """
    # Campiona indici casuali se non forniti
    if indices is None:
        indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    else:
        indices = indices[:num_samples]
    
    # Crea directory di salvataggio se fornita
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Elabora ogni campione
    for i, idx in enumerate(indices):
        # Ottieni il campione
        image, targets = test_dataset[idx]
        
        # Effettua la predizione
        pred_masks, pred_scores = model.predict(image, score_threshold=0.5)
        
        # Converti l'immagine in numpy
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Converti in intervallo 0-255
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        # Ottieni le maschere ground truth
        gt_masks = targets['masks']
        
        # Converti gt_masks in numpy se necessario
        if isinstance(gt_masks, torch.Tensor):
            gt_masks_np = gt_masks.cpu().numpy()
        else:
            gt_masks_np = gt_masks
        
        # Crea visualizzazioni
        gt_vis = visualize_prediction(image_np, gt_masks_np)
        pred_vis = visualize_prediction(image_np, pred_masks)
        
        # Crea figura
        plt.figure(figsize=(15, 5))
        
        # Visualizza l'immagine originale
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("Immagine Originale")
        plt.axis('off')
        
        # Visualizza il ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(gt_vis)
        plt.title("Ground Truth")
        plt.axis('off')
        
        # Visualizza le predizioni
        plt.subplot(1, 3, 3)
        plt.imshow(pred_vis)
        plt.title(f"Predizioni (n={len(pred_masks)})")
        plt.axis('off')
        
        # Adatta il layout
        plt.tight_layout()
        
        # Salva se viene fornita una directory
        if save_dir:
            save_path = os.path.join(save_dir, f"test_prediction_{i}.png")
            plt.savefig(save_path)
            print(f"Visualizzazione salvata in {save_path}")
        
        # Mostra la figura
        plt.show()

def save_visualization(fig, output_path):
    """
    Save the visualization figure to a specified path.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Path to save the visualization
    """
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)