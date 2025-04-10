import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNeXt101_32X8D_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from config import TENSORBOARD_DIR, NUM_CLASSES

class MaskRCNNModel:

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, backbone_name="resnext101_32x8d"):
        """
        Inizializza il modello Mask R-CNN con backbone ResNeXt-101-FPN
        
        Args:
            num_classes: Numero di classi (background + classi di oggetti)
            pretrained: Se usare pesi preaddestrati
            backbone_name: Nome del backbone da usare (default: "resnext101_32x8d")
        """
        # Seleziona il dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilizzando dispositivo: {self.device}")
        
        # Crea il backbone personalizzato
        if backbone_name == "resnext101_32x8d":
            print("Utilizzo backbone ResNeXt-101-FPN per Mask R-CNN")
            
            if pretrained:
                # Usa i pesi preaddestrati per il backbone
                weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V1
                print("Caricamento pesi preaddestrati su ImageNet")
            else:
                weights = None
                print("Inizializzazione da zero")
            
            # Crea il backbone ResNeXt-101 con FPN
            backbone = resnet_fpn_backbone(
                backbone_name=backbone_name, 
                weights=weights,
                trainable_layers=5  # Tutti i layers sono addestrabili
            )
            
            self.model = MaskRCNN(
                backbone=backbone,
                num_classes=num_classes,
                min_size=800,
                max_size=1333,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
            
        else:
            # Fallback al ResNet-50-FPN standard
            print(f"Backbone {backbone_name} non supportato, utilizzo ResNet-50-FPN standard")
            
            if pretrained:
                weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = maskrcnn_resnet50_fpn(weights=weights)
            else:
                self.model = maskrcnn_resnet50_fpn()
        
        # Ottieni il numero di feature di input
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Classificazione ternaria (background + generic object + in_hand object)
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Ottieni il numero di feature di input per il mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        # Sostituisci il mask predictor
        self.model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        # Sposta il modello sul dispositivo appropriato
        self.model.to(self.device)
        
        # Stampa informazioni sul modello
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Numero totale di parametri: {num_params:,}")

    def predict(self, image, score_threshold=0.5):
        """
        Esegue la predizione su una singola immagine, filtrando per classe
        
        Args:
            image: Immagine di input (PIL Image o tensore)
            score_threshold: Soglia di confidenza per le predizioni
            
        Returns:
            Tuple con maschere, punteggi, etichette per ogni classe
        """
        # Ottieni il modello effettivo (gestisce DataParallel)
        model_to_predict = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model_to_predict.eval()
        
        # Converti l'immagine in tensore se necessario
        if not isinstance(image, torch.Tensor):
            from torchvision.transforms import functional as F
            image = F.to_tensor(image)
        
        # Aggiungi la dimensione del batch e sposta sul dispositivo
        image = image.unsqueeze(0).to(self.device)
        
        # Esegui la predizione
        with torch.no_grad():
            prediction = model_to_predict(image)[0]
        
        # Ottieni maschere e punteggi
        masks = prediction['masks'].cpu()
        scores = prediction['scores'].cpu()
        labels = prediction['labels'].cpu()
        boxes = prediction['boxes'].cpu()
        
        # Applica la soglia di punteggio a tutte le predizioni
        high_conf_indices = scores >= score_threshold
        masks = masks[high_conf_indices]
        scores = scores[high_conf_indices]
        labels = labels[high_conf_indices]
        boxes = boxes[high_conf_indices]
        
        # CORREZIONE: Dividi le predizioni per classe
        class1_indices = labels == 1  # Oggetti non in mano
        class2_indices = labels == 2  # Oggetti in mano
        
        # Crea i risultati per ogni classe
        class1_results = {
            'masks': masks[class1_indices] if class1_indices.sum() > 0 else torch.tensor([]),
            'scores': scores[class1_indices] if class1_indices.sum() > 0 else torch.tensor([]),
            'labels': labels[class1_indices] if class1_indices.sum() > 0 else torch.tensor([]),
            'boxes': boxes[class1_indices] if class1_indices.sum() > 0 else torch.tensor([])
        }
        
        class2_results = {
            'masks': masks[class2_indices] if class2_indices.sum() > 0 else torch.tensor([]),
            'scores': scores[class2_indices] if class2_indices.sum() > 0 else torch.tensor([]),
            'labels': labels[class2_indices] if class2_indices.sum() > 0 else torch.tensor([]),
            'boxes': boxes[class2_indices] if class2_indices.sum() > 0 else torch.tensor([])
        }
        
        print("\nRisultati predizione:")
        print(f"Totale predizioni: {len(labels)}")
        print(f"Classe 1 (non in mano): {class1_indices.sum().item()} oggetti")
        print(f"Classe 2 (in mano): {class2_indices.sum().item()} oggetti")
        
        return {
            'all': {'masks': masks, 'scores': scores, 'labels': labels, 'boxes': boxes},
            'not_in_hand': class1_results,
            'in_hand': class2_results
        }

    def visualize_predictions(self, image_tensor, predictions, threshold=0.5):
        """
        Visualizza le predizioni con colori diversi per ogni classe
        
        Args:
            image_tensor: Tensore dell'immagine
            predictions: Dizionario con le predizioni
            threshold: Soglia per binarizzare le maschere
            
        Returns:
            Immagine numpy con le visualizzazioni
        """
        # Converti l'immagine in numpy
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Crea una copia dell'immagine per ogni categoria
        img_all = image_np.copy()
        img_not_in_hand = image_np.copy()
        img_in_hand = image_np.copy()
        
        # Colori per le diverse classi
        colors = {
            1: (255, 0, 0),    # Rosso per classe 1 (non in mano)
            2: (0, 255, 0)     # Verde per classe 2 (in mano)
        }
        
        # Visualizza tutte le predizioni
        if len(predictions['all']['masks']) > 0:
            for i, (mask, label, box, score) in enumerate(zip(
                    predictions['all']['masks'], 
                    predictions['all']['labels'], 
                    predictions['all']['boxes'],
                    predictions['all']['scores'])):
                
                # Binarizza la maschera
                mask_np = mask.squeeze().numpy() > threshold
                
                # Applica miglioramenti alla maschera
                mask_np = self._refine_mask(mask_np)
                
                # Colore basato sulla classe
                color = colors.get(label.item(), (255, 255, 255))
                
                # Sovrapponi la maschera all'immagine
                img_all = self._overlay_mask(img_all, mask_np, color, alpha=0.5)
                
                # Disegna il bounding box
                x1, y1, x2, y2 = box.int().numpy()
                cv2.rectangle(img_all, (x1, y1), (x2, y2), color, 2)
                
                # Aggiungi etichetta e punteggio
                text = f"Class {label.item()}: {score.item():.2f}"
                cv2.putText(img_all, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Visualizza oggetti non in mano (classe 1)
        if len(predictions['not_in_hand']['masks']) > 0:
            for i, (mask, box, score) in enumerate(zip(
                    predictions['not_in_hand']['masks'], 
                    predictions['not_in_hand']['boxes'],
                    predictions['not_in_hand']['scores'])):
                
                # Binarizza la maschera
                mask_np = mask.squeeze().numpy() > threshold
                
                # Applica miglioramenti alla maschera
                mask_np = self._refine_mask(mask_np)
                
                # Sovrapponi la maschera all'immagine
                img_not_in_hand = self._overlay_mask(img_not_in_hand, mask_np, colors[1], alpha=0.5)
                
                # Disegna il bounding box
                x1, y1, x2, y2 = box.int().numpy()
                cv2.rectangle(img_not_in_hand, (x1, y1), (x2, y2), colors[1], 2)
                
                # Aggiungi punteggio
                text = f"Score: {score.item():.2f}"
                cv2.putText(img_not_in_hand, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 2)
        
        # Visualizza oggetti in mano (classe 2)
        if len(predictions['in_hand']['masks']) > 0:
            for i, (mask, box, score) in enumerate(zip(
                    predictions['in_hand']['masks'], 
                    predictions['in_hand']['boxes'],
                    predictions['in_hand']['scores'])):
                
                # Binarizza la maschera
                mask_np = mask.squeeze().numpy() > threshold
                
                # Applica miglioramenti alla maschera
                mask_np = self._refine_mask(mask_np)
                
                # Sovrapponi la maschera all'immagine
                img_in_hand = self._overlay_mask(img_in_hand, mask_np, colors[2], alpha=0.5)
                
                # Disegna il bounding box
                x1, y1, x2, y2 = box.int().numpy()
                cv2.rectangle(img_in_hand, (x1, y1), (x2, y2), colors[2], 2)
                
                # Aggiungi punteggio
                text = f"Score: {score.item():.2f}"
                cv2.putText(img_in_hand, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 2)
        
        return {
            'all': img_all,
            'not_in_hand': img_not_in_hand,
            'in_hand': img_in_hand
        }

    def _refine_mask(self, mask_np, kernel_size=3):
        """
        Migliora la qualità della maschera con operazioni morfologiche
        
        Args:
            mask_np: Maschera binaria numpy
            kernel_size: Dimensione del kernel per le operazioni morfologiche
            
        Returns:
            Maschera binaria numpy migliorata
        """
        # Converti in formato corretto per opencv
        mask_uint8 = mask_np.astype(np.uint8) * 255
        
        # Crea il kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Applica apertura morfologica (erosione seguita da dilatazione)
        # Rimuove piccole imperfezioni e rumore
        mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Applica chiusura morfologica (dilatazione seguita da erosione)
        # Riempie piccoli buchi nella maschera
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
        
        # Riconverti in maschera binaria
        return mask_closed > 0

    def _overlay_mask(self, image, mask, color, alpha=0.5):
        """
        Sovrappone una maschera colorata all'immagine
        
        Args:
            image: Immagine numpy
            mask: Maschera binaria numpy
            color: Tupla (R, G, B) per il colore della maschera
            alpha: Trasparenza della sovrapposizione
            
        Returns:
            Immagine numpy con la maschera sovrapposta
        """
        # Crea un'immagine per la maschera
        mask_image = np.zeros_like(image)
        
        # Applica il colore alla maschera
        mask_image[mask] = color
        
        # Sovrapponi la maschera all'immagine originale
        return cv2.addWeighted(image, 1.0, mask_image, alpha, 0)
    
    def evaluate(self, data_loader, epoch=None, writer=None, iou_threshold=0.5):
        """
        Valuta il modello su dati di validazione/test, separando le metriche per classe
        
        Args:
            data_loader: DataLoader per i dati di validazione/test
            epoch: Numero dell'epoca corrente (per logging)
            writer: SummaryWriter per TensorBoard logging
            iou_threshold: Soglia IoU per considerare una predizione corretta
            
        Returns:
            Dictionary con metriche di valutazione per classe
        """
        # Ottieni il modello effettivo (gestisce DataParallel)
        model_to_eval = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model_to_eval.eval()
        
        # Liste per memorizzare le metriche
        class_ious = {1: [], 2: []}
        class_tp = {1: 0, 2: 0}    # True positives
        class_fp = {1: 0, 2: 0}    # False positives
        class_fn = {1: 0, 2: 0}    # False negatives
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Valutazione per classe"):
                # Sposta le immagini sul dispositivo
                images = [image.to(self.device) for image in images]
                
                # Ottieni le predizioni
                predictions = self.model(images)
                
                # Elabora ogni immagine
                for i, (pred, target) in enumerate(zip(predictions, targets)):
                    # Sposta il target su CPU per la valutazione
                    target = {k: v.cpu() for k, v in target.items()}
                    
                    # Ottieni le maschere e etichette predette
                    pred_masks = pred['masks'].cpu()
                    pred_scores = pred['scores'].cpu()
                    pred_labels = pred['labels'].cpu()
                    
                    # Ottieni le maschere e etichette target
                    target_masks = target['masks']
                    target_labels = target['labels']
                    
                    # Salta le immagini senza maschere target
                    if len(target_masks) == 0:
                        continue
                    
                    # Converti le maschere target in formato binario (0 o 1)
                    target_masks = target_masks > 0.5
                    
                    # Usa solo le predizioni ad alta confidenza
                    high_conf_indices = pred_scores > 0.5
                    if high_conf_indices.sum() == 0:
                        # Per ogni target mancato, incrementa FN per quella classe
                        for target_label in target_labels:
                            class_fn[target_label.item()] += 1
                        continue
                    
                    pred_masks = pred_masks[high_conf_indices]
                    pred_labels = pred_labels[high_conf_indices]
                    
                    # Converti le maschere predette in formato binario
                    pred_masks = pred_masks > 0.5
                    
                    # Salta se non ci sono maschere predette dopo il thresholding
                    if len(pred_masks) == 0:
                        # Per ogni target mancato, incrementa FN per quella classe
                        for target_label in target_labels:
                            class_fn[target_label.item()] += 1
                        continue
                    
                    # Calcola IoU per ogni coppia di maschera predetta e target della stessa classe
                    for pred_idx, (pred_mask, pred_label) in enumerate(zip(pred_masks, pred_labels)):
                        pred_mask = pred_mask.squeeze()
                        pred_mask_area = pred_mask.sum().item()
                        pred_label = pred_label.item()
                        
                        # Se la maschera predetta è vuota, è un falso positivo
                        if pred_mask_area == 0:
                            class_fp[pred_label] += 1
                            continue
                        
                        # Trova le maschere target della stessa classe
                        matching_target_indices = [i for i, l in enumerate(target_labels) if l.item() == pred_label]
                        
                        if not matching_target_indices:
                            # Se non ci sono target della stessa classe, è un falso positivo
                            class_fp[pred_label] += 1
                            continue
                        
                        # Calcola IoU con tutte le maschere target della stessa classe
                        max_iou = 0.0
                        best_target_idx = -1
                        
                        for target_idx in matching_target_indices:
                            target_mask = target_masks[target_idx].squeeze()
                            target_mask_area = target_mask.sum().item()
                            
                            # Se la maschera target è vuota, salta
                            if target_mask_area == 0:
                                continue
                            
                            # Calcola intersezione e unione
                            intersection = (pred_mask & target_mask).sum().item()
                            union = pred_mask_area + target_mask_area - intersection
                            
                            # Calcola IoU
                            iou = intersection / union if union > 0 else 0.0
                            
                            if iou > max_iou:
                                max_iou = iou
                                best_target_idx = target_idx
                        
                        # Se abbiamo trovato un match valido
                        if max_iou > 0:
                            class_ious[pred_label].append(max_iou)
                            
                            # Se IoU è sopra la soglia, è un vero positivo
                            if max_iou > iou_threshold:
                                class_tp[pred_label] += 1
                            else:
                                class_fp[pred_label] += 1
                        else:
                            class_fp[pred_label] += 1
                    
                    # Controlla per false negative (target non rilevati)
                    # Per ogni classe, conta quanti target non hanno una predizione corrispondente
                    for target_class in [1, 2]:
                        target_indices = [i for i, l in enumerate(target_labels) if l.item() == target_class]
                        pred_indices = [i for i, l in enumerate(pred_labels) if l.item() == target_class]
                        
                        if len(target_indices) > len(pred_indices):
                            class_fn[target_class] += (len(target_indices) - len(pred_indices))
            
            # Calcola le metriche per ciascuna classe
            class_metrics = {}
            
            for class_id in [1, 2]:
                # Calcola IoU medio
                avg_iou = sum(class_ious[class_id]) / len(class_ious[class_id]) if class_ious[class_id] else 0.0
                
                # Calcola precision e recall
                precision = class_tp[class_id] / (class_tp[class_id] + class_fp[class_id]) if (class_tp[class_id] + class_fp[class_id]) > 0 else 0.0
                recall = class_tp[class_id] / (class_tp[class_id] + class_fn[class_id]) if (class_tp[class_id] + class_fn[class_id]) > 0 else 0.0
                
                # Calcola F1 score
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                class_metrics[class_id] = {
                    "mIoU": avg_iou,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": class_tp[class_id],
                    "fp": class_fp[class_id],
                    "fn": class_fn[class_id]
                }
            
            # Log di TensorBoard
            if writer and epoch is not None:
                for class_id in [1, 2]:
                    writer.add_scalar(f'Epoch/class{class_id}_mIoU', class_metrics[class_id]["mIoU"], epoch)
                    writer.add_scalar(f'Epoch/class{class_id}_precision', class_metrics[class_id]["precision"], epoch)
                    writer.add_scalar(f'Epoch/class{class_id}_recall', class_metrics[class_id]["recall"], epoch)
                    writer.add_scalar(f'Epoch/class{class_id}_f1', class_metrics[class_id]["f1"], epoch)
            
            # Calcola anche la media complessiva
            overall_miou = sum([m["mIoU"] for m in class_metrics.values()]) / len(class_metrics)
            
            return {
                "per_class": class_metrics,
                "mIoU": overall_miou
            }

    def save(self, path):
        """Salva il modello in un file"""
        # Gestisce il salvataggio di modelli con DataParallel
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Modello salvato in {path}")
    
    def load(self, path):
        """Carica il modello da un file"""
        if os.path.exists(path):
            # Carica il modello con map_location per gestire GPU diverse
            state_dict = torch.load(path, map_location=self.device)
            
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
            print(f"Modello caricato da {path}")
            return True
        else:
            print(f"Attenzione: File del modello {path} non trovato")
            return False
