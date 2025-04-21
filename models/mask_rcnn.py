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
from config import TENSORBOARD_DIR, NUM_CLASSES, MODEL_SAVE_PATH, NUM_EPOCHS, LEARNING_RATE

class MaskRCNNModel:

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, backbone_name="resnext101_32x8d", 
                clip_grad_norm=0.0, optimizer_type='sgd'):
        """
        Inizializza il modello Mask R-CNN con backbone ResNeXt-101-FPN
        
        Args:
            num_classes: Numero di classi (background + classi di oggetti)
            pretrained: Se usare pesi preaddestrati
            backbone_name: Nome del backbone da usare (default: "resnext101_32x8d")
            clip_grad_norm: Valore per il clipping dei gradienti (0 per disattivare)
            optimizer_type: Tipo di ottimizzatore ('sgd' o 'adamw')
        """
        # Seleziona il dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilizzando dispositivo: {self.device}")
        
        # Salva i parametri di configurazione
        self.clip_grad_norm = clip_grad_norm
        self.optimizer_type = optimizer_type
        
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
        
        # Classificazione binaria (background + object_in_hand)
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

    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, model_save_path=MODEL_SAVE_PATH):
        """
        Addestra il modello Mask R-CNN
        
        Args:
            train_loader: DataLoader per i dati di training
            val_loader: DataLoader per i dati di validazione
            num_epochs: Numero di epoche di addestramento
            learning_rate: Learning rate iniziale
            model_save_path: Percorso dove salvare il modello
                
        Returns:
            Dictionary con storico delle metriche di addestramento
        """
        # Inizializza il writer per TensorBoard
        writer = SummaryWriter(TENSORBOARD_DIR)
        
        # Imposta l'ottimizzatore in base al tipo specificato
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0001)
            print(f"Utilizzo ottimizzatore AdamW con learning rate {learning_rate}")
        else:
            optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            print(f"Utilizzo ottimizzatore SGD con learning rate {learning_rate}")
        
        # Scheduler per learning rate decay - rimuovi parametro 'verbose' per eliminare il warning
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        
        # Dizionario per tracciare le metriche
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": []
        }
        
        # Loop di addestramento
        best_val_miou = 0.0
        
        # Imposta il modello in modalità training
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"\nEpoca {epoch+1}/{num_epochs}")
            
            # Training
            running_loss = 0.0
            epoch_loss = []
            
            # Progress bar per il training
            train_pbar = tqdm(train_loader, desc=f"Training [{epoch+1}/{num_epochs}]")
            
            for images, targets in train_pbar:
                # Sposta i dati sul device
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                # Verifica che ci siano bounding box validi
                valid_batch = True
                for t in targets:
                    if t["boxes"].numel() == 0:
                        valid_batch = False
                        break
                
                if not valid_batch:
                    continue
                
                # Azzera i gradienti
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    loss_dict = self.model(images, targets)
                    
                    # Verifica che loss_dict sia un dizionario
                    if not isinstance(loss_dict, dict):
                        print(f"Warning: loss_dict non è un dizionario ma {type(loss_dict)}")
                        continue
                    
                    # Calcola la loss totale
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Verifica se la loss è NaN o infinita
                    if not torch.isfinite(losses):
                        print(f"Warning: Loss è {losses.item()}, salto questo batch")
                        continue
                    
                    # Backward pass
                    losses.backward()
                    
                    # Applica clipping dei gradienti se specificato
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Aggiorna i pesi
                    optimizer.step()
                    
                    # Aggiorna la progress bar
                    running_loss += losses.item()
                    epoch_loss.append(losses.item())
                    train_pbar.set_postfix(loss=f"{running_loss/(train_pbar.n+1):.4f}")
                except Exception as e:
                    print(f"Errore durante il training: {str(e)}")
                    continue
            
            # Calcola la loss media per l'epoca
            avg_train_loss = sum(epoch_loss) / max(len(epoch_loss), 1)
            history["train_loss"].append(avg_train_loss)
            
            # Log su TensorBoard
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            
            # Imposta il modello in modalità valutazione
            self.model.eval()
            
            # Calcola le metriche sul validation set
            val_metrics = self.evaluate(val_loader, epoch=epoch, writer=writer)
            val_loss = val_metrics["loss"]
            val_miou = val_metrics["miou"]
            
            # Aggiorna lo scheduler
            lr_scheduler.step(val_loss)
            
            # Salva le metriche
            history["val_loss"].append(val_loss)
            history["val_miou"].append(val_miou)
            
            # Stampa le metriche
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
            
            # Salva il modello se è il migliore finora
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                self.save(model_save_path)
                print(f"Salvato miglior modello con mIoU: {val_miou:.4f}")
            
            # Ripristina modalità training per la prossima epoca
            self.model.train()
        
        # Salva il modello finale
        self.save(model_save_path)
        
        # Chiudi il writer
        writer.close()
        
        return history

    def evaluate(self, data_loader, epoch=None, writer=None):
        """
        Valuta il modello su dati di validazione/test calcolando mIoU
        
        Args:
            data_loader: DataLoader per i dati di validazione/test
            epoch: Numero dell'epoca corrente (per logging)
            writer: SummaryWriter per TensorBoard logging
                
        Returns:
            Dictionary con metriche di valutazione
        """
        self.model.eval()
        
        all_losses = []
        all_preds = []
        all_targets = []
        
        # Disabilita il calcolo dei gradienti
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Validazione"):
                # Sposta i dati sul device
                images = [image.to(self.device) for image in images]
                targets_device = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                # Verifica che ci siano bounding box validi
                valid_batch = True
                for t in targets_device:
                    if t["boxes"].numel() == 0:
                        valid_batch = False
                        break
                
                if not valid_batch:
                    continue
                
                try:
                    # In modalità eval, facciamo due passaggi:
                    # 1. Forward con targets per calcolare la loss
                    self.model.train()  # Temporaneamente in train mode per calcolare la loss
                    loss_dict = self.model(images, targets_device)
                    self.model.eval()   # Torna in eval mode
                    
                    # Verifica il tipo di output
                    if isinstance(loss_dict, dict):
                        # Loss dict è un dizionario come previsto
                        losses = sum(loss for loss in loss_dict.values())
                        all_losses.append(losses.item())
                    else:
                        print(f"Tipo di output non gestito durante il calcolo loss: {type(loss_dict)}")
                    
                    # 2. Forward senza targets per le predizioni
                    outputs = self.model(images)
                    
                    # Raccolta predizioni e target per il calcolo delle metriche
                    for output, target in zip(outputs, targets):
                        # Verifica che output sia un dizionario
                        if not isinstance(output, dict):
                            print(f"Errore: output non è un dizionario ma {type(output)}")
                            continue
                            
                        pred_boxes = output['boxes'].cpu()
                        pred_scores = output['scores'].cpu()
                        pred_labels = output['labels'].cpu()
                        pred_masks = output['masks'].cpu()
                        
                        # Usa solo predizioni con score > 0.5
                        keep = pred_scores > 0.5
                        if torch.any(keep):
                            pred_boxes = pred_boxes[keep]
                            pred_scores = pred_scores[keep]
                            pred_labels = pred_labels[keep]
                            pred_masks = pred_masks[keep]
                        
                            # Salva predizioni e ground truth per calcolare le metriche
                            all_preds.append({
                                'boxes': pred_boxes,
                                'scores': pred_scores,
                                'labels': pred_labels,
                                'masks': pred_masks
                            })
                            
                            all_targets.append({
                                'boxes': target['boxes'],
                                'labels': target['labels'],
                                'masks': target['masks']
                            })
                except Exception as e:
                    print(f"Errore durante la valutazione: {str(e)}")
                    continue
        
        # Calcola la loss media
        avg_loss = sum(all_losses) / max(len(all_losses), 1) if all_losses else 0.0
        
        # Calcola mIoU (Mean Intersection over Union)
        try:
            if all_preds and all_targets:
                miou = self.calculate_miou(all_preds, all_targets)
            else:
                print("Nessuna predizione o target valido per calcolare mIoU")
                miou = 0.0
        except Exception as e:
            print(f"Errore nel calcolo della mIoU: {str(e)}")
            miou = 0.0
        
        # Logging su TensorBoard se richiesto
        if writer is not None and epoch is not None:
            writer.add_scalar('Loss/val', avg_loss, epoch)
            writer.add_scalar('mIoU/val', miou, epoch)
        
        # Ritorna le metriche
        metrics = {
            "loss": avg_loss,
            "miou": miou
        }
        
        print(f"Validation: Loss={avg_loss:.4f}, mIoU={miou:.4f}")
        
        return metrics

    def calculate_miou(self, all_preds, all_targets, iou_threshold=0.5, mask_binarize_threshold=0.5):
        """
        Calcola la Mean Intersection over Union (mIoU) per le maschere.
        Questa versione cerca la *migliore corrispondenza* per ogni maschera GT.

        Args:
            all_preds: Lista di dizionari con le predizioni filtrate per score.
                       Ogni dizionario contiene 'masks': Tensor (N_preds, 1, H, W) float [0,1]
            all_targets: Lista di dizionari con i ground truth.
                         Ogni dizionario contiene 'masks': Tensor (N_gt, H, W) o (N_gt, 1, H, W) uint8/bool
            iou_threshold: Soglia IoU per considerare una corrispondenza (non usata per mIoU puro).
            mask_binarize_threshold: Soglia per convertire le maschere predette (float) in binarie.

        Returns:
            mIoU: Mean Intersection over Union media su tutte le maschere GT.
        """
        all_ious_per_gt = [] # Lista per raccogliere il miglior IoU per ogni maschera GT

        num_total_gt_masks = 0
        num_preds_considered = 0

        for img_idx, (pred_dict, target_dict) in enumerate(zip(all_preds, all_targets)):
            # Estrai maschere predette e GT, gestendo casi vuoti
            pred_masks_tensor = pred_dict.get('masks', torch.empty(0, 1, 0, 0)) # (N, 1, H, W)
            gt_masks_tensor = target_dict.get('masks', torch.empty(0, 0, 0))     # (N, H, W)

            # --- Conversione a Numpy e Binarizzazione ---
            # Predizioni: Binarizza e converti in numpy. MANTIENE la dimensione del canale se presente.
            pred_masks_np = (pred_masks_tensor.cpu().numpy() > mask_binarize_threshold).astype(np.uint8)
            # -> pred_masks_np avrà shape (N_preds, 1, H, W) se l'input era così

            # Ground Truth: Converti in numpy. Gestisci possibile dimensione canale extra.
            gt_masks_np_raw = gt_masks_tensor.cpu().numpy().astype(np.uint8)
            # Se le maschere GT hanno una dimensione canale extra (es. shape (N_gt, 1, H, W)), rimuovila.
            if gt_masks_np_raw.ndim == 4 and gt_masks_np_raw.shape[1] == 1:
                gt_masks_np = gt_masks_np_raw[:, 0, :, :] # -> shape (N_gt, H, W)
            elif gt_masks_np_raw.ndim == 3:
                gt_masks_np = gt_masks_np_raw # -> shape (N_gt, H, W) già corretta
            else:
                # Gestisci forme inattese o vuote
                if gt_masks_np_raw.numel() == 0:
                    gt_masks_np = np.empty((0, 0, 0), dtype=np.uint8)
                else:
                    print(f"Warning: Forma maschera GT inattesa: {gt_masks_np_raw.shape}")
                    gt_masks_np = gt_masks_np_raw # Prova a procedere, ma potrebbe fallire


            num_preds = pred_masks_np.shape[0]
            num_gts = gt_masks_np.shape[0]
            num_total_gt_masks += num_gts
            num_preds_considered += num_preds

            if num_gts == 0:
                continue # Non ci sono maschere GT in questa immagine, passa alla successiva

            if num_preds == 0:
                # Se non ci sono predizioni, tutte le GT hanno IoU=0
                all_ious_per_gt.extend([0.0] * num_gts)
                continue

            # --- Controllo Dimensioni H, W (dopo aver gestito i canali) ---
            # Assicurati che altezza e larghezza corrispondano tra pred e GT
            # Fallback: Prova a ridimensionare se necessario (ma è meglio risolvere nel dataset)
            pred_h, pred_w = pred_masks_np.shape[2:] # Ottiene H, W da (N, 1, H, W)
            gt_h, gt_w = gt_masks_np.shape[1:]     # Ottiene H, W da (N, H, W)

            if pred_h != gt_h or pred_w != gt_w:
                 print(f"ATTENZIONE (img {img_idx}): Dimensioni H,W non corrispondono! Pred: ({pred_h},{pred_w}), GT: ({gt_h},{gt_w}). Impossibile calcolare IoU.")
                 # Se le dimensioni non combaciano, l'IoU per tutte le GT di questa immagine è 0
                 all_ious_per_gt.extend([0.0] * num_gts)
                 # Puoi provare a usare skimage.transform.resize qui come fallback, ma è rischioso:
                 # try:
                 #     from skimage.transform import resize
                 #     print(f"Warning (img {img_idx}): Tentativo di ridimensionamento GT a ({pred_h},{pred_w})...")
                 #     gt_masks_np_resized = np.array([resize(gt, (pred_h, pred_w), order=0, preserve_range=True, anti_aliasing=False) for gt in gt_masks_np]).astype(np.uint8)
                 #     gt_masks_np = gt_masks_np_resized
                 #     print("Ridimensionamento completato.")
                 # except ImportError:
                 #     print("Skipping resize: skimage non trovata.")
                 #     all_ious_per_gt.extend([0.0] * num_gts)
                 #     continue
                 # except Exception as e:
                 #     print(f"Errore durante il resize: {e}")
                 #     all_ious_per_gt.extend([0.0] * num_gts)
                 #     continue
                 continue # Salta al prossimo immagine se le dimensioni non combaciano

            # Calcola matrice IoU: righe = GT, colonne = Predizioni
            iou_matrix = np.zeros((num_gts, num_preds))

            for i in range(num_gts):
                gt_mask = gt_masks_np[i] # Shape (H, W)
                if gt_mask.sum() == 0: continue # Salta maschere GT vuote

                for j in range(num_preds):
                    # Estrai maschera predetta j-esima, ha shape (1, H, W)
                    pred_mask_with_channel = pred_masks_np[j]

                    # *** LA CORREZIONE CHIAVE È QUI ***
                    # Rimuovi la dimensione del canale per ottenere shape (H, W)
                    if pred_mask_with_channel.shape[0] == 1:
                         pred_mask = pred_mask_with_channel[0]
                    else:
                         # Questo non dovrebbe succedere se l'input è standard
                         print(f"Warning: Pred mask {j} ha shape inattesa {pred_mask_with_channel.shape}")
                         pred_mask = pred_mask_with_channel # Prova ad usarla, potrebbe fallire dopo
                         # Verifica se ha comunque le dimensioni H,W corrette
                         if pred_mask.shape != (pred_h, pred_w):
                             print(f"Skipping pred mask {j} due to unexpected shape post-indexing.")
                             continue

                    if pred_mask.sum() == 0: continue # Salta maschere predette vuote

                    # Ora pred_mask e gt_mask dovrebbero entrambe avere shape (H, W)
                    # Il controllo di shape qui sotto non dovrebbe più fallire se H,W combaciano
                    # if pred_mask.shape != gt_mask.shape: # Check ridondante se H,W globali combaciano
                    #     print(f"Shape mismatch INTERNO! GT {gt_mask.shape}, Pred {pred_mask.shape}")
                    #     continue

                    # Calcola Intersezione e Unione
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum()

                    if union > 0:
                        iou = intersection / union
                        iou_matrix[i, j] = iou
                    # else: iou è 0 (già inizializzato)

            # Trova il miglior IoU per ogni maschera Ground Truth
            if num_preds > 0:
                max_iou_per_gt = iou_matrix.max(axis=1) # Trova il max IoU per ogni riga (GT)
            else:
                # Già gestito all'inizio del loop per immagine (num_preds == 0)
                # max_iou_per_gt = np.zeros(num_gts) # Se non ci sono predizioni, IoU è 0
                pass # max_iou_per_gt non viene usato se num_preds è 0

            # Aggiungi i migliori IoU per le GT di questa immagine alla lista totale
            # (Se num_preds era 0, questo estende con zeri)
            all_ious_per_gt.extend(max_iou_per_gt.tolist())


        # --- Calcolo Finale e Statistiche ---
        if not all_ious_per_gt:
             # Questo succede se non c'erano maschere GT in nessuna immagine
             print("Warning calculate_miou: Nessun IoU calcolato (nessuna maschera GT trovata nel set di valutazione).")
             return 0.0

        # Calcola la mIoU media su tutte le maschere GT considerate
        miou = np.mean(all_ious_per_gt)

        # Stampa statistiche IoU dettagliate
        if print_debug:
            ious_np = np.array(all_ious_per_gt)
            print(f"Statistiche IoU per GT (calcolate su {len(ious_np)} maschere GT totali):")
            print(f"  Min IoU: {ious_np.min():.4f}")
            print(f"  Max IoU: {ious_np.max():.4f}")
            print(f"  Median IoU: {np.median(ious_np):.4f}")
            # Conta quante GT hanno trovato una corrispondenza sopra certe soglie
            print(f"  GT con IoU > 0.1: {(ious_np > 0.1).sum()}/{len(ious_np)} ({(ious_np > 0.1).mean()*100:.2f}%)")
            print(f"  GT con IoU > 0.25: {(ious_np > 0.25).sum()}/{len(ious_np)} ({(ious_np > 0.25).mean()*100:.2f}%)")
            print(f"  GT con IoU > 0.5: {(ious_np > 0.5).sum()}/{len(ious_np)} ({(ious_np > 0.5).mean()*100:.2f}%)")
            print(f"  GT con IoU > 0.75: {(ious_np > 0.75).sum()}/{len(ious_np)} ({(ious_np > 0.75).mean()*100:.2f}%)")
            print(f"mIoU Finale (media su GT): {miou:.4f}")
            # print(f"Numero totale maschere GT: {num_total_gt_masks}") # Uguale a len(ious_np)
            # print(f"Numero totale predizioni considerate (post-filtro score): {num_preds_considered}") # Utile per vedere quante predizioni vengono fatte

        return float(miou)

    def predict(self, image, score_threshold=0.5):
        """
        Esegue la predizione su una singola immagine
        
        Args:
            image: Immagine di input (numpy array o tensore)
            score_threshold: Soglia di confidenza per le predizioni
            
        Returns:
            dict: Dizionario con maschere, scatole, punteggi e etichette sopra la soglia
        """
        self.model.eval()
        
        # Converti l'immagine se necessario
        if isinstance(image, np.ndarray):
            # Verifica che l'immagine sia nel formato corretto (H, W, C)
            if image.ndim == 2:
                # Immagine grayscale a singolo canale, aggiungi dimensione canale
                image = np.expand_dims(image, axis=2)
                image = np.repeat(image, 3, axis=2)  # Converti in RGB replicando il canale
            elif image.shape[2] == 4:
                # Immagine RGBA, elimina canale alpha
                image = image[:, :, :3]
            
            # Normalizza l'immagine se non è già normalizzata
            if image.dtype == np.uint8:
                # Converti da numpy a tensor e normalizza
                image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
                # Normalizza con mean e std di ImageNet
                image = torchvision.transforms.functional.normalize(
                    image, 
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
        elif isinstance(image, torch.Tensor):
            # Se è già un tensore, assicurati che sia nel formato corretto (C, H, W)
            if image.dim() == 2:
                # Immagine grayscale a singolo canale, aggiungi dimensione canale
                image = image.unsqueeze(0)
                image = image.repeat(3, 1, 1)  # Converti in RGB replicando il canale
            elif image.dim() == 3 and image.shape[0] == 4:
                # Immagine RGBA, elimina canale alpha
                image = image[:3, :, :]
            
            # Normalizza se necessario (se i valori sono tra 0 e 255)
            if image.max() > 1.0:
                image = image / 255.0
                # Normalizza con mean e std di ImageNet
                image = torchvision.transforms.functional.normalize(
                    image, 
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
        else:
            raise TypeError(f"Formato immagine non supportato: {type(image)}")
        
        # Aggiungi dimensione batch
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Sposta sul device
        image = image.to(self.device)
        
        try:
            # Disabilita il calcolo dei gradienti
            with torch.no_grad():
                # Esegui la predizione
                predictions = self.model(image)
            
            # Estrai i risultati (prendi il primo elemento perché abbiamo un batch di 1)
            if len(predictions) == 0:
                # Nessuna predizione
                return {
                    'masks': np.array([]),
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([])
                }
            
            pred = predictions[0]
            
            # Sposta i tensori su CPU e converti in numpy
            masks = pred['masks'].cpu().numpy()
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Filtra per score threshold
            keep = scores > score_threshold
            masks = masks[keep]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Converti le maschere in formato binario
            # Le maschere sono in formato (N, 1, H, W)
            if masks.size > 0:
                binary_masks = (masks > 0.5).astype(np.uint8)
                # Rimuovi la dimensione del canale se presente
                if binary_masks.shape[1] == 1:
                    binary_masks = np.squeeze(binary_masks, axis=1)
            else:
                binary_masks = np.array([])
            
            return {
                'masks': binary_masks,
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Errore durante la predizione: {str(e)}")
            # Ritorna un dizionario vuoto in caso di errore
            return {
                'masks': np.array([]),
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }

    def save(self, path):
        """
        Salva il modello in un file
        
        Args:
            path: Percorso dove salvare il modello
        """
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Gestisce il salvataggio di modelli con DataParallel
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Modello salvato in {path}")

    def load(self, path):
        """Carica il modello da un file"""
        if not os.path.exists(path):
            print(f"Attenzione: File del modello {path} non trovato")
            return False
        
        try:
            # Aggiungi weights_only=True per risolvere il warning
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
            print(f"Modello caricato da {path}")
            return True
        except Exception as e:
            print(f"Errore durante il caricamento del modello: {str(e)}")
            return False