import os
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm
import random
import time
import argparse
from config import *
from models.mask_rcnn import MaskRCNNModel

def get_class_names(num_classes_from_args):
    """
    Ottiene la lista dei nomi delle classi per l'inferenza.
    Restituisce una lista con 'background' all'indice 0.

    Args:
        num_classes_from_args (int): Numero di classi totali (incl. background) dal modello/argomenti.

    Returns:
        list: Lista dei nomi delle classi, con 'background' all'indice 0.
    """
    class_names = None
    expected_foreground_classes = num_classes_from_args - 1

    # 1. Prova con CLASS_NAMES da config.py
    if 'CLASS_NAMES' in globals():
        config_class_names = globals()['CLASS_NAMES']
        if isinstance(config_class_names, (list, tuple)):
            print("INFO: Tentativo di usare CLASS_NAMES da config.py.")
            # Assumiamo che config.CLASS_NAMES includa 'background' o meno.
            # La cosa più sicura è controllare la lunghezza rispetto alle classi foreground.
            if len(config_class_names) == expected_foreground_classes:
                 # config.py contiene solo i nomi foreground
                 print(f"INFO: Usati {len(config_class_names)} nomi foreground da config.py.")
                 class_names = ["background"] + list(config_class_names)
            elif len(config_class_names) == num_classes_from_args:
                 # config.py include già 'background'
                 print(f"INFO: Usati {len(config_class_names)} nomi (incluso background) da config.py.")
                 class_names = list(config_class_names)
            else:
                print(f"ATTENZIONE: CLASS_NAMES in config.py ha {len(config_class_names)} elementi, ma attesi {expected_foreground_classes} (foreground) o {num_classes_from_args} (totali). Ignorati.")
        else:
             print("ATTENZIONE: CLASS_NAMES trovato in config.py ma non è una lista/tupla. Ignorato.")

    # 2. Fallback: Genera nomi di default
    if class_names is None:
        print(f"INFO: Generazione nomi di classe di default ('Obj 1', ..., 'Obj {expected_foreground_classes}').")
        foreground_names = [f"Obj {i}" for i in range(1, num_classes_from_args)]
        class_names = ["background"] + foreground_names # Aggiunge background

    # Controllo finale di sicurezza sulla lunghezza
    if len(class_names) != num_classes_from_args:
         print(f"ERRORE INTERNO: Lunghezza finale class_names ({len(class_names)}) != num_classes ({num_classes_from_args}). Rigenero.")
         foreground_names = [f"Obj {i}" for i in range(1, num_classes_from_args)]
         class_names = ["background"] + foreground_names

    # Assicurati che il primo elemento sia 'background' (case-insensitive per robustezza)
    if not class_names[0].lower() == "background":
         print(f"ATTENZIONE: Il primo nome di classe '{class_names[0]}' non è 'background'. Potrebbe causare problemi con l'indicizzazione.")
         # Si potrebbe forzare: class_names[0] = "background"

    print(f"Nomi classi finali (usati per l'output): {class_names}")
    return class_names

def get_colors(num_classes):
    """Genera colori casuali distinti per ogni classe (escluso background)."""
    # Usa RANDOM_SEED da config se importato, altrimenti un default
    seed = globals().get('RANDOM_SEED', 42)
    random.seed(seed)
    colors = [(0, 0, 0)] # Background nero
    for i in range(1, num_classes):
        # Genera colori più brillanti
        colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
    return colors

def draw_predictions(image, predictions, class_names, colors, threshold):
    """
    Disegna maschere, bounding box, etichette e score sull'immagine.
    Funzione robusta per gestire i bordi e possibili errori.

    Args:
        image (np.ndarray): Immagine originale (BGR).
        predictions (dict): Output da model.predict(). Contiene 'masks', 'boxes', 'scores', 'labels'.
        class_names (list): Lista nomi classi (indice 0 = background).
        colors (list): Lista tuple colore BGR (indice 0 = background).
        threshold (float): Soglia confidenza usata per la predizione.

    Returns:
        np.ndarray: Immagine con le annotazioni.
    """
    output_image = image.copy()
    masks = predictions.get('masks')     # Shape (N, H, W), uint8 o None
    boxes = predictions.get('boxes')     # Shape (N, 4), float32 o None
    scores = predictions.get('scores')   # Shape (N,), float32 o None
    labels = predictions.get('labels')   # Shape (N,), int64 o None

    # Verifica che tutte le predizioni necessarie siano presenti e non vuote
    if masks is None or boxes is None or scores is None or labels is None or len(scores) == 0:
        # print("Nessuna predizione valida trovata da disegnare.")
        return output_image # Restituisce l'immagine originale se non c'è nulla

    overlay = output_image.copy()
    # Usa MASK_ALPHA da config se esiste, altrimenti default
    alpha = globals().get('MASK_ALPHA', 0.5)
    num_predictions = len(scores)

    for i in range(num_predictions):
        # Solo istanze sopra la soglia (anche se predict dovrebbe già filtrare)
        if scores[i] < threshold:
            continue

        # --- Estrazione Dati ---
        box = boxes[i].astype(np.int32)
        label_idx = labels[i]
        score = scores[i]
        mask = masks[i] # Shape (H, W)

        # --- Validazione Indice Etichetta ---
        if label_idx <= 0 or label_idx >= len(class_names):
            print(f"  ATTENZIONE: Indice etichetta non valido {label_idx}. Max: {len(class_names)-1}. Uso 'Unknown'.")
            class_name = "Unknown"
            # Usa un colore di fallback (es. bianco) se l'indice è fuori range anche per i colori
            color = colors[0] if label_idx >= len(colors) else colors[label_idx] if label_idx >=0 else (255,255,255)
            if label_idx >= len(colors) or label_idx < 0: color = (255, 255, 255)

        else:
            class_name = class_names[label_idx]
            color = colors[label_idx]

        # --- Disegno Maschera (con controllo shape) ---
        try:
            if mask.shape == overlay.shape[:2]: # Controlla H, W
                 overlay[mask > 0] = color
            else:
                 print(f"  ATTENZIONE: Shape maschera {mask.shape} non corrisponde a overlay {overlay.shape[:2]}. Maschera non disegnata.")
                 # Potresti tentare un resize qui, ma è meglio che le predizioni siano corrette
                 # mask_resized = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                 # overlay[mask_resized > 0] = color
        except IndexError:
             print(f"  ATTENZIONE: Errore Indice nel disegnare maschera {i}.")
             continue # Salta questa istanza

        # --- Disegno Bounding Box (con clipping ai bordi) ---
        x1, y1, x2, y2 = box
        h, w = output_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x1 >= x2 or y1 >= y2: # Salta box non valide/degenerate
            print(f"  ATTENZIONE: Box {i} non valida dopo clipping: ({x1},{y1},{x2},{y2}).")
            continue
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2) # Spessore 2

        # --- Disegno Testo (Etichetta + Score, con gestione bordi) ---
        text = f"{class_name}: {score:.2f}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, font_thickness)

        # Posiziona sopra la box per default
        text_x = x1 + 2
        text_y = y1 - 5 # Posizione della baseline del testo

        # Se troppo vicino al bordo superiore, sposta sotto
        if text_y - text_height - baseline < 0:
            text_y = y2 + text_height + 5 # Sotto la box

        # Se esce dal bordo inferiore (improbabile se spostato sotto), riporta dentro
        if text_y > h:
             text_y = y1 + text_height + 5 # Dentro la box, in alto

        # Sfondo per leggibilità
        cv2.rectangle(output_image,
                      (text_x, text_y - text_height - baseline), # Top-left del background
                      (text_x + text_width, text_y + baseline),  # Bottom-right del background
                      color, -1) # Riempito
        # Testo (nero per contrasto)
        cv2.putText(output_image, text, (text_x, text_y),
                    font_face, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)

    # Applica l'overlay delle maschere con trasparenza all'immagine finale
    cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

    return output_image

# --- Funzione Principale per l'Inferenza ---

def run_inference(args):
    """
    Esegue l'inferenza usando il modello Mask R-CNN specificato su un'immagine
    o una directory di immagini.

    Args:
        args: Oggetto (simile ad argparse) contenente gli attributi necessari:
              - model (str): Percorso al file del modello .pth.
              - input (str): Percorso all'immagine o directory di input.
              - output (str): Directory dove salvare i risultati.
              - threshold (float): Soglia di confidenza per le predizioni.
              - num_classes (int): Numero totale di classi (incl. background).
              - backbone (str): Nome del backbone usato (es. 'resnext101_32x8d').
              - show (bool): Se mostrare l'immagine risultante (solo per input singolo).
              - class_names (str or None): (Opzionale) Stringa nomi classi separati da virgola.
                                            Se non fornito o None, verranno usati quelli
                                            da config.py o generati di default.
    """
    start_time = time.time()
    print("\n--- Inizio Inferenza ---")
    print(f"Modello: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Soglia confidenza: {args.threshold}")

    num_classes_to_use = args.num_classes
    print(f"Numero Classi (incl. background): {num_classes_to_use}")
    print(f"Backbone: {args.backbone}")
    if hasattr(args, 'class_names') and args.class_names: # Stampa solo se l'attributo esiste ed è valorizzato
        print(f"Nomi classi forniti esternamente: {args.class_names}")
    if args.show:
        print("Modalità visualizzazione: Attiva (solo per singola immagine)")

    # --- 1. Preparazione Modello ---
    print("Inizializzazione struttura modello...")
    try:
        # Passa solo num_classes e backbone, caricheremo i pesi dopo
        model_wrapper = MaskRCNNModel(
            num_classes=num_classes_to_use,
            pretrained=False, # Non usiamo i pesi preaddestrati torchvision qui
            backbone_name=args.backbone
        )
    except Exception as e:
         print(f"ERRORE durante l'inizializzazione di MaskRCNNModel: {e}")
         print("Controlla che num_classes e backbone_name siano corretti.")
         return # Esce se non può creare il modello

    # Carica i pesi addestrati dal percorso specificato
    print(f"Caricamento pesi da: {args.model}")
    if not model_wrapper.load(args.model):
        print(f"ERRORE: Impossibile caricare i pesi del modello da {args.model}. Uscita.")
        return

    # Imposta il modello in modalità valutazione (importante!)
    model_wrapper.model.eval()
    print("Modello caricato correttamente e impostato in modalità valutazione.")

    # --- 2. Preparazione Nomi Classi e Colori ---
    # Cerca l'attributo 'class_names' in args, altrimenti passa None
    class_names_arg = getattr(args, 'class_names', None)
    class_names = get_class_names(num_classes_to_use) # Non passiamo più class_names_arg qui
                                                     # get_class_names usa globals() per config
    colors = get_colors(num_classes_to_use)

    # --- 3. Creazione Directory Output ---
    try:
        os.makedirs(args.output, exist_ok=True)
        print(f"I risultati verranno salvati in: {args.output}")
    except OSError as e:
        print(f"ERRORE: Impossibile creare la directory di output '{args.output}': {e}")
        return

    # --- 4. Identificazione Input (File o Directory) ---
    image_paths = []
    process_multiple = False
    if os.path.isdir(args.input):
        print(f"Input è una directory. Ricerca immagini...")
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        for ext in supported_extensions:
            # Cerca ricorsivamente se necessario? Per ora no.
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        if not image_paths:
            print(f"ATTENZIONE: Nessuna immagine con estensioni supportate trovata in: {args.input}")
            return
        image_paths.sort() # Ordina per un output più prevedibile
        print(f"Trovate {len(image_paths)} immagini.")
        process_multiple = True
    elif os.path.isfile(args.input):
        print(f"Input è un singolo file: {args.input}")
        image_paths.append(args.input)
        process_multiple = False
    else:
        print(f"ERRORE: Percorso di input '{args.input}' non è un file o una directory valida.")
        return

    # --- 5. Loop di Inferenza sulle Immagini ---
    processed_count = 0
    errors_count = 0
    total_instances_detected = 0

    # Disabilita tqdm se si processa una sola immagine
    iterable = tqdm(image_paths, desc="Inferenza", disable=not process_multiple)

    for img_path in iterable:
        base_filename = os.path.basename(img_path)
        # Aggiungi un suffisso chiaro al nome del file di output
        output_filename = f"{os.path.splitext(base_filename)[0]}_inference.png" # Usa PNG per output lossless
        output_path = os.path.join(args.output, output_filename)

        if not process_multiple: # Log aggiuntivo per singola immagine
             print(f"\nProcessando: {base_filename}")

        # Carica immagine (BGR per OpenCV)
        try:
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                tqdm.write(f"ATTENZIONE: Impossibile caricare l'immagine {img_path}. Saltata.")
                errors_count += 1
                continue
        except Exception as e:
            tqdm.write(f"ERRORE durante il caricamento di {img_path}: {e}. Saltata.")
            errors_count += 1
            continue

        # Esegui predizione (MaskRCNNModel.predict gestisce la conversione/normalizzazione)
        try:
            prediction_start_time = time.time()
            # Passa l'immagine BGR, predict dovrebbe gestirla
            predictions = model_wrapper.predict(image_bgr, score_threshold=args.threshold)
            prediction_time = time.time() - prediction_start_time
            num_detected = len(predictions.get('scores', [])) # Usa get per sicurezza
            total_instances_detected += num_detected
            if not process_multiple:
                 print(f"  Predizione completata in {prediction_time:.3f}s. Rilevate {num_detected} istanze.")

        except Exception as e:
             tqdm.write(f"ERRORE durante la predizione su {img_path}: {e}. Saltata.")
             import traceback
             tqdm.write(traceback.format_exc()) # Stampa traceback per debug
             errors_count += 1
             continue

        # Disegna i risultati sull'immagine BGR
        try:
            drawing_start_time = time.time()
            output_image = draw_predictions(image_bgr, predictions, class_names, colors, args.threshold)
            drawing_time = time.time() - drawing_start_time
            if not process_multiple:
                print(f"  Disegno completato in {drawing_time:.3f}s.")

        except Exception as e:
             tqdm.write(f"ERRORE durante il disegno delle predizioni su {img_path}: {e}. Saltata.")
             errors_count += 1
             continue

        # Salva l'immagine risultante (PNG)
        try:
            success = cv2.imwrite(output_path, output_image)
            if not success:
                 raise IOError("imwrite ha restituito False")
            processed_count += 1
        except Exception as e:
            tqdm.write(f"ERRORE durante il salvataggio di {output_path}: {e}. Saltata.")
            errors_count += 1
            continue

        # Mostra l'immagine se richiesto (solo per singola immagine)
        if args.show and not process_multiple:
            print("Visualizzazione immagine... Premere un tasto qualsiasi sulla finestra per chiudere.")
            try:
                # Ridimensiona finestra se immagine troppo grande (opzionale)
                max_display_h = 800
                h_img, w_img = output_image.shape[:2]
                if h_img > max_display_h:
                    scale_factor = max_display_h / float(h_img)
                    display_width = int(w_img * scale_factor)
                    display_height = max_display_h
                    display_img = cv2.resize(output_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
                else:
                    display_img = output_image

                cv2.imshow(f"Inference Result - {base_filename}", display_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Errore durante la visualizzazione dell'immagine: {e}")

    # --- 6. Riepilogo Finale ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- Inferenza Completata ---")
    print(f"Immagini totali trovate: {len(image_paths)}")
    print(f"Immagini processate con successo: {processed_count}")
    if errors_count > 0:
        print(f"Errori / Immagini saltate: {errors_count}")
    print(f"Istanze totali rilevate (sopra soglia {args.threshold}): {total_instances_detected}")
    print(f"Risultati (immagini annotate) salvati in: {args.output}")
    print(f"Tempo totale impiegato: {total_time:.2f} secondi")
    if processed_count > 0:
         avg_time = total_time / processed_count
         print(f"Tempo medio per immagine (I/O, pred, draw): {avg_time:.3f} secondi ({1/avg_time:.2f} FPS)")
    print("--------------------------\n")


# --- Blocco per test standalone (senza logica HOT3D) ---
if __name__ == '__main__':
    print("Esecuzione di inference.py come script standalone per test.")

    # Definisci un parser di base per testare questa funzione
    parser = argparse.ArgumentParser(description="Test Script per run_inference")

    # Argomenti richiesti
    parser.add_argument('--model', type=str, required=True, help='Percorso al file del modello .pth')
    parser.add_argument('--input', type=str, required=True, help='Percorso all\'immagine o directory di immagini di input')
    parser.add_argument('--output', type=str, default=RESULTS_DIR, help=f"Directory di output per i risultati (default: {RESULTS_DIR})")
    parser.add_argument('--threshold', type=float, default=0.5, help='Soglia di confidenza per le predizioni (default: 0.5)')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help=f"Numero di classi (incl. background) (default: {NUM_CLASSES})")
    parser.add_argument('--backbone', type=str, default="resnext101_32x8d", choices=["resnext101_32x8d", "resnet50"], help='Tipo di backbone utilizzato nel modello (default: resnext101_32x8d)')
    parser.add_argument('--show', action='store_true', help='Mostra le immagini elaborate (solo se l\'input è un singolo file)')

    test_args = parser.parse_args()

    if not hasattr(test_args, 'class_names'):
         test_args.class_names = None 

    run_inference(test_args)