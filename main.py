import argparse
import os

def main():
    # Analizza gli argomenti della riga di comando
    parser = argparse.ArgumentParser(description="Mask R-CNN per la segmentazione degli oggetti in mano nel dataset Hot3D")
    parser.add_argument("--mode", choices=["train", "evaluate", "inference"], default="train", help="Modalità di esecuzione")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Percorso del checkpoint da cui continuare (opzionale per train)")
    parser.add_argument("--epochs", type=int, default=None, 
                       help="Numero di epoche di addestramento")
    parser.add_argument("--image", help="Percorso dell'immagine per l'inferenza (solo per la modalità 'inference')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Soglia di confidenza per le predizioni (solo per la modalità 'inference')")
    parser.add_argument("--output", help="Percorso per salvare la visualizzazione dell'inferenza (solo per la modalità 'inference')")
    
    args = parser.parse_args()
    
    # Esegui in base alla modalità
    if args.mode == "train":
        print("Avvio dell'addestramento...")
        from train import main as train_main
        
        # Determina la modalità di addestramento in base alla presenza di checkpoint
        train_mode = "continue" if args.checkpoint else "train"
        
        # Gestisci gli argomenti di addestramento
        train_args = ["--mode", train_mode]
        
        if args.checkpoint:
            train_args.extend(["--checkpoint", args.checkpoint])
        
        if args.epochs:
            train_args.extend(["--epochs", str(args.epochs)])
        
        # Modifica sys.argv per passare gli argomenti a train_main
        import sys
        old_argv = sys.argv
        sys.argv = [old_argv[0]] + train_args
        
        try:
            train_main()
        finally:
            # Ripristina sys.argv
            sys.argv = old_argv
    
    elif args.mode == "evaluate":
        print("Avvio della valutazione...")
        from evaluate import main as evaluate_main
        evaluate_main()
    
    elif args.mode == "inference":
        if not args.image:
            print("Errore: Per la modalità 'inference', è necessario specificare un'immagine con '--image'")
            return
        
        print("Avvio dell'inferenza...")
        from inference import inference
        
        # Imposta il percorso di output predefinito se non specificato
        output_path = args.output
        if not output_path:
            from config import VISUALIZATIONS_DIR
            os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            output_path = os.path.join(VISUALIZATIONS_DIR, f"{base_name}_pred.png")
        
        # Esegui l'inferenza
        inference(
            image_path=args.image,
            threshold=args.threshold,
            output_path=output_path
        )

if __name__ == "__main__":
    main()