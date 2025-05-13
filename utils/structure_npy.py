import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Analizza un file .npy contenente una lista di dizionari")
    parser.add_argument("--file", required=True, help="Percorso al file .npy da analizzare")
    args = parser.parse_args()

    # Carica il file
    array = np.load(args.file, allow_pickle=True)

    # Intestazione file
    print(f"\n📂 File: {os.path.basename(args.file)}")
    print(f"🧩 Dimensioni (ndim): {array.ndim}")
    print(f"📐 Forma (shape): {array.shape}")
    print(f"#️⃣ Numero di elementi: {array.size}")
    print(f"🔤 Tipo di dati: {array.dtype}")

    # Mostra anteprima se possibile
    print("\n👀 Anteprima dei primi 5 elementi (se applicabile):")
    preview = array[:5] if array.ndim > 0 else [array]
    for i, item in enumerate(preview):
        print(f"\n🗂️  Elemento {i+1}:")
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                print(f"  🔑 {key}: array con shape {value.shape} e dtype {value.dtype}")
            elif isinstance(value, list) and all(isinstance(v, np.ndarray) for v in value):
                print(f"  🔑 {key}: lista di {len(value)} array")
            else:
                print(f"  🔑 {key}: {value}")

if __name__ == "__main__":
    main()
