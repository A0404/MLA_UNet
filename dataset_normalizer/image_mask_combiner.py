import cv2
import numpy as np
from glob import glob
import os

# --- Configurations de Chemin ---
SOURCE_PATHS = [r"C:\Users\adrie\Documents\5A\MLA\bdd\isbi-datasets\data\images", r"C:\Users\adrie\Documents\5A\MLA\bdd\isbi-datasets\data\labels"] 
OUTPUT_PATH = r"C:\Users\adrie\Documents\5A\MLA\bdd\isbi-datasets\formed" 
SIZE = 572 

# --- Boucle de Traitement et d'Enregistrement ---
for source_folder in SOURCE_PATHS:
    
    category = os.path.basename(source_folder)
    output_folder = os.path.join(OUTPUT_PATH, category)
    
    # CRÉATION SIMPLIFIÉE : Crée le sous-dossier, ou ne fait rien s'il existe (exist_ok=True).
    os.makedirs(output_folder, exist_ok=True)
        
    all_files = sorted(glob(os.path.join(source_folder, "*")))

    current_mask_data = None
    last_image_filename = None 
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # --- Chargement et Vérification (Explicite) ---
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # Si le fichier est illisible ou corrompu, 'img' sera None.
        if img is None:
            continue

        # --- Prétraitement Uniforme ---
        
        # Redimensionnement
        img_resized = cv2.resize(img, (SIZE, SIZE)) 
        
        # Conversion en Grayscale
        if len(img_resized.shape) > 2: 
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) 
        else:
             img_gray = img_resized

        # Normalisation
        img_normalized = img_gray.astype(np.float32) / 255.0

        # --- Logique Masque vs. Image ---
        if "mask" in filename.lower(): 
            # MASQUE : Combinaison
            if current_mask_data is None:
                current_mask_data = img_normalized
            else:
                current_mask_data += img_normalized

        else:
            # IMAGE ORIGINALE : Sauvegarde de l'image et du masque précédent
            
            # Enregistrer le MASQUE COMBINÉ PRÉCÉDENT
            if current_mask_data is not None and last_image_filename is not None:
                
                # Binarisation finale 
                final_mask_normalized = np.where(current_mask_data > 0.5, 1.0, 0.0)
                
                base_name, _ = os.path.splitext(last_image_filename)
                mask_filename = f"{base_name}_combined_mask.png"
                mask_output_path = os.path.join(output_folder, mask_filename)
                
                # Reconvertir à [0, 255] et enregistrer
                mask_to_save = (final_mask_normalized * 255).astype(np.uint8)
                cv2.imwrite(mask_output_path, mask_to_save)
                
                current_mask_data = None
                # last_image_filename = None (Ligne supprimée car inutile avant la réaffectation)
                
            # Enregistrer l'IMAGE ORIGINALE TRAITÉE
            output_img_path = os.path.join(output_folder, filename)
            img_to_save = img_gray.astype(np.uint8) 
            cv2.imwrite(output_img_path, img_to_save)
            
            last_image_filename = filename # Affectation nécessaire
            
    # --- FIN DE BOUCLE : Gérer le dernier masque combiné s'il reste ---
    if current_mask_data is not None and last_image_filename is not None:
        final_mask_normalized = np.where(current_mask_data > 0.5, 1.0, 0.0)
        base_name, _ = os.path.splitext(last_image_filename)
        mask_filename = f"{base_name}_combined_mask.png"
        mask_output_path = os.path.join(output_folder, mask_filename)
        
        mask_to_save = (final_mask_normalized * 255).astype(np.uint8)
        cv2.imwrite(mask_output_path, mask_to_save)