#branch alex
#dataset /home/alex23/.cache/kagglehub/datasets/omkargurav/face-mask-dataset/versions/1/data/without_mask
#Path to dataset: /home/alex23/.cache/kagglehub/datasets/anku5hk/5-faces-dataset/versions/1/Five_Faces
#Path to dataset: /home/alex23/.cache/kagglehub/datasets/freak2209/face-data/versions/1/Custom_Data/images/train

import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

RESULTS_FILE = "results.txt"

# LOG SYSTEM
def write_result(text: str):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# INIT MODELS
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

mtcnn = MTCNN(image_size=160, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)



# EXTRACT EMBEDDING
def get_embedding(image_path: str):
    try:
        img = Image.open(image_path).convert("RGB")  # fixes RGBA PNG
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        write_result(f"Error opening {image_path}: {e}")
        return None

    print(f"\nLoaded: {image_path}, size={img.size}")
    write_result(f"\nLoaded: {image_path}, size={img.size}")

    face = mtcnn(img)

    if face is None:
        print("⚠ No face found.")
        write_result("⚠ No face found.")
        return None

    face = face.to(device)

    with torch.no_grad():
        emb = facenet(face.unsqueeze(0))

    return emb.squeeze(0).cpu()


# BUILD DATABASE FROM persons
def build_database(root: str = "persons"):
    db = {}

    if not os.path.isdir(root):
        print(f"⚠ Folder '{root}' not found, creating empty one.")
        os.makedirs(root, exist_ok=True)
        return db

    for person_name in os.listdir(root):
        person_folder = os.path.join(root, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"\n🔵 Loading person: {person_name}")
        write_result(f"\nLoading person: {person_name}")

        person_embeddings = []

        for file in os.listdir(person_folder):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_folder, file)
            emb = get_embedding(img_path)

            if emb is not None:
                person_embeddings.append(emb)

        if person_embeddings:
            db[person_name] = torch.stack(person_embeddings, dim=0)
            print(f" → {person_name}: {len(person_embeddings)} embeddings saved.")
            write_result(f" → {person_name}: {len(person_embeddings)} embeddings saved.")

    print(f"\nDATABASE READY: {list(db.keys())}")
    write_result(f"DATABASE READY: {list(db.keys())}")
    return db


# FACE RECOGNITION
def recognize_face(embedding: torch.Tensor, database: dict):
    if not database:
        return None, float("inf")

    best_name = None
    best_dist = float("inf")

    for name, emb_list in database.items():
        distances = torch.norm(emb_list - embedding, dim=1)
        dist = distances.min().item()

        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name, best_dist


# SAVE OR MERGE UNKNOWN PERSON
def save_unknown_person(
    embedding: torch.Tensor,
    image_path: str,
    database: dict,
    threshold_unknown: float = 0.9,
):
    img = Image.open(image_path).convert("RGB")

    # Try to match unknown persons
    best_person = None
    best_dist = float("inf")

    for name, embeds in database.items():
        distances = torch.norm(embeds - embedding, dim=1)
        dist = distances.min().item()

        if dist < best_dist:
            best_dist = dist
            best_person = name

    # Merge with existing unknown
    if best_person is not None and best_person.startswith("unknown_") and best_dist < threshold_unknown:
        print(f"→ Same UNKNOWN person: {best_person} (dist={best_dist:.3f})")
        write_result(f"→ Same UNKNOWN person: {best_person} (dist={best_dist:.3f})")

        folder = os.path.join("persons", best_person)
        os.makedirs(folder, exist_ok=True)

        count = len([f for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        img.save(os.path.join(folder, f"{count + 1}.jpg"))

        database[best_person] = torch.cat([database[best_person], embedding.unsqueeze(0)], dim=0)

        return best_person

    # Create new unknown
    existing_unknowns = [name for name in database if name.startswith("unknown_")]
    next_id = len(existing_unknowns) + 1
    new_name = f"unknown_{next_id}"

    folder = os.path.join("persons", new_name)
    os.makedirs(folder, exist_ok=True)

    img.save(os.path.join(folder, "1.jpg"))

    database[new_name] = embedding.unsqueeze(0)

    print(f"→ New UNKNOWN person created: {new_name}")
    write_result(f"→ New UNKNOWN person created: {new_name}")

    return new_name


# PROCESS FACES FOLDER
def process_faces(
    database: dict,
    folder: str = "faces",
    known_threshold: float = 1.0,
    unknown_merge_threshold: float = 0.9,
):
    print("\n🔍 Checking faces in:", folder)
    write_result("\nChecking faces...")

    if not os.path.isdir(folder):
        print(f"⚠ Folder '{folder}' not found.")
        write_result(f"⚠ Folder '{folder}' not found.")
        return

    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder, file)

        print("\n===============================")
        print(f"Processing: {file}")
        write_result(f"\nProcessing: {file}")

        emb = get_embedding(img_path)

        if emb is None:
            print("- NO PERSON DETECTED")
            write_result("- NO PERSON DETECTED")
            continue

        name, dist = recognize_face(emb, database)

        # DB empty -> create unknown directly
        if name is None:
            print(f"+ UNKNOWN PERSON (no DB)")
            write_result(f"+ UNKNOWN PERSON (no DB)")
            save_unknown_person(emb, img_path, database, unknown_merge_threshold)
            continue


        # PERSON IDENTIFIED -> SAVE IMAGE & EMBEDDING
        if not name.startswith("unknown_") and dist < known_threshold:
            print(f"** PERSON IDENTIFIED: {name} (dist={dist:.3f})")
            write_result(f"** PERSON IDENTIFIED: {name} (dist={dist:.3f})")

            person_folder = os.path.join("persons", name)
            os.makedirs(person_folder, exist_ok=True)

            count = len([f for f in os.listdir(person_folder)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])

            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(person_folder, f"{count + 1}.jpg"))

            database[name] = torch.cat([database[name], emb.unsqueeze(0)], dim=0)

            continue

        # UNKNOWN PERSON
        print(f"+ UNKNOWN PERSON (best match={name}, dist={dist:.3f})")
        write_result(f"+ UNKNOWN PERSON (best match={name}, dist={dist:.3f})")

        save_unknown_person(emb, img_path, database, unknown_merge_threshold)


# MAIN
if __name__ == "__main__":
    open(RESULTS_FILE, "w", encoding="utf-8").close()

    print("Building database from 'persons/' ...")
    write_result("Starting run...\n")

    db = build_database("persons")

    print("\nProcessing faces from 'faces/' ...")
    process_faces(db, "faces", known_threshold=1.0, unknown_merge_threshold=0.9)

    print("\n✅ Done. Check results.txt for full log.")
    write_result("\nDone.")
