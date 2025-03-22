import os

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from model_classifier.model import ResNet50

load_dotenv()

client = MongoClient(os.getenv("URL_MONGODB"))
db_name = "IntroAI"
collection_name = "clothes_imgs"

db = client[db_name]
collection = db[collection_name]

categories = [
    "bag",
    "dress",
    "flats",
    "hat",
    "heels",
    "jacket",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "sneakers",
    "tshirt",
]


def top_k_cosine_indices(vector, cursor, k=5):
    # Chuyển đổi thành numpy array
    vector = np.array(vector)
    list_of_vectors = [doc["feature"] for doc in cursor]
    names = [doc["name"] for doc in cursor]
    matrix = np.array(list_of_vectors)  # (N, 2048)

    # Tính cosine similarity
    dot_product = np.dot(matrix, vector)  # (N,)
    norm_matrix = np.linalg.norm(matrix, axis=1)  # (N,)
    norm_vector = np.linalg.norm(vector)  # (scalar)

    cosine_similarities = dot_product / (norm_matrix * norm_vector)  # (N,)

    # Lấy top k indices có cosine similarity cao nhất
    top_k_indices = np.argsort(cosine_similarities)[-k:][::-1]

    return [names[i] for i in top_k_indices]


root_ckpt = r"trained_models/best_resnet50.pt"
checkpoint = torch.load(root_ckpt)
model = ResNet50(13)
model.load_state_dict(checkpoint["model"])


def get_feature(path, size_img=224):
    model.eval()

    img = Image.open(path).convert("RGB").copy()

    transform_test = Compose(
        [
            Resize((size_img, size_img)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = transform_test(img)
    img = img.unsqueeze(0)
    img = img.float()
    with torch.no_grad():
        probs, vector = model(img)
    output = categories[torch.argmax(probs)]
    return output, vector.tolist()[0]


def get_img_recommend(path):
    label, query_embedding = get_feature(path)

    cursor = list(collection.find({"label": label}))

    fol_img = os.path.join("./datasets_infer/train", str(label))
    img_paths = []
    results = top_k_cosine_indices(query_embedding, cursor)
    for r in results:
        img_path = os.path.join(fol_img, r)
        img_paths.append(img_path)
    return label, img_paths
