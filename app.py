import os

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

from model import ResNet50

load_dotenv()
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

root_ckpt = r"trained_model/best_resnet50.pt"
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


client = MongoClient(os.getenv("URL_MONGODB"))
db_name = "IntroAI"
collection_name = "clothes_imgs"

db = client[db_name]
collection = db[collection_name]

num_candidates = 70
label, query_embedding = get_feature(
    r"D:\Python\AI_Learning\Computer_Vision\IntroAI_Project-Clothes_Recommendation_System\datasets\train\dress\154.png"
)
print(label)

vector_search_stage = {
    "$vectorSearch": {
        "index": "vector_index",
        "queryVector": query_embedding,
        "path": "feature",
        "numCandidates": num_candidates,  # Số lượng vector ứng viên.
        "limit": 20,  # Trả về k vector gần nhất.
    },
}

unset_stage = {
    # Loại bỏ trường embedding khỏi kết quả trả về.
    "$unset": "feature",
}

project_stage = {
    "$project": {
        "_id": 0,  # Exclude the _id field
        "name": 1,  # Include the Phone url
        "score": {
            "$meta": "vectorSearchScore",  # Include the search score
        },
    },
}

# Xây dựng pipeline
pipeline = [vector_search_stage, unset_stage, project_stage]
res = list(collection.aggregate(pipeline))
sorted_data = sorted(res, key=lambda x: x["score"], reverse=True)

# Thực thi pipeline
results = [doc["name"] for doc in sorted_data]

fol_img = os.path.join("./datasets/train", str(label))
count = 0

for r in results:
    try:
        img_path = os.path.join(fol_img, r)
        Image.open(img_path).convert("RGB").copy().show()
        count += 1
        if count == 5:
            break
    except Exception as e:
        print(e)
