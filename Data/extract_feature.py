import argparse
import os

import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from data import ClotheDatasetSave
from model import ResNet50


def get_args():
    parser = argparse.ArgumentParser(description="Animals classifier")
    parser.add_argument(
        "--model", type=str, default="model", help="the root folder of the data"
    )
    parser.add_argument(
        "--data", type=str, default="model", help="the root folder of the data"
    )
    args = parser.parse_args()
    return args


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
load_dotenv()
args = get_args()


def extract_feature(
    root_ckpt, model, dataloader, db_name="IntroAI", collection_name="clothes_imgs"
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(root_ckpt)
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Kết nối MongoDB
    client = MongoClient(os.getenv("URL_MONGODB"))
    db = client[db_name]
    collection = db[collection_name]

    # Xóa dữ liệu cũ trong collection (nếu cần)
    collection.delete_many({})

    doc_count = 0  # Biến đếm số document đã lưu
    print("Start extracting features...")
    for images, names in dataloader:
        names = list(names)
        images = images.to(device)

        with torch.no_grad():
            label, output = model(images)

        features = output.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()

        docs = []

        for name, feature, lst in zip(names, features, label):
            max_index = lst.index(max(lst))
            doc = {
                "name": name,
                "feature": feature,
                "label": categories[int(max_index)],
            }
            docs.append(doc)
            doc_count += 1
        # for doc in features

        if docs:
            collection.insert_many(docs)  # Chỉ insert ID và metadata
            print("Inserted", doc_count, "documents")

    print(f"Tổng số document đã lưu: {doc_count}")


transform_test = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_path = args.data
train_data = ClotheDatasetSave(
    datasets_path=data_path,
    part="train",
    transform=transform_test,
)

train_dataloader = DataLoader(
    train_data, batch_size=128, shuffle=False, drop_last=False
)

print("done preparing data")
extract_feature(
    args.model,
    ResNet50(len(categories)),
    train_dataloader,
)
