import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from model import ResNet50


def test(root_ckpt, model, path, list_label, size_img=224):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(root_ckpt)
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])

    softmax = nn.Softmax(dim=-1)
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
    img = img.to(device).float()
    with torch.no_grad():
        output = model.forward_feature(img)
        print(output.tolist())
        probs = softmax(output)
        print(list_label[torch.argmax(probs)])


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

model = ResNet50(len(categories))

test(
    r"trained_model/best_resnet50.pt",
    model,
    r"D:\Python\AI_Learning\Computer_Vision\IntroAI_Project-Clothes_Recommendation_System\datasets\train\dress\0a69db60-c052-4b9a-a90d-e53120d091d5.jpg",
    categories,
)
