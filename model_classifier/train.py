import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomAffine,
    Resize,
    ToTensor,
)
from tqdm.autonotebook import tqdm

from data import ClotheDataset
from model import ResNet50


def get_args():
    parser = argparse.ArgumentParser(description="Animals classifier")
    parser.add_argument(
        "--data_path", type=str, default="datasets", help="the root folder of the data"
    )
    parser.add_argument("--epochs", default=20, type=int, help="Total number of epochs")
    parser.add_argument("--batch_size", default=4, type=int)
    args = parser.parse_args()
    return args


def train(
    model,
    train_dataloader,
    test_dataloader,
    is_load_model=False,
    epochs=20,
    lr=1e-3,
):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số trong model: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    if is_load_model:
        checkpoint = torch.load(r"trained_model/last_resnet50.pt")
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_val_acc = torch.load(r"trained_model/best_resnet50.pt")["best_acc"]
    else:
        start_epoch = 0
        best_val_acc = 0.0

    num_iters = len(train_dataloader)

    # save model
    if not os.path.isdir("trained_model"):
        os.mkdir("trained_model")

    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(start_epoch, epochs):
        progress_bar = tqdm(train_dataloader)
        model.train()
        for iter, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description(
                "Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(
                    epoch + 1, epochs, iter + 1, num_iters, loss_value
                )
            )

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                res = model(images)
                res = torch.argmax(res.cpu(), dim=1)
                all_predictions.extend(res)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]

        val_acc = accuracy_score(all_labels, all_predictions)
        print("val_acc: ", val_acc)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, r"trained_model/last_resnet50.pt")

        if val_acc > best_val_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_val_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, r"trained_model/best_resnet50.pt")
            best_val_acc = val_acc


if __name__ == "__main__":
    args = get_args()
    transform_train = Compose(
        [
            ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),
            RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_test = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = ClotheDataset(
        datasets_path=args.data_path,
        part="train",
        transform=transform_train,
    )
    val_data = ClotheDataset(
        datasets_path=args.data_path,
        part="val",
        transform=transform_test,
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    val_dataloader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    print("train_data: ", train_data.__len__())
    print("test_data: ", val_data.__len__())

    model = ResNet50(num_classes=len(train_data.categories))

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        is_load_model=False,
        epochs=args.epochs,
        lr=1e-3,
    )
