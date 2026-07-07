#!/usr/bin/env python
"""
"""
from argparse import ArgumentParser
import logging
import os
import pdb
import pytorch_lightning as pl
import torch
from torch.utils.data import default_collate, DataLoader
from torchinfo import summary
import classifier_module
from torchsig.utils.yaml import load_dataset_yaml
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import DatasetCreator

logger = logging.getLogger(__name__)


transforms = [ComplexTo2D()]


def gen_training_set(args):
    """Generate into dataset a train subdirectory and validate"""
    dataset_iter = load_dataset_yaml(args.yaml_file)
    # dataset_iter.set_transforms(transforms)

    class_list = dataset_iter.class_names

    if not dataset_iter.validate_metadata_fields():
        logger.error("Metadata fields failed.")
        raise RuntimeError("dataset metadata validator failed")

    train_dataloader = WorkerSeedingDataLoader(
        dataset_iter, batch_size=4, collate_fn=default_collate
    )

    # Prepare a training set
    dc = DatasetCreator(
        dataloader=train_dataloader, root=f"dataset/train",
        overwrite=True, dataset_length=1000
    )
    dc.create()

    # Prepare a validation set
    dc = DatasetCreator(
        dataloader=train_dataloader, root=f"dataset/validate",
        overwrite=True, dataset_length=1000
    )
    dc.create()
    return class_list, train_dataloader


def train_xcit_model(train_dataloader, class_list, num_epochs=1):
    num_classes = len(class_list)
    model = classifier_module.XCiTClassifier(
        input_channels=2, num_classes=num_classes,
        xcit_version="nano_12_p16_224",
        ds_method="downsample", ds_rate=1)
    summary(model)
    train_dataset = StaticTorchSigDataset(
        root="dataset/train", target_labels=class_list, transform=ComplexTo2D())

    # 2. Create a standard PyTorch DataLoader for the actual training loop
    actual_train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    trainer = pl.Trainer(
        limit_train_batches=50,
        limit_val_batches=5,
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
    )

    trainer.fit(model, actual_train_loader)
    return model


def test_model(model, class_list):
    test_dataset = StaticTorchSigDataset(root=f"dataset/validate", target_labels=class_list, transform=ComplexTo2D())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, class_index = test_dataset[0]

    model.to(device)
    model.eval()

    with torch.no_grad():  # do not update model weights

        # convert to tensor and add a batch dimension
        data = torch.from_numpy(data).to(device).unsqueeze(dim=0)
        # have model predict data
        # returns a probability the data is each signal class
        pred = model(data)
        # print(pred) # if you want to see the list of probabilities

        # choose the class with highest confidence
        predicted_class = torch.argmax(pred).cpu().numpy()
        print(f"Predicted = {predicted_class} ({class_list[predicted_class]})")
        print(f"Actual = {class_index} ({class_list[class_index]})")


def main():
    parser = ArgumentParser()
    parser.add_argument("yaml_file")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs of training")
    args = parser.parse_args()

    if not os.path.isfile(args.yaml_file):
        logger.error(f"{args.yaml_file} not found!")
        raise FileNotFoundError(f"{args.yaml_file} not found!")

    # =========================================================================
    # Generate Training dataset
    # =========================================================================
    class_list, train_dataloader = gen_training_set(args)

    # =========================================================================
    # Classifier
    # =========================================================================
    model = train_xcit_model(train_dataloader, class_list, args.epochs)


    test_model(model, class_list)

    pdb.set_trace()

if __name__ == "__main__":
    main()