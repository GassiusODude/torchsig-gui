#!/usr/bin/env python
"""
target_labels: should be a single element list with either {"class_name", "class_index"}
    Use ["class_name"] to return str name instead of arbitrary index
"""
from argparse import ArgumentParser, Namespace
import logging
import numpy as np
import os
from pathlib import Path
import pdb
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import default_collate, DataLoader
from torchinfo import summary
from torchsig.utils.yaml import load_dataset_yaml, load_config_from_yaml
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset, TorchSigDatasetConfig
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.signal_building import lookup_signal_generator_by_string
from torchsig.utils.writer import DatasetCreator
from typing import Optional
import yaml

import classifier_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transforms = [ComplexTo2D()]
batch_size = 4
class_list = []

def create_dataset_from_yaml_config(yaml_path : Path):
    """
    Custom load from YAML config to initialize the dataset

    Avoids `load_dataset_yaml()`, which initializes the dataset with
    `metadata`, `transforms`=[], `target_labels`, and `seed`.  The rest of
    the arguments is left to default settings in TorchSigIterableDataset,
    which includes all 57 signal types in its signal generator list.

    This function will support a subset of signals and even signal families
    instead of all signals.
    """
    # Load YAML dictionary
    yaml_dict : dict = {}
    with open(yaml_path) as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    # Load config
    my_dataset_cfg : TorchSigDatasetConfig = load_config_from_yaml(yaml_path)
    logger.info(f"Dataset Id={my_dataset_cfg.dataset_id} of length = {my_dataset_cfg.dataset_length}")
    logger.info(f"{my_dataset_cfg.output_representation} {my_dataset_cfg.signal_sampling_mode}")
    logger.info(f"{my_dataset_cfg.dataset_metadata}")

    # Initialize dataset
    ds = TorchSigIterableDataset(
        metadata=my_dataset_cfg.dataset_metadata,
        signal_generators=[],
        transforms=[ComplexTo2D()],
        target_labels=yaml_dict["target_labels"],
        seed=my_dataset_cfg.seed
    )

    # NOTE: initialize classes based on the YAML dictionary
    logger.info(f"Class list = {yaml_dict.get("class_list", "all")}")
    for c_class in yaml_dict["class_list"]:
        c_gen = lookup_signal_generator_by_string(c_class)
        ds.add_signal_generator(c_gen, likelihood=1)

    if not ds.validate_metadata_fields():
        logger.error("Metadata fields failed.")
        raise RuntimeError("dataset metadata validator failed")

    return yaml_dict, my_dataset_cfg, ds


def gen_training_set(args : Namespace):
    """
    Generate into dataset a train subdirectory and validate
    """
    ydict, my_dataset_cfg, my_dataset = create_dataset_from_yaml_config(
        Path(os.path.abspath(args.yaml_file)))

    class_list : list[str] = ydict["class_list"]

    train_dataloader = WorkerSeedingDataLoader(
        my_dataset, batch_size=batch_size, collate_fn=lambda x: x)

    # Prepare a training set
    # -----------------------------------------------------
    dc = DatasetCreator(
        dataloader=train_dataloader, root=f"dataset/train",
        overwrite=True, dataset_length=10000)
    dc.create()

    # Prepare a validation set
    # -----------------------------------------------------
    dc = DatasetCreator(
        dataloader=train_dataloader, root=f"dataset/validate",
        overwrite=True, dataset_length=10000)
    dc.create()
    return class_list, train_dataloader


def train_xcit_model(
        args: Namespace, class_list : list[str]):
    """
    Train the XCiT model for modulation classification
    """
    # -----------------------------------------------------
    # Initialize model
    # -----------------------------------------------------
    num_classes = len(class_list)
    model = classifier_module.XCiTClassifier(
        input_channels=2, num_classes=num_classes,
        # xcit_version="nano_12_p16_224",
        ds_method=args.ds_type, ds_rate=args.ds_rate,
        learning_rate=args.lr,
    )
    summary(model)

    # NOTE: prepare the number of epochs (extend if loading previous checkpoint)
    if args.model is not None and os.path.isfile(args.model):
        # Load just the metadata loop dictionary from the checkpoint safely
        checkpoint_meta = torch.load(args.model, map_location="cpu")
        starting_epoch = checkpoint_meta.get("epoch", 0)
        logger.info(f" Found existing checkpoint at epoch: {starting_epoch}")

        # Add your requested incremental epochs to the current checkpoint level
        num_epochs = starting_epoch + args.epochs
    else:
        num_epochs = args.epochs

    # NOTE: if epochs is zero or skip training...return the loaded model
    if num_epochs == 0 or args.skip_train:
        return model

    # -----------------------------------------------------
    # Prepare training data
    # -----------------------------------------------------
    train_dataset = StaticTorchSigDataset(
        root="dataset/train", target_labels=["class_index"])

    actual_train_loader = WorkerSeedingDataLoader(
        train_dataset, batch_size=args.batch, collate_fn=lambda x: x
    )

    # -----------------------------------------------------
    # Prepare validation data
    # -----------------------------------------------------
    valid_dataset = StaticTorchSigDataset(
        root="dataset/validate", target_labels=["class_index"])

    actual_valid_loader = WorkerSeedingDataLoader(
        valid_dataset, batch_size=4, collate_fn=lambda x: x
    )

    # NOTE: added "enable_checkpointing=False"
    trainer = pl.Trainer(
        limit_train_batches=50,
        limit_val_batches=5,
        max_epochs=num_epochs,
        check_val_every_n_epoch=args.validate_period,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=True,
    )

    # -----------------------------------------------------
    # Smart Load and Train
    # -----------------------------------------------------
    # Check if a Lightning checkpoint exists instead of raw state_dict
    if args.model is not None and os.path.isfile(args.model):
        logger.info(f"Resuming training from checkpoint: {args.model} with NEW LR: {args.lr}")

        trainer.fit(
            model,
            train_dataloaders=actual_train_loader,
            val_dataloaders=actual_valid_loader,
            ckpt_path=args.model  # Resumes perfectly
        )
    else:
        logger.info("Starting training from scratch...")
        trainer.fit(
            model,
            train_dataloaders=actual_train_loader,
            val_dataloaders=actual_valid_loader
        )

    # save checkpoint
    trainer.save_checkpoint(args.model)

    return model


def test_model(model, class_list):
    test_dataset = StaticTorchSigDataset(
        root=f"dataset/validate",
        target_labels=["class_index"],
        # target_labels = class_list
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():  # do not update model weights
        for data, class_index in test_dataset:
            # convert to tensor and add a batch dimension
            data = torch.from_numpy(data).to(device).unsqueeze(dim=0)

            # have model predict data
            pred = model(data)

            # choose the class with highest confidence
            predicted_class : np.ndarray = torch.argmax(pred).cpu().numpy()
            actual_class : np.ndarray = np.array(class_index, dtype=np.int64)
            # logger.info(f"Predicted = {predicted_class}, Actual = {actual_class}")

            try:
                all_preds.append(predicted_class.item())
                all_targets.append(actual_class.item())
            except Exception as e:
                pdb.set_trace()
            if predicted_class == actual_class:
                correct_predictions += 1

            total_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100

    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_report = classification_report(all_targets, all_preds, target_names=class_list)

    print(f"Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)


def main():
    parser = ArgumentParser()
    parser.add_argument("yaml_file")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs of training")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--ds_rate", default=2, type=int, help="Downsample rate")
    parser.add_argument("--ds_type", default="downsample", choices=["downsample", "chunk"])
    parser.add_argument("--validate_period", default=5, type=int, help="Check validation every N epochs")
    parser.add_argument("--batch", default=4, type=int, help="Size of batch")
    args = parser.parse_args()

    if not os.path.isfile(args.yaml_file):
        logger.error(f"{args.yaml_file} not found!")
        raise FileNotFoundError(f"{args.yaml_file} not found!")

    # =========================================================================
    # Generate Training dataset
    # =========================================================================
    class_list, _ = gen_training_set(args)

    # =========================================================================
    # Classifier
    # =========================================================================
    model = train_xcit_model(args, class_list)

    test_model(model, class_list)


if __name__ == "__main__":
    main()