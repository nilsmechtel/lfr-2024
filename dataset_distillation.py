import os
import time
from dotenv import load_dotenv
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from dataset import prepare_mnist_data
from models import DatasetDistillation


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Define the path for the model checkpoints
    if args.checkpoint_path is not None:
        args.checkpoint_path = os.path.abspath(args.checkpoint_path)
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    # Initialize the Wandb logger
    load_dotenv()
    wandb.login()
    wandb_logger = WandbLogger(project="DatasetDistillation", log_model="all")

    # Initialize the DataLoader
    train_loader, val_loader = prepare_mnist_data()

    # Initialize the model
    model = DatasetDistillation(
        training_mode="pretrain",  # Set the mode to pretrain for the FeatureExtractor
        latent_dim=args.latent_dim,
        num_distilled_data=args.num_distilled_data,
        prime_proportion=args.prime_distilled_data_proportion,
        prime_data_loader=train_loader,
        classification_scale=args.classification_scale,
        gradient_penalty_scale=args.gradient_penalty_scale,
        reconstruction_scale=args.reconstruction_scale,
        diversity_scale=args.diversity_scale,
        schedule_reconstruction_weight=args.schedule_reconstruction_weight,
        schedule_diversity_weight=args.schedule_diversity_weight,
        weight_decay=args.weight_decay,
        learning_rate_ae=args.learning_rate_ae,
        learning_rate_distill=args.learning_rate_distill,
        log_image_every=args.log_image_every,
        feature_extractor_cls=args.feature_extractor_cls,
        model_head_cls=args.model_head_cls,
    )

    # Pretraining phase
    if args.pretrained_model:
        print("Loading pretrained model...")
        pretrained_model_path = os.path.join(
            args.checkpoint_path, args.pretrained_model
        )
        pretrained_model = DatasetDistillation.load_from_checkpoint(
            pretrained_model_path,
            map_location="cuda:0",
        )
        model.feature_extractor.load_state_dict(
            pretrained_model.feature_extractor.state_dict()
        )
        del pretrained_model
    else:
        # Early stopping for pretraining
        early_stopping = EarlyStopping(
            monitor="val_ae_loss",  # Monitor validation autoencoder loss
            patience=10,  # Patience before stopping
            verbose=True,
            mode="min",  # Stop when the validation loss is minimized
        )

        # Model checkpoint for pretraining
        filename = f"feature_extractor_{args.feature_extractor_cls.lower()}_latent_dim_{args.latent_dim}"
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_path,
            filename=filename,
            monitor="val_ae_loss",
            mode="min",
            save_top_k=1,
        )

        # Pretraining phase for FeatureExtractor
        trainer = pl.Trainer(
            max_epochs=150,
            accelerator="gpu",
            logger=wandb_logger,
            callbacks=[early_stopping, checkpoint_callback],
        )

        # Pretrain the FeatureExtractor model
        print("Starting pretraining the FeatureExtractor...")
        trainer.fit(model, train_loader, val_loader)

    if args.only_pretrain:
        print("Pretraining completed!")
        return

    # After pretraining, switch to distillation mode
    model.training_mode = "distillation"

    # Early stopping for distillation
    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=10, verbose=True, mode="max"
    )

    # Model checkpoint for distillation
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f"data_distillation_model_{timestamp}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
    )

    # Dataset distillation phase
    trainer = pl.Trainer(
        max_epochs=250,
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback],
    )

    # Train the model using distillation
    print("Starting distillation training...")
    trainer.fit(model, train_loader, val_loader)

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Distillation")
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Latent dimension of the autoencoder"
    )
    parser.add_argument(
        "--num_distilled_data", type=int, default=80, help="Number of distilled data"
    )
    parser.add_argument(
        "--prime_distilled_data_proportion",
        type=float,
        default=0.25,
        help="Proportion of distilled data to prime",
    )
    parser.add_argument(
        "--classification_scale", type=float, default=1.0, help="Classification scale"
    )
    parser.add_argument(
        "--reconstruction_scale", type=float, default=1.0, help="Reconstruction scale"
    )
    parser.add_argument(
        "--gradient_penalty_scale",
        type=float,
        default=100.0,
        help="Gradient penalty scale",
    )
    parser.add_argument(
        "--diversity_scale", type=float, default=1.0, help="Diversity scale"
    )
    parser.add_argument(
        "--schedule_reconstruction_weight",
        nargs="+",
        type=list,
        default=[0, 100],
        help="Increase reconstruction scale over n epochs, then decrease over n epochs",
    )
    parser.add_argument(
        "--schedule_diversity_weight",
        nargs="+",
        type=list,
        default=[0, 0],
        help="Increase diversity scale over n epochs, then decrease over n epochs",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay to limit overfitting",
    )
    parser.add_argument(
        "--learning_rate_ae",
        type=float,
        default=1e-3,
        help="Learning rate for the autoencoder",
    )
    parser.add_argument(
        "--learning_rate_distill",
        type=float,
        default=1e-4,
        help="Learning rate for distillation",
    )
    parser.add_argument(
        "--log_image_every", type=int, default=10, help="Log image every n epochs"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to the pretrained model checkpoint",
    )
    parser.add_argument(
        "--only_pretrain",
        action="store_true",
        default=False,
        help="Only perform pretraining",
    )
    parser.add_argument(
        "--feature_extractor_cls",
        type=str,
        default="WeightSharedAutoencoder",
        choices=[
            "Autoencoder",
            "WeightSharedAutoencoder",
            "Autoencoder2L",
            "WeightSharedAutoencoder2L",
            "GRBM",
        ],
        help="Feature extractor class name",
    )
    parser.add_argument(
        "--model_head_cls",
        type=str,
        default="LinearClassifier",
        choices=["LinearClassifier"],
        help="Model head class name",
    )
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU ID")
    args = parser.parse_args()

    main(args)
