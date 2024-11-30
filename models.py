import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from typing import Literal

from visualisation import create_image_grid, plot_contingency_table


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),  # Assumes input is normalized to [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Returns:
            - reconstructed: The reconstructed input.
            - latent: The latent representation of the input.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def reconstruction_loss(self, x):
        """
        Compute reconstruction loss (mean squared error).

        Args:
            x (Tensor): Input tensor.
        Returns:
            loss (Tensor): Reconstruction loss.
        """
        reconstructed, _ = self.forward(x)
        return F.mse_loss(reconstructed, x)


class WeightSharedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder_bias = nn.Parameter(
            torch.zeros(input_dim)
        )  # Separate decoder bias

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Returns:
            - reconstructed: The reconstructed input.
            - latent: The latent representation of the input.
        """
        latent = torch.relu(self.encoder(x))
        reconstructed = torch.sigmoid(
            torch.matmul(latent, self.encoder.weight) + self.decoder_bias
        )
        return reconstructed, latent

    def reconstruction_loss(self, x):
        """
        Compute reconstruction loss (mean squared error).

        Args:
            x (Tensor): Input tensor.
        Returns:
            loss (Tensor): Reconstruction loss.
        """
        reconstructed, _ = self.forward(x)
        return F.mse_loss(reconstructed, x)


class GRBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sigma=1.0):
        """
        Gaussian Restricted Boltzmann Machine (GRBM) with visible and hidden layers.

        Args:
            visible_dim (int): Number of visible units.
            hidden_dim (int): Number of hidden units.
            sigma (float): Standard deviation of the Gaussian noise.
        """
        super(GRBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.sigma = sigma  # Standard deviation for visible units

        # Weight and bias initialization
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(visible_dim))  # Bias for visible units
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))  # Bias for hidden units

    def sample_hidden(self, v):
        """
        Sample the hidden units given visible units.

        Args:
            v (Tensor): Input tensor for visible units.
        Returns:
            h_prob (Tensor): Probabilities of hidden units being activated.
            h_sample (Tensor): Sampled hidden units.
        """
        h_prob = torch.sigmoid(F.linear(v, self.W, self.b_h))
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample

    def sample_visible(self, h):
        """
        Sample the visible units given hidden units.

        Args:
            h (Tensor): Input tensor for hidden units.
        Returns:
            v_mean (Tensor): Reconstructed visible units (mean).
            v_sample (Tensor): Sampled visible units.
        """
        v_mean = F.linear(h, self.W.t(), self.b_v)
        v_sample = v_mean + torch.randn_like(v_mean) * self.sigma
        return v_mean, v_sample

    def forward(self, v):
        """
        Perform one step of Gibbs sampling.

        Args:
            v (Tensor): Input tensor for visible units.
        Returns:
            v_mean (Tensor): Reconstructed visible units (mean).
            h_prob (Tensor): Probabilities of hidden units being activated.
        """
        h_prob, h_sample = self.sample_hidden(v)
        v_mean, _ = self.sample_visible(h_sample)
        return v_mean, h_prob

    def reconstruction_loss(self, v):
        """
        Compute the Negative Log-Likelihood (NLL) loss for Gaussian visible units.

        Args:
            v (Tensor): Input tensor for visible units.
        Returns:
            loss (Tensor): NLL reconstruction loss.
        """
        # Perform one step of Gibbs sampling to get reconstructed visible units
        v_mean, _ = self.forward(v)
        loss = F.gaussian_nll_loss(v_mean, v, torch.ones_like(v) * self.sigma, reduction="mean")
        return loss


class LinearClassifier(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(latent_dim, n_classes)

    def forward(self, latent):
        """
        Forward pass through the model head.

        Returns:
            - logits: Output logits for classification.
        """
        logits = self.head(latent)
        return logits

    def classification_loss(self, latent, labels):
        """
        Compute classification loss (cross-entropy).

        Args:
            latent (Tensor): Latent feature vector.
            labels (Tensor): Ground truth class indices.
        Returns:
            loss (Tensor): Classification loss.
        """
        logits = self.forward(latent)
        return F.cross_entropy(logits, labels)


class DatasetDistillation(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 784,  # Flattened image size (28*28)
        latent_dim: int = 64,
        n_classes: int = 10,  # Number of classes in MNIST
        num_distilled_data: int = 300,  # Size of the distilled dataset
        prime_proportion: float = 0.0,
        prime_data_loader: DataLoader = None,
        classification_scale: float = 1.0,
        reconstruction_scale: float = 1.0,
        gradient_penalty_scale: float = 1.0,
        diversity_scale: float = 1.0,
        increase_reconstruction_over: int = 1,
        increase_diversity_over: int = 1,
        weight_decay: float = 1e-4,
        learning_rate_ae: float = 1e-3,
        learning_rate_distill: float = 1e-4,
        training_mode: Literal["pretrain", "distillation"] = "distillation",
        log_image_every: int = 10,
        feature_extractor_cls: Literal[
            "Autoencoder", "WeightSharedAutoencoder", "GRBM"
        ] = "WeightSharedAutoencoder",
        model_head_cls: Literal["LinearClassifier"] = "LinearClassifier",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model components
        feature_extractor_cls = {
            "Autoencoder": Autoencoder,
            "WeightSharedAutoencoder": WeightSharedAutoencoder,
            "GRBM": GRBM,
        }[feature_extractor_cls]
        self.feature_extractor = feature_extractor_cls(input_dim, latent_dim)
        model_head_cls = {
            "LinearClassifier": LinearClassifier,
        }[model_head_cls]
        self.model_head = model_head_cls(latent_dim, n_classes)

        # Distilled data as trainable parameters (range [0, 1])
        self.distilled_data = self.initialize_distilled_data(
            num_distilled_data,
            input_dim,
            prime_proportion,
            prime_data_loader,
        )

        # Scaling parameters
        self.classification_scale = classification_scale
        self.reconstruction_scale = reconstruction_scale
        self.gradient_penalty_scale = gradient_penalty_scale
        self.diversity_scale = diversity_scale
        self.increase_reconstruction_over = increase_reconstruction_over
        self.increase_diversity_over = increase_diversity_over

        # Learning rates and weight decay
        self.weight_decay = weight_decay
        self.learning_rate_ae = learning_rate_ae
        self.learning_rate_distill = learning_rate_distill

        # Toggle for training mode
        self.training_mode = training_mode

        # Logging parameters
        self.log_image_every = log_image_every

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value):
        if value not in ["pretrain", "distillation"]:
            raise ValueError(
                "Invalid training mode. Must be 'pretrain' or 'distillation'."
            )
        self._training_mode = value

    def initialize_distilled_data(
        self,
        num_distilled_data: int,
        input_dim: int,
        prime_proportion: float,
        prime_data_loader: DataLoader,
    ):
        """Create the distilled data with a random sample from the data loader."""
        distilled_data = torch.rand(num_distilled_data, input_dim)

        if prime_proportion > 0 and prime_data_loader is not None:
            indices = torch.randperm(len(prime_data_loader.dataset))[
                :num_distilled_data
            ]
            prime_data = torch.cat(
                [prime_data_loader.dataset[idx][0][None, :] for idx in indices]
            )
            distilled_data = (
                prime_proportion * prime_data + (1 - prime_proportion) * distilled_data
            )

        return nn.Parameter(distilled_data)

    def forward(self, x):
        """Forward pass through the autoencoder and model head."""
        reconstructed, z = self.feature_extractor(x)
        logits = self.model_head(z)
        return reconstructed, logits

    def reconstruction_loss(self, x):
        """Feature extractor reconstruction loss."""
        return self.feature_extractor.reconstruction_loss(x)

    def classification_loss(self, latent, labels):
        """Cross-entropy classification loss."""
        return self.model_head.classification_loss(latent, labels)

    def penalizer_term(self):
        """Penalizer term based on gradients of the autoencoder."""
        # Compute MSE loss between reconstructed and original distilled data
        mse_loss = self.reconstruction_loss(self.distilled_data)

        # Compute gradients of the loss with respect to the distilled data
        # Construct graph of the derivative to allow computing higher order derivative products.
        gradients = torch.autograd.grad(
            outputs=mse_loss,
            inputs=self.distilled_data,
            create_graph=True,
        )[0]

        # Apply smooth L1 loss
        return F.smooth_l1_loss(gradients, torch.zeros_like(gradients))

    def diversity_penalty(self):
        """Diversity penalty to encourage variation in distilled data."""
        return F.mse_loss(
            self.distilled_data,
            self.distilled_data.mean(dim=0, keepdim=True).expand_as(
                self.distilled_data
            ),
        )

    def calculate_performance(self, logits, labels):
        """
        Calculate the accuracy of predictions compared to the true labels.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
            labels (torch.Tensor): True labels (one-hot encoded or class indices) of shape (batch_size, num_classes) or (batch_size,).

        Returns:
            float: Accuracy value between 0 and 1.
            torch.Tensor: Contingency table of shape (num_classes, num_classes).
        """
        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Convert predictions to class indices
        predicted_classes = torch.argmax(probabilities, dim=1)

        # If labels are one-hot encoded, convert to class indices
        if labels.ndim > 1 and labels.size(1) > 1:
            true_classes = torch.argmax(labels, dim=1)
        else:
            true_classes = labels

        # Calculate contingency table
        n_classes = labels.size(1)
        contingency_table = torch.zeros(n_classes, n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                contingency_table[i, j] = (
                    ((predicted_classes == i) & (true_classes == j)).sum().item()
                )

        # Calculate the number of correct predictions
        correct_predictions = (predicted_classes == true_classes).sum().item()

        # Calculate accuracy
        total_samples = labels.size(0)
        accuracy = correct_predictions / total_samples

        return accuracy, contingency_table

    def configure_optimizers(self):
        """Set up optimizers for AE pretraining and dataset distillation."""
        if self.training_mode == "pretrain":
            ae_optimizer = torch.optim.Adam(
                self.feature_extractor.parameters(),
                lr=self.learning_rate_ae,
                weight_decay=self.weight_decay,
            )
            return ae_optimizer
        else:
            distill_optimizer = torch.optim.Adam(
                list(self.feature_extractor.parameters())
                + list(self.model_head.parameters())
                + [self.distilled_data],
                lr=self.learning_rate_distill,
                weight_decay=self.weight_decay,
            )
            return distill_optimizer

    def training_step(self, batch, batch_idx):
        """Define the training step."""
        x, y = batch

        if self.training_mode == "pretrain":
            # Pretrain autoencoder on reconstruction
            ae_loss = self.reconstruction_loss(x)
            self.log("train_ae_loss", ae_loss, on_step=False, on_epoch=True)
            return ae_loss
        else:
            # Dataset distillation training

            # Classification loss
            _, z = self.feature_extractor(x)
            classification_loss = self.classification_scale * self.classification_loss(
                z, y
            )
            self.log(
                "train_classification_loss",
                classification_loss,
                on_step=False,
                on_epoch=True,
            )

            # Reconstruction loss
            reconstruction_loss = self.reconstruction_scale * self.reconstruction_loss(
                self.distilled_data
            )
            self.log(
                "train_reconstruction_loss",
                reconstruction_loss,
                on_step=False,
                on_epoch=True,
            )

            # Gradient penalty
            grad_penalty = self.gradient_penalty_scale * self.penalizer_term()
            self.log("train_grad_penalty", grad_penalty, on_step=False, on_epoch=True)

            # Diversity penalty
            diversity_penalty = self.diversity_scale * self.diversity_penalty()
            self.log(
                "train_diversity_penalty",
                diversity_penalty,
                on_step=False,
                on_epoch=True,
            )

            # Increase weights over time
            reconstruction_weight = min(
                1, ((self.current_epoch + 1) / self.increase_reconstruction_over)
            )
            self.log(
                "train_reconstruction_weight",
                reconstruction_weight,
                on_step=False,
                on_epoch=True,
            )
            diversity_weight = min(
                1, ((self.current_epoch + 1) / self.increase_diversity_over)
            )
            self.log(
                "train_diversity_weight", diversity_weight, on_step=False, on_epoch=True
            )

            # Combined loss
            distillation_loss = (
                classification_loss
                + grad_penalty
                + reconstruction_weight * reconstruction_loss
                - diversity_weight * diversity_penalty
            )

            self.log(
                "train_distillation_loss",
                distillation_loss,
                on_step=False,
                on_epoch=True,
            )
            return distillation_loss

    def validation_step(self, batch, batch_idx):
        """Define the validation step."""
        x, y = batch
        reconstructed, logits = self.forward(x)

        # Log real image reconstruction
        if self.current_epoch % self.log_image_every == 0 and batch_idx == 0:
            if not hasattr(self, "first_indices"):
                # Find the classes from the one-hot encoded tensor
                classes = torch.argmax(y, dim=1)

                # Create an array to hold the first indices
                first_indices = -torch.ones(
                    y.shape[1], dtype=torch.int
                )  # Initialize to -1 (not found)

                for idx, label in enumerate(classes):
                    if (
                        first_indices[label.item()] == -1
                    ):  # If the class hasn't been recorded yet
                        first_indices[label.item()] = idx

                self.first_indices = first_indices

            # Log first 10 ground truth and reconstructed images (FeatureExtractor)
            reconstructed_images = create_image_grid(
                torch.cat([x[self.first_indices], reconstructed[self.first_indices]])
            )
            reconstructed_images = wandb.Image(
                reconstructed_images, caption=f"Epoch {self.current_epoch}"
            )
            self.logger.log_image(
                key=f"{self.training_mode}_reconstructed_images",
                images=[reconstructed_images],
            )

        if self.training_mode == "pretrain":
            # Log autoencoder reconstruction loss
            ae_loss = self.reconstruction_loss(x)
            self.log("val_ae_loss", ae_loss, on_step=False, on_epoch=True)

            return ae_loss

        else:
            # Log classification accuracy
            accuracy, contingency_table = self.calculate_performance(logits, y)
            self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)

            if self.current_epoch % self.log_image_every == 0 and batch_idx == 0:
                # Reset contingency table
                self.contingency_table = contingency_table

                # Log first 80 distilled images
                distilled_data = self.distilled_data[:80].clone()
                distilled_images = create_image_grid(distilled_data)
                distilled_images = wandb.Image(
                    distilled_images, caption=f"Epoch {self.current_epoch}"
                )
                self.logger.log_image(
                    key="val_distilled_images", images=[distilled_images]
                )
            else:
                self.contingency_table += contingency_table

            return accuracy

    def on_validation_epoch_end(self):
        if (
            self.training_mode == "distillation"
            and self.current_epoch % self.log_image_every == 0
        ):
            # Normalize contingency table (avoid division by zero)
            contingency_table = self.contingency_table / (
                self.contingency_table.sum(dim=1, keepdim=True) + 1e-6
            )
            # Plot contingency table as a heatmap
            heatmap = plot_contingency_table(contingency_table)
            heatmap = wandb.Image(heatmap, caption=f"Epoch {self.current_epoch}")
            self.logger.log_image(key="val_contingency_table", images=[heatmap])


if __name__ == "__main__":
    from dataset import prepare_mnist_data
    import numpy as np
    from PIL import Image

    # Prepare data
    train_loader, val_loader = prepare_mnist_data()
    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)

    # Test autoencoder
    autoencoder = Autoencoder(784, 64)
    print(autoencoder)
    loss = autoencoder.reconstruction_loss(x)

    # Test weight-shared autoencoder
    wsae = WeightSharedAutoencoder(784, 64)
    print(wsae)
    loss = wsae.reconstruction_loss(x)

    # Test GRBM
    grbm = GRBM(784, 64)
    print(grbm)
    loss = grbm.reconstruction_loss(x)

    # Initialize model
    model = DatasetDistillation(training_mode="pretrain")
    print(model)

    # Test model training steps
    loss = model.training_step((x, y), 0)
    print("AE loss:", loss)

    model.training_mode = "distillation"
    loss = model.training_step((x, y), 0)
    print("Distillation loss:", loss)

    # Test image creation
    image = create_image_grid(torch.cat([x[:10], x[-10:]]))
    image_data = np.array(wandb.Image(image).image)
    pil_image = Image.fromarray(image_data)
    pil_image.save("test_image.png")

    image = create_image_grid(model.distilled_data[:80])
    image_data = image_data = np.array(wandb.Image(image).image)
    pil_image = Image.fromarray(image_data)
    pil_image.save("test_image_distilled.png")
