"""
==============================================================================
Brain Tumor Classification Head
==============================================================================

Purpose:
    Converts the 128-dimensional feature vector produced by the multi-scale
    MLP into tumor classification logits and probabilities.

Input:
    (B, 128)

Output:
    (B, 4)

Classes:
    0 -> No Tumor
    1 -> Necrotic / Non-enhancing Tumor Core
    2 -> Edema
    3 -> Enhancing Tumor

Run:
    python models/classification/classifier.py
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s",
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Device
# ==============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    logger.info(
        "Using CUDA device: %s",
        torch.cuda.get_device_name(0),
    )
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU device")


# ==============================================================================
# Classification Head
# ==============================================================================


class BrainTumorClassifier(nn.Module):
    """
    Classification head for the multi-scale brain tumor feature vector.

    Parameters
    ----------
    input_dim:
        Dimension of the input feature vector.

    hidden_dim:
        Number of neurons in the hidden classification layer.

    num_classes:
        Number of output classes.

    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

        logger.info("Brain tumor classifier initialized.")
        logger.info(
            "Input dimension : %d",
            input_dim,
        )
        logger.info(
            "Hidden dimension : %d",
            hidden_dim,
        )
        logger.info(
            "Output classes   : %d",
            num_classes,
        )

    # ==========================================================================
    # Forward
    # ==========================================================================

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        features:
            Tensor with shape (B, input_dim).

        Returns
        -------
        torch.Tensor
            Classification logits with shape (B, num_classes).
        """

        if features.ndim != 2:
            raise ValueError(
                "Classifier expects a 2D tensor with shape "
                "(B, feature_dimension). "
                f"Received {tuple(features.shape)}."
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, "
                f"received {features.shape[1]}."
            )

        logits = self.classifier(features)

        return logits

    # ==========================================================================
    # Probabilities
    # ==========================================================================

    def predict_proba(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert classification logits into probabilities.
        """

        logits = self.forward(features)

        return torch.softmax(
            logits,
            dim=1,
        )

    # ==========================================================================
    # Prediction
    # ==========================================================================

    def predict(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the predicted class index.
        """

        logits = self.forward(features)

        return torch.argmax(
            logits,
            dim=1,
        )


# ==============================================================================
# Parameter Counter
# ==============================================================================


def count_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


# ==============================================================================
# Model Summary
# ==============================================================================


def print_model_summary(
    model: BrainTumorClassifier,
) -> None:
    """
    Print classifier configuration.
    """

    logger.info("=" * 70)
    logger.info("Brain Tumor Classifier Summary")
    logger.info("=" * 70)

    logger.info(
        "Input Dimension     : %d",
        model.input_dim,
    )

    logger.info(
        "Hidden Dimension    : %d",
        model.hidden_dim,
    )

    logger.info(
        "Output Classes      : %d",
        model.num_classes,
    )

    logger.info(
        "Trainable Parameters: %s",
        f"{count_parameters(model):,}",
    )

    logger.info("=" * 70)


# ==============================================================================
# Built-In GPU Test
# ==============================================================================


def test_model() -> None:
    """
    Test the classifier on GPU or CPU.
    """

    logger.info("Initializing brain tumor classifier...")

    model = BrainTumorClassifier(
        input_dim=128,
        hidden_dim=128,
        num_classes=4,
        dropout=0.2,
    )

    model = model.to(DEVICE)

    model.eval()

    print_model_summary(model)

    # --------------------------------------------------------------------------
    # Create synthetic MLP output
    # --------------------------------------------------------------------------

    logger.info("Creating test feature vector...")

    features = torch.randn(
        1,
        128,
        device=DEVICE,
    )

    logger.info(
        "Input Feature Shape : %s",
        tuple(features.shape),
    )

    # --------------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------------

    with torch.no_grad():

        logits = model(features)

        probabilities = model.predict_proba(features)

        prediction = model.predict(features)

    # --------------------------------------------------------------------------
    # Output information
    # --------------------------------------------------------------------------

    logger.info(
        "Logits Shape        : %s",
        tuple(logits.shape),
    )

    logger.info(
        "Probability Shape    : %s",
        tuple(probabilities.shape),
    )

    logger.info(
        "Prediction Shape     : %s",
        tuple(prediction.shape),
    )

    logger.info(
        "Predicted Class      : %d",
        prediction.item(),
    )

    # --------------------------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------------------------

    expected_logits_shape = (
        1,
        4,
    )

    expected_prediction_shape = (1,)

    if tuple(logits.shape) != expected_logits_shape:
        raise RuntimeError(
            "Incorrect logits shape. "
            f"Expected {expected_logits_shape}, "
            f"received {tuple(logits.shape)}."
        )

    if tuple(probabilities.shape) != expected_logits_shape:
        raise RuntimeError(
            "Incorrect probability shape. "
            f"Expected {expected_logits_shape}, "
            f"received {tuple(probabilities.shape)}."
        )

    if tuple(prediction.shape) != expected_prediction_shape:
        raise RuntimeError(
            "Incorrect prediction shape. "
            f"Expected {expected_prediction_shape}, "
            f"received {tuple(prediction.shape)}."
        )

    # --------------------------------------------------------------------------
    # Device validation
    # --------------------------------------------------------------------------

    if DEVICE.type == "cuda":

        if logits.device.type != "cuda":
            raise RuntimeError(
                "Incorrect logits device. " f"Expected CUDA, received {logits.device}."
            )

        if probabilities.device.type != "cuda":
            raise RuntimeError(
                "Incorrect probability device. "
                f"Expected CUDA, received {probabilities.device}."
            )

    else:

        if logits.device.type != "cpu":
            raise RuntimeError(
                "Incorrect logits device. " f"Expected CPU, received {logits.device}."
            )

    # --------------------------------------------------------------------------
    # Dtype validation
    # --------------------------------------------------------------------------

    if logits.dtype != torch.float32:
        raise RuntimeError(
            "Incorrect logits dtype. "
            f"Expected torch.float32, received {logits.dtype}."
        )

    # --------------------------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------------------------

    probability_sum = probabilities.sum(dim=1)

    if not torch.allclose(
        probability_sum,
        torch.ones_like(probability_sum),
        atol=1e-5,
    ):
        raise RuntimeError("Classification probabilities do not sum to 1.")

    if not torch.isfinite(logits).all():
        raise RuntimeError("Logits contain NaN or Inf values.")

    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Probabilities contain NaN or Inf values.")

    # --------------------------------------------------------------------------
    # CUDA memory
    # --------------------------------------------------------------------------

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / (1024**2)

        reserved = torch.cuda.memory_reserved() / (1024**2)

        logger.info(
            "CUDA Allocated Memory : %.2f MB",
            allocated,
        )

        logger.info(
            "CUDA Reserved Memory  : %.2f MB",
            reserved,
        )

    # --------------------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Brain Tumor Classifier Test Passed")
    print("=" * 70)

    print(f"Input Shape        : {tuple(features.shape)}")

    print(f"Logits Shape       : {tuple(logits.shape)}")

    print(f"Probability Shape  : {tuple(probabilities.shape)}")

    print(f"Prediction Shape   : {tuple(prediction.shape)}")

    print(f"Predicted Class    : {prediction.item()}")

    print(f"Parameters         : {count_parameters(model):,}")

    print(f"Device             : {logits.device}")

    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    test_model()
