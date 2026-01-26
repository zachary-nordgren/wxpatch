"""Model configuration classes for weather imputation.

This module defines Pydantic models for configuring imputation models,
including classical methods (linear, spline, MICE) and deep learning methods (SAITS, CSDI).
"""

from typing import Literal

from pydantic import Field, model_validator

from weather_imputation.config.base import BaseConfig


class ModelConfig(BaseConfig):
    """Base configuration for imputation models.

    All model-specific configurations inherit from this base class.

    Attributes:
        model_type: Type of imputation model
        n_variables: Number of weather variables to impute
    """

    model_type: Literal["linear", "spline", "mice", "saits", "csdi"] = Field(
        ..., description="Type of imputation model"
    )
    n_variables: int = Field(
        default=6,
        ge=1,
        description="Number of variables to impute (Tier 1 default: 6)",
    )


# ============================================================================
# Classical Methods
# ============================================================================


class LinearInterpolationConfig(ModelConfig):
    """Configuration for linear interpolation baseline.

    Simple linear interpolation between observed values.
    No trainable parameters.

    Example:
        >>> config = LinearInterpolationConfig()
        >>> config.model_type
        'linear'
    """

    model_type: Literal["linear"] = Field(
        default="linear", description="Model type (always 'linear')"
    )
    max_gap_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum gap size for interpolation (None = no limit)",
    )


class SplineInterpolationConfig(ModelConfig):
    """Configuration for spline interpolation baseline.

    Uses Akima spline interpolation for smooth curves through observed values.
    No trainable parameters.

    Example:
        >>> config = SplineInterpolationConfig(spline_order=3)
    """

    model_type: Literal["spline"] = Field(
        default="spline", description="Model type (always 'spline')"
    )
    spline_order: int = Field(
        default=3, ge=1, le=5, description="Spline polynomial order (1-5)"
    )
    max_gap_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum gap size for interpolation (None = no limit)",
    )


class MICEConfig(ModelConfig):
    """Configuration for MICE (Multiple Imputation by Chained Equations).

    Iteratively imputes variables using predictive models conditioned on other variables.

    Example:
        >>> config = MICEConfig(n_iterations=10, n_imputations=5)
    """

    model_type: Literal["mice"] = Field(
        default="mice", description="Model type (always 'mice')"
    )
    n_iterations: int = Field(
        default=10,
        ge=1,
        description="Number of MICE iterations per imputation",
    )
    n_imputations: int = Field(
        default=5,
        ge=1,
        description="Number of multiple imputations to generate",
    )
    predictor_method: Literal["bayesian_ridge", "random_forest", "linear"] = Field(
        default="bayesian_ridge",
        description="Predictor model for each variable",
    )


# ============================================================================
# Deep Learning Methods
# ============================================================================


class SAITSConfig(ModelConfig):
    """Configuration for SAITS (Self-Attention-based Imputation for Time Series).

    Transformer-based model with diagonally-masked self-attention (DMSA)
    and joint MIT (Masked Imputation Task) + ORT (Observed Reconstruction Task) loss.

    Example:
        >>> config = SAITSConfig(
        ...     n_layers=2,
        ...     n_heads=4,
        ...     d_model=128,
        ...     mit_weight=1.0,
        ...     ort_weight=1.0
        ... )

    References:
        Du et al. (2023) "SAITS: Self-Attention-based Imputation for Time Series"
        arXiv:2202.08516
    """

    model_type: Literal["saits"] = Field(
        default="saits", description="Model type (always 'saits')"
    )

    # Architecture hyperparameters
    n_layers: int = Field(
        default=2, ge=1, le=8, description="Number of DMSA encoder layers"
    )
    n_heads: int = Field(
        default=4, ge=1, le=16, description="Number of attention heads"
    )
    d_model: int = Field(
        default=128,
        ge=32,
        le=1024,
        description="Model dimension (embedding size)",
    )
    d_ff: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Feedforward network dimension",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Dropout probability"
    )

    # Loss weights
    mit_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for Masked Imputation Task (MIT) loss",
    )
    ort_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for Observed Reconstruction Task (ORT) loss",
    )

    # Position encoding
    use_learnable_position_encoding: bool = Field(
        default=True,
        description="Use learnable position encoding (vs sinusoidal)",
    )
    max_seq_len: int = Field(
        default=512,
        ge=1,
        description="Maximum sequence length for position encoding",
    )

    # Weather-specific adaptations
    circular_wind_encoding: bool = Field(
        default=True,
        description="Encode wind direction as sin/cos components",
    )
    use_station_metadata: bool = Field(
        default=False,
        description="Condition on station metadata (lat, lon, elevation)",
    )

    @model_validator(mode="after")
    def validate_d_model_divisible_by_n_heads(self) -> "SAITSConfig":
        """Validate that d_model is divisible by n_heads.

        Returns:
            Self for chaining

        Raises:
            ValueError: If d_model not divisible by n_heads
        """
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self


class CSDIConfig(ModelConfig):
    """Configuration for CSDI (Conditional Score-based Diffusion Models for Imputation).

    Diffusion-based probabilistic model for time series imputation with uncertainty quantification.
    Generates multiple samples from the posterior distribution of missing values.

    Example:
        >>> config = CSDIConfig(
        ...     n_diffusion_steps=50,
        ...     n_samples=10,
        ...     noise_schedule="cosine"
        ... )

    References:
        Tashiro et al. (2021) "CSDI: Conditional Score-based Diffusion Models
        for Probabilistic Time Series Imputation" NeurIPS
        arXiv:2107.03502
    """

    model_type: Literal["csdi"] = Field(
        default="csdi", description="Model type (always 'csdi')"
    )

    # Diffusion process hyperparameters
    n_diffusion_steps: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Number of diffusion steps (T)",
    )
    noise_schedule: Literal["linear", "cosine", "quadratic"] = Field(
        default="cosine",
        description="Noise schedule for forward diffusion",
    )
    beta_start: float = Field(
        default=0.0001,
        ge=0.0,
        le=1.0,
        description="Starting beta for linear schedule",
    )
    beta_end: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Ending beta for linear schedule",
    )

    # Denoising network architecture (transformer-based)
    n_layers: int = Field(
        default=4, ge=1, le=12, description="Number of transformer layers"
    )
    n_heads: int = Field(
        default=4, ge=1, le=16, description="Number of attention heads"
    )
    d_model: int = Field(
        default=128,
        ge=32,
        le=1024,
        description="Model dimension (embedding size)",
    )
    d_ff: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Feedforward network dimension",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Dropout probability"
    )

    # Sampling hyperparameters
    n_samples: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of samples to generate for uncertainty quantification",
    )
    sampling_strategy: Literal["ddpm", "ddim"] = Field(
        default="ddpm",
        description="Sampling strategy (DDPM=full diffusion, DDIM=faster deterministic)",
    )
    ddim_steps: int | None = Field(
        default=None,
        ge=1,
        description="Number of DDIM steps (if using DDIM sampling)",
    )

    # Conditioning
    use_time_embedding: bool = Field(
        default=True,
        description="Use sinusoidal time embedding for diffusion timestep",
    )
    use_station_metadata: bool = Field(
        default=False,
        description="Condition on station metadata (lat, lon, elevation)",
    )

    @model_validator(mode="after")
    def validate_d_model_and_beta(self) -> "CSDIConfig":
        """Validate d_model divisibility and beta range.

        Returns:
            Self for chaining

        Raises:
            ValueError: If d_model not divisible by n_heads or beta_end <= beta_start
        """
        # Validate d_model divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        # Validate beta_end > beta_start
        if self.beta_end <= self.beta_start:
            raise ValueError(
                f"beta_end ({self.beta_end}) must be greater than beta_start ({self.beta_start})"
            )

        return self
