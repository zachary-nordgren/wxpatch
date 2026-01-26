"""MICE (Multiple Imputation by Chained Equations) imputation method.

This module implements MICE for imputing missing values in multivariate time series.
MICE iteratively imputes each variable using regression models conditioned on
other variables. It can generate multiple imputations for uncertainty quantification.

References:
    van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation
    by Chained Equations in R. Journal of Statistical Software, 45(3), 1-67.
"""

import json
from pathlib import Path

import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from torch import Tensor
from torch.utils.data import DataLoader

from weather_imputation.models.base import BaseImputer


class MICEImputer(BaseImputer):
    """Impute missing values using MICE (Multiple Imputation by Chained Equations).

    MICE is an iterative imputation method that models each variable with missing
    values as a function of other variables. The algorithm cycles through variables
    with missing values, imputing them using predictions from other variables.

    The method operates on PyTorch tensors with shape (N, T, V) where:
    - N is the batch/sample dimension
    - T is the time dimension
    - V is the variable dimension

    For time series, MICE treats each timestep as an observation and uses the
    variable dimension as features. This is suitable for multivariate imputation
    where variables are correlated.

    Suitable for:
    - Multivariate time series with correlations between variables
    - Missing values scattered across time and variables
    - Datasets where variable relationships can be modeled

    Not suitable for:
    - Univariate time series (no cross-variable information)
    - Data with strong temporal dependencies (MICE ignores time structure)
    - Very long sequences (memory intensive)

    Attributes:
        name: "MICE"
        n_iterations: Number of imputation cycles per dataset.
        n_imputations: Number of complete imputations to generate.
        predictor_method: Regression method ("bayesian_ridge", "random_forest", "linear").

    Example:
        >>> imputer = MICEImputer(n_iterations=10, n_imputations=5)
        >>> imputer.fit(train_loader)
        >>> imputed = imputer.impute(observed, mask)  # Returns mean of imputations

    Note:
        - MICE generates multiple imputations but impute() returns their mean
        - For uncertainty quantification, use generate_imputations() to get all samples
        - Training data is used to learn variable relationships
        - Larger n_iterations improves convergence but increases computation time
    """

    def __init__(
        self,
        n_iterations: int = 10,
        n_imputations: int = 5,
        predictor_method: str = "bayesian_ridge",
        random_state: int | None = None,
    ):
        """Initialize the MICE imputer.

        Args:
            n_iterations: Number of imputation cycles. More iterations improve
                convergence but increase computation time. Default: 10.
            n_imputations: Number of complete imputations to generate.
                Multiple imputations enable uncertainty quantification. Default: 5.
            predictor_method: Regression method for imputing each variable.
                Options: "bayesian_ridge" (default), "random_forest", "linear".
            random_state: Random seed for reproducibility. If None, results may vary.
        """
        super().__init__(name="MICE")
        self.n_iterations = n_iterations
        self.n_imputations = n_imputations
        self.predictor_method = predictor_method
        self.random_state = random_state

        self._hyperparameters = {
            "n_iterations": n_iterations,
            "n_imputations": n_imputations,
            "predictor_method": predictor_method,
            "random_state": random_state,
        }

        # Will be initialized during fit
        self._imputers = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Fit the MICE imputer using training data.

        MICE learns the relationships between variables from the training data.
        This information is used to generate predictions during imputation.

        Args:
            train_loader: Training data with observed values and masks.
            val_loader: Validation data (unused for MICE).
        """
        # Collect all training data
        all_observed = []
        all_masks = []

        for batch in train_loader:
            if isinstance(batch, dict):
                observed = batch["observed"]
                mask = batch["mask"]
            else:
                observed, mask = batch

            all_observed.append(observed)
            all_masks.append(mask)

        # Concatenate all batches
        train_observed = torch.cat(all_observed, dim=0)  # (N_total, T, V)
        train_mask = torch.cat(all_masks, dim=0)  # (N_total, T, V)

        # Reshape to (N_total * T, V) for sklearn
        # Each timestep becomes an observation
        N, T, V = train_observed.shape
        X_train = train_observed.reshape(N * T, V).cpu().numpy()
        mask_train = train_mask.reshape(N * T, V).cpu().numpy()

        # Create missing value mask for sklearn (True=missing, opposite of our convention)
        sklearn_mask = ~mask_train

        # Set missing values to NaN for sklearn
        X_train[sklearn_mask] = float("nan")

        # Create multiple IterativeImputers for multiple imputations
        self._imputers = []
        for i in range(self.n_imputations):
            estimator = self._get_predictor()
            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=self.n_iterations,
                random_state=None if self.random_state is None else self.random_state + i,
                initial_strategy="mean",
                imputation_order="ascending",
            )
            # Fit this imputer
            imputer.fit(X_train)
            self._imputers.append(imputer)

        self._is_fitted = True

    def _get_predictor(self):
        """Get the predictor model based on predictor_method.

        Returns:
            Sklearn estimator for IterativeImputer.
        """
        if self.predictor_method == "bayesian_ridge":
            return BayesianRidge()
        elif self.predictor_method == "random_forest":
            return RandomForestRegressor(
                n_estimators=10,  # Keep small for speed
                max_depth=10,
                random_state=self.random_state,
            )
        elif self.predictor_method == "linear":
            return LinearRegression()
        else:
            raise ValueError(
                f"Unknown predictor_method: {self.predictor_method}. "
                f"Choose from: bayesian_ridge, random_forest, linear"
            )

    def impute(self, observed: Tensor, mask: Tensor) -> Tensor:
        """Impute missing values using MICE.

        Generates n_imputations complete datasets and returns their mean.
        This provides a single imputed dataset while leveraging multiple
        imputations internally.

        Args:
            observed: (N, T, V) tensor with observed values.
                Missing positions can contain any value (will be replaced).
            mask: (N, T, V) boolean tensor indicating which values are observed.
                True = observed, False = missing (to be imputed).

        Returns:
            (N, T, V) tensor with imputed values. Observed positions are
            preserved; missing positions are filled with mean of multiple imputations.

        Raises:
            RuntimeError: If imputer has not been fitted.
            ValueError: If input tensors have invalid shape or type.
        """
        self._check_fitted()
        self._validate_inputs(observed, mask)

        # Generate all imputations and average them
        imputations = self.generate_imputations(observed, mask)
        # Average across the imputation dimension (dim=0)
        mean_imputation = imputations.mean(dim=0)

        return mean_imputation

    def generate_imputations(self, observed: Tensor, mask: Tensor) -> Tensor:
        """Generate multiple imputations for uncertainty quantification.

        Args:
            observed: (N, T, V) tensor with observed values.
            mask: (N, T, V) boolean tensor (True=observed, False=missing).

        Returns:
            (M, N, T, V) tensor with M imputations, where M = n_imputations.

        Raises:
            RuntimeError: If imputer has not been fitted.
        """
        self._check_fitted()
        self._validate_inputs(observed, mask)

        N, T, V = observed.shape
        device = observed.device
        dtype = observed.dtype

        # Reshape to (N*T, V) for sklearn
        X = observed.reshape(N * T, V).cpu().numpy()
        mask_flat = mask.reshape(N * T, V).cpu().numpy()

        # Create sklearn mask (True=missing)
        sklearn_mask = ~mask_flat

        # Set missing values to NaN
        X[sklearn_mask] = float("nan")

        # Generate imputations
        imputations = []
        for imputer in self._imputers:
            # Transform (impute) using this fitted imputer
            X_imputed = imputer.transform(X)

            # Convert back to torch and reshape
            imputed_tensor = torch.from_numpy(X_imputed).to(device=device, dtype=dtype)
            imputed_tensor = imputed_tensor.reshape(N, T, V)

            # Preserve observed values (only fill missing positions)
            result = observed.clone()
            result[~mask] = imputed_tensor[~mask]

            imputations.append(result)

        # Stack into (M, N, T, V) where M = n_imputations
        return torch.stack(imputations, dim=0)

    def save(self, path: Path) -> None:
        """Save model state to disk.

        Saves MICE imputer state including fitted sklearn IterativeImputers.

        Args:
            path: Directory where model should be saved.
        """
        import pickle

        self._save_metadata(path)

        # Save hyperparameters as JSON
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._hyperparameters, f, indent=2)

        # Save fitted imputers as pickle
        if self._imputers is not None:
            imputers_path = path / "imputers.pkl"
            with open(imputers_path, "wb") as f:
                pickle.dump(self._imputers, f)

    def load(self, path: Path) -> None:
        """Load model state from disk.

        Args:
            path: Directory where model was saved.

        Raises:
            FileNotFoundError: If the specified path doesn't exist.
        """
        import pickle

        metadata = self._load_metadata(path)

        # Restore state
        self._is_fitted = metadata["is_fitted"]
        self._hyperparameters = metadata["hyperparameters"]
        self.n_iterations = self._hyperparameters["n_iterations"]
        self.n_imputations = self._hyperparameters["n_imputations"]
        self.predictor_method = self._hyperparameters["predictor_method"]
        self.random_state = self._hyperparameters.get("random_state")

        # Load fitted imputers
        if self._is_fitted:
            imputers_path = path / "imputers.pkl"
            if not imputers_path.exists():
                raise FileNotFoundError(f"Imputers file not found: {imputers_path}")
            with open(imputers_path, "rb") as f:
                self._imputers = pickle.load(f)
