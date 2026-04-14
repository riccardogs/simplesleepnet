import json
import logging

logger = logging.getLogger(__name__)

def validate_config(config):
    schema = {
        "experiment_num": int,
        "seed": int,
        "results_folder": str,
        "num_workers": int,
        "dataset": {
            "dset_path": str,
            "max_files": int,
        },
        "augmentations": {
            "*": {  # '*' indicates any augmentation name
                "p": (float, int),
                # ...additional augmentation parameters...
            }
        },
        "pretraining_params": {
            "batch_size": int,
            "latent_dim": int,
            "learning_rate": (float, int),
            "max_epochs": int,
            "check_interval": int,
            "min_improvement": (float, int),
            "temperature": (float, int),
            "best_model_pth": str,
            "dropout_rate": (float, int),
        },
        "latent_space_params": {
            "umap_enabled": bool,
            "pca_enabled": bool,
            "tsne_enabled": bool,
            "visualize": bool,
            "compute_metrics": bool,
            "n_clusters": int,
            "output_image_dir": str,
            "output_metrics_dir": str,
            "visualization_fraction": float,
        },
        "sup_training_params": {
            "learning_rate": (float, int),
            "max_epochs": int,
            "check_interval": int,
            "min_improvement": (float, int),
            "dropout_rate": (float, int),
            "best_model_pth": str,
        }
    }

    validate_section(config, schema)

    logger.info("Configuration validation passed successfully.")

def validate_section(config_section, schema, path=''):
    """
    Recursively validate a configuration section against a schema.

    Args:
        config_section (dict): The configuration section to validate.
        schema (dict): The schema definition.
        path (str): The path to the current section (used for error messages).

    Raises:
        ValueError: If any required key is missing.
        TypeError: If any value has an incorrect type.
    """
    for key, expected in schema.items():
        current_path = f"{path}.{key}" if path else key

        if key == "*":
            # Wildcard key: validate all subkeys with the same schema
            for subkey, subsection in config_section.items():
                validate_section(subsection, expected, f"{current_path}.{subkey}")
        else:
            if key not in config_section:
                logger.debug(f"Missing required config key: '{current_path}'")
                raise ValueError(f"Missing required config key: '{current_path}'")
            value = config_section[key]

            if isinstance(expected, dict):
                if not isinstance(value, dict):
                    logger.debug(f"'{current_path}' should be a dict, got {type(value).__name__}")
                    raise TypeError(f"'{current_path}' should be a dict, got {type(value).__name__}")
                validate_section(value, expected, current_path)
            else:
                validate_type(value, expected, current_path)
                # Additional checks for specific keys
                if current_path.endswith("dropout_rate") or current_path.endswith(".p"):
                    if not (0.0 <= value <= 1.0):
                        logger.debug(f"'{current_path}' should be between 0.0 and 1.0, got {value}")
                        raise ValueError(f"'{current_path}' should be between 0.0 and 1.0, got {value}")

def validate_type(value, expected_type, key_name):
    """
    Check if the value is of the expected type.

    Args:
        value: The value to check.
        expected_type (type or tuple of types): The expected type(s) of the value.
        key_name (str): The name of the key being checked.

    Raises:
        TypeError: If the value is not of the expected type.
    """
    if not isinstance(value, expected_type):
        expected_type_names = (
            expected_type.__name__ if isinstance(expected_type, type)
            else ', '.join(t.__name__ for t in expected_type)
        )
        logger.debug(f"'{key_name}' should be of type {expected_type_names}, got {type(value).__name__}")
        raise TypeError(f"'{key_name}' should be of type {expected_type_names}, got {type(value).__name__}")