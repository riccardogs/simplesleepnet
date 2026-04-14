import json
import itertools

# Define the base configuration without 'experiment_num' and 'augmentations'
base_config = {
    "seed": 42,
    "results_folder": "results",
    "num_workers": 8,

    "dataset": {
        "dset_path": "./dset/Sleep-EDF-2018/npz/Fpz-Cz",
        "max_files": 200
    },

    "pretraining_params": {
        "batch_size": 1024,
        "temperature": 0.1,
        "dropout_rate": 0.2,
        "latent_dim": 128,
        "learning_rate": 1e-3,
        "max_epochs": 1500,
        "check_interval": 50,
        "min_improvement": 0.005,
        "best_model_pth": "checkpoints/encoder/encoder_"
    },

    "latent_space_params": {
        "umap_enabled": False,
        "pca_enabled": False,
        "tsne_enabled": True,
        "visualize": True,
        "compute_metrics": False,
        "n_clusters": 5,
        "output_image_dir": "latent_space/visualizations",
        "output_metrics_dir": "latent_space/metrics",
        "visualization_fraction": 0.5
    },

    "sup_training_params": {
        "learning_rate": 1e-3,
        "max_epochs": 500,
        "dropout_rate": 0.5,
        "check_interval": 25,
        "min_improvement": 0.005,
        "best_model_pth": "checkpoints/classifier/classifier_"
    }
}

# Define the augmentations and their categories
all_categories = {
    "Amplitude-Based": ["RandomAmplitudeScale", "RandomDCShift", "SignFlip"],
    "Frequency-Based": ["RandomBandStopFilter", "TailoredMixup"],
    "Masking-Cropping": ["RandomZeroMasking", "CutoutResize"],
    "Noise-and-Filtering": ["RandomAdditiveGaussianNoise", "AverageFilter"],
    "Temporal": ["RandomTimeShift", "TimeWarping", "TimeReverse", "Permutation"]
}

# List of all augmentations
all_augmentations_list = [
    "RandomAmplitudeScale",
    "RandomDCShift",
    "SignFlip",
    "RandomBandStopFilter",
    "TailoredMixup",
    "RandomZeroMasking",
    "CutoutResize",
    "RandomAdditiveGaussianNoise",
    "AverageFilter",
    "RandomTimeShift",
    "TimeWarping",
    "TimeReverse",
    "Permutation"
]

augmentations_list = []

categories = {
    "Frequency-Based": ["TailoredMixup"],
    "Masking-Cropping": ["RandomZeroMasking", "CutoutResize"],
    "Temporal": [ "TimeWarping", "Permutation"]
}

# experiment_num = 1

# # Set 1: Single augmentation experiments
# for augmentation in all_augmentations_list:
#     config = base_config.copy()
#     config["experiment_num"] = experiment_num
#     config["augmentations"] = {
#         augmentation: {"p": 1.0}
#     }
#     # Write the config to a JSON file
#     with open(f"config_{experiment_num}.json", "w") as f:
#         json.dump(config, f, indent=2)
#     experiment_num += 1

# # Set 2: Combinations within the same category
# for category_name, augmentations_in_category in all_categories.items():
#     x = len(augmentations_in_category)
#     # Generate combinations of size 2 to x
#     for r in range(2, x + 1):
#         combinations = list(itertools.combinations(augmentations_in_category, r))
#         for combination in combinations:
#             config = base_config.copy()
#             config["experiment_num"] = experiment_num
#             config["augmentations"] = {}
#             for aug in combination:
#                 config["augmentations"][aug] = {"p": 1.0}
#             # Write the config to a JSON file
#             with open(f"config_{experiment_num}.json", "w") as f:
#                 json.dump(config, f, indent=2)
#             experiment_num += 1

# Define category pairs to combine
category_pairs = [
    ("Frequency-Based", "Temporal"),
    ("Temporal", "Masking-Cropping"),
    ("Frequency-Based", "Masking-Cropping")
]

experiment_num = 67
seed=1
# Generate combinations between categories
for cat1_name, cat2_name in category_pairs:
    cat1_augs = categories[cat1_name]
    cat2_augs = categories[cat2_name]
    # Generate all combinations of augmentations in cat1
    x1 = len(cat1_augs)
    for r1 in range(1, x1 + 1):
        combinations_cat1 = list(itertools.combinations(cat1_augs, r1))
        # Generate all combinations of augmentations in cat2
        x2 = len(cat2_augs)
        for r2 in range(1, x2 + 1):
            combinations_cat2 = list(itertools.combinations(cat2_augs, r2))
            # Combine augmentations from both categories
            for comb1 in combinations_cat1:
                for comb2 in combinations_cat2:
                    config = base_config.copy()
                    config["experiment_num"] = experiment_num
                    config["seed"] = seed
                    config["augmentations"] = {}
                    for aug in comb1 + comb2:
                        config["augmentations"][aug] = {"p": 1.0}
                    # Write the config to a JSON file
                    with open(f"config_{experiment_num}.json", "w") as f:
                        json.dump(config, f, indent=2)
                    experiment_num += 1

total_experiments = experiment_num - 67
print(f"Total experiments generated: {total_experiments}")