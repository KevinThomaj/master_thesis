from torch.utils.data import Dataset
import torch


class FmowTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for the WILDS Functional Map of the World (FMoW) dataset.

    Handles on-the-fly image loading via `FmowManager`, applying torchvision transformations,
    mapping category string labels to continuous integers, and optionally loading precomputed
    foundation model feature embeddings for efficient feature distillation without keeping the
    large teacher model in VRAM during training.

    Args:
        df (pd.DataFrame): Dataframe containing metadata ('image_index', 'category', and optionally 'embedding').
        manager (FmowManager): Instance of FmowManager providing the `get_input(image_idx)` method.
        class_to_idx (dict): Mapping from category string names to integer class indices.
        transform (callable, optional): Torchvision transform pipeline to apply to raw PIL images.
        use_embeddings (bool): If True, returns (image, teacher_embedding, label) for distillation;
                               if False, returns (image, label).
    """
    def __init__(self, df, manager, class_to_idx, transform=None, use_embeddings=False):
        self.df = df.reset_index(drop=True)
        self.manager = manager
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_embeddings = use_embeddings  # Flag to toggle teacher feature distillation

    def __len__(self):
        """Returns the total number of samples in the dataset split."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads the image from disk, applies transformations, and retrieves labels (and embeddings).

        Args:
            idx (int): Sample index.

        Returns:
            tuple:
                - If use_embeddings=False: (image_tensor, category_int)
                - If use_embeddings=True: (image_tensor, teacher_embedding_tensor, category_int)
        """
        row = self.df.iloc[idx]

        # Safely get the image index whether it's the raw DF or the embeddings DF
        image_idx = row['image_index']

        # 1. Load and transform image from disk via manager
        image = self.manager.get_input(image_idx)
        if self.transform:
            image = self.transform(image)

        # 2. Translate string category to integer index
        category_str = row['category']
        category_int = self.class_to_idx[category_str]

        # 3. Conditionally return the precomputed foundation model embedding
        if self.use_embeddings:
            if 'embedding' not in row:
                raise ValueError("use_embeddings is True, but 'embedding' column missing from dataframe.")

            # Convert embedding list/array back to a PyTorch tensor
            teacher_embedding = torch.tensor(row['embedding'], dtype=torch.float32)
            return image, teacher_embedding, category_int

        # Default return (No distillation)
        return image, category_int