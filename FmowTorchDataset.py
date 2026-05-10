from torch.utils.data import Dataset
import torch


class FmowTorchDataset(Dataset):
    def __init__(self, df, manager, class_to_idx, transform=None, use_embeddings=False):
        self.df = df.reset_index(drop=True)
        self.manager = manager
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_embeddings = use_embeddings  # New flag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Safely get the image index whether it's the raw DF or the embeddings DF
        image_idx = row['image_index']

        # 1. Load and transform image
        image = self.manager.get_input(image_idx)
        if self.transform:
            image = self.transform(image)

        # 2. Translate string category to integer
        category_str = row['category']
        category_int = self.class_to_idx[category_str]

        # 3. Conditionally return the embedding
        if self.use_embeddings:
            if 'embedding' not in row:
                raise ValueError("use_embeddings is True, but 'embedding' column missing from dataframe.")

            # Convert embedding list/array back to a PyTorch tensor
            teacher_embedding = torch.tensor(row['embedding'], dtype=torch.float32)
            return image, teacher_embedding, category_int

        # Default return (No distillation)
        return image, category_int