import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from torch.utils.data import DataLoader
from wilds import get_dataset
from tqdm import tqdm

from FmowTorchDataset import FmowTorchDataset


class FmowManager:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataset(self):
        # Datetime patching for ISO8601
        import pandas.core.tools.datetimes as _pd_dt_module
        _true_orig_to_datetime = _pd_dt_module.to_datetime

        def _patched_to_datetime(*args, **kwargs):
            kwargs['format'] = 'ISO8601'
            return _true_orig_to_datetime(*args, **kwargs)

        pd.to_datetime = _patched_to_datetime
        dataset = get_dataset(dataset="fmow", download=True)
        pd.to_datetime = _true_orig_to_datetime

        df = dataset.metadata.copy()

        # The DataFrame index matches the WILDS image ID perfectly.
        df['image_index'] = np.arange(len(df))

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        df['year_extracted'] = df['timestamp'].dt.year

        region_names_list = dataset.metadata_map['region']
        df['region_names'] = df['region'].map(lambda x: region_names_list[int(x)])

        return df

    def sample_dataset(self, df, category_col='category', samples_per_class=32, random_state=42):
        """Samples a balanced subset from the main dataframe."""
        filtered_df = df.copy()
        print(f"--- Sampling {samples_per_class} images per class ---")
        sample_indices = []

        for _, group in filtered_df.groupby(category_col):
            n_to_sample = min(len(group), samples_per_class)
            sampled_group = group.sample(n=n_to_sample, random_state=random_state)
            # We no longer need to append to a custom list, we just grab the pre-made image_index
            sample_indices.extend(sampled_group['image_index'].tolist())

        # Filter the original dataframe to keep only sampled indices
        df_sample = df[df['image_index'].isin(sample_indices)].copy()
        return df_sample

    def get_input(self, idx):
        """
        Returns an RGB image for a given idx, ensuring file handles are safely closed.
        """
        file_path = Path('./data/' + 'fmow_v1.1') / 'images' / f'rgb_img_{idx}.png'
        with Image.open(file_path) as img:
            return img.convert('RGB')


    #Think if this should be an operation in the FoundationModel class
    def get_embeddings(self, df_sample, model, transform_fn, batch_size=128,
                       save_path=None):
        """
        Extracts embeddings using DataLoader for parallel I/O processing.
        """
        if save_path and os.path.exists(save_path):
            print(f"Found saved dataset at {save_path}. Loading from disk...")
            return pd.read_parquet(save_path)

        print("Extracting features...")
        model = model.to(self.device)
        model.eval()

        indices = df_sample['image_index'].tolist()

        # 1. Initialize our Unified Dataset in extraction mode
        embed_dataset = FmowTorchDataset(
            df=df_sample,
            manager=self,
            class_to_idx={cat: 0 for cat in df_sample['category'].unique()},
            transform=transform_fn
        )

        embed_loader = DataLoader(
            embed_dataset,
            batch_size=batch_size,
            shuffle=False,           # Must be False to match the dataframe order
            num_workers=4, # Parallel image loading
            pin_memory=True
        )

        all_embeddings = []

        with torch.no_grad():
            # 2. The Clean Loop: DataLoader yields purely batch_imgs
            for batch_imgs,labels in tqdm(embed_loader, desc="Extracting"):

                batch_imgs = batch_imgs.to(self.device)
                features = model(batch_imgs)
                all_embeddings.append(features.cpu().numpy())

                # Memory Cleanup
                del batch_imgs, features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 3. Reconstruct into a Matrix
        embeddings_matrix = np.concatenate(all_embeddings, axis=0)

        # 4. Create the final DataFrame
        df_dataset = pd.DataFrame({
            'image_index': indices,
            'category': df_sample['category'].values,
            'embedding': list(embeddings_matrix)
        })

        if save_path:
            print("Saving new embeddings dataset to disk...")
            df_dataset.to_parquet(save_path, engine='pyarrow')

        return df_dataset

    # 4. IMPLEMENTED DIVIDE METHOD
    def divide(self, df=None):

        #Splits the dataset into two distinct temporal groups.

        print("Splitting dataset temporally...")

        # Group 1: 2002 to 2013
        mask_early = (df['year_extracted'] >= 2002) & (df['year_extracted'] <= 2013)
        df_early = df[mask_early].copy()

        # Group 2: 2016 to 2017
        mask_late = (df['year_extracted'] >= 2016) & (df['year_extracted'] <= 2017)
        df_late = df[mask_late].copy()

        print(f"-> 2002-2013 Dataset: {len(df_early)} images")
        print(f"-> 2016-2017 Dataset: {len(df_late)} images")

        return df_early, df_late

    def create_concepts(self, df,class_to_concept_mapping):
        # Modify the dataset dataframe to include the new labels
        df['concept'] = df['category'].map(class_to_concept_mapping)
        print("Distribution of images across the new macro-classes:")
        print(df['concept'].value_counts())
        return df

    def cluster_by_class(self, df, num_clusters=6, method='ward', metric='euclidean'):
        """Takes the unified dataframe, extracts embeddings natively, and performs clustering."""
        print("Clustering embeddings...")

        # Unstack the list of embeddings back into a 2D numpy array
        embeddings_matrix = np.stack(df['embedding'].values)

        df_emb = pd.DataFrame(embeddings_matrix)
        df_emb['category_name'] = df['category'].values

        class_centroids = df_emb.groupby('category_name').mean()
        name_classes = list(class_centroids.index)

        linkage_data = linkage(class_centroids, method=method, metric=metric)
        cluster_labels = fcluster(linkage_data, t=num_clusters, criterion='maxclust')

        class_to_macro_mapping = dict(zip(name_classes, cluster_labels))

        # Modify the dataset dataframe to include the new labels
        df['concept'] = df['category'].map(class_to_macro_mapping)

        print("Distribution of images across the new macro-classes:")
        print(df['concept'].value_counts())

        plt.figure(figsize=(20, 10))
        dendrogram(linkage_data, labels=name_classes, leaf_rotation=90, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
        plt.xlabel('Category Name', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.tight_layout()
        plt.show()

        return df


#1)Total Dataset
#2)Divide in preDF(2002-2013) and postDF(2016-2017)
#3)Sample 1100 images per class in preDF (This is the training + Validation data) using 25 most popular classes. Use SampleDataset for this
#4)Sample 500 images per class using (25 most popular classes chosen in previous step) in postDF, shuffle, create concepts and then order by concepts
#5)Pretraining of FM using all preDF
#6)Pretraining of resnet18 using output of 3)
#7)Online learning of Resnet18 using output of 4)
#8)Calculate embeddings using Foundation Model(FM) for preDF and output of 4)
