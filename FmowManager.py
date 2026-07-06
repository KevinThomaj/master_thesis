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
from sklearn.model_selection import train_test_split

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

    def prepare_data_splits(self, num_classes=25, pre_samples=2200, post_samples=2300):
        print("\n--- STEP 1: Total Dataset ---")
        total_df = self.get_dataset()

        print("\n--- STEP 2: Divide into preDF (2002-2013) and postDF (2016-2017) ---")
        preDF, postDF = self.divide(total_df)

        # Find the most popular classes in preDF
        top_classes = preDF['category'].value_counts().nlargest(num_classes).index.tolist()
        print(f"Top {num_classes} classes identified: {top_classes}")

        # Filter preDF to only include these top classes
        preDF_top = preDF[preDF['category'].isin(top_classes)].copy()

        # Create the mapping required for the TorchDataset
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(top_classes)}

        print(f"\n--- STEP 4: Sample {pre_samples} images per class in preDF ---")
        preDF_sampled = self.sample_dataset(
            preDF_top,
            category_col='category',
            samples_per_class=pre_samples,
            random_state=42
        )

        print(f"\n--- STEP 5: Sample {post_samples} images per class in postDF, shuffle, create/order by concepts ---")
        # Filter postDF to include the SAME classes
        postDF_top = postDF[postDF['category'].isin(top_classes)].copy()

        postDF_sampled = self.sample_dataset(
            postDF_top,
            category_col='category',
            samples_per_class=post_samples,
            random_state=42
        )

        return preDF_sampled, postDF_sampled, top_classes, class_to_idx, preDF

    def prepare_streaming_concepts(self, postDF_sampled, top_classes, test_size_per_concept=100, config_id=1, recurrent_concept=None):
        # Shuffle and prepare the extended embeddings for the streaming experiments
        postDF_sampled_final = postDF_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

        if config_id == 1:
            # Config 1: Modulo grouping
            dummy_concept_mapping = {cls: (f"Concept_{(i % 5)}") for i, cls in enumerate(top_classes)}
        elif config_id == 2:
            # Config 2: Sequential grouping (blocks of 5)
            dummy_concept_mapping = {cls: (f"Concept_{((i // 5) % 5)}") for i, cls in enumerate(top_classes)}
        elif config_id == 3:
            # Config 3: Pseudo-random grouping
            import random
            rng = random.Random(42)
            shuffled_classes = top_classes.copy()
            rng.shuffle(shuffled_classes)
            dummy_concept_mapping = {cls: (f"Concept_{(i % 5)}") for i, cls in enumerate(shuffled_classes)}
        else:
            dummy_concept_mapping = {cls: (f"Concept_{(i % 5)}") for i, cls in enumerate(top_classes)}

        postDF_sampled_final = self.create_concepts(postDF_sampled_final, dummy_concept_mapping)

        # Order by the newly created concepts
        postDF_sampled_final = postDF_sampled_final.sort_values(by='concept').reset_index(drop=True)

        # Split into stream_df and test_dict
        stream_parts = []
        test_dict = {}
        recurrent_part = None
        
        # Group by concept and split
        for concept, group in postDF_sampled_final.groupby('concept', sort=False):
            stream_part, test_part = train_test_split(
                group, 
                test_size=test_size_per_concept, 
                stratify=group['category'], 
                random_state=42
            )
                
            stream_parts.append(stream_part)
            if len(test_part) > 0:
                test_dict[concept] = test_part
                
            if recurrent_concept is not None and concept == recurrent_concept:
                recurrent_part = stream_part.copy()
                
        if recurrent_part is not None:
            recurrent_part['concept'] = f"{recurrent_concept}_recurrent"
            print(f"Adding recurrent concept {recurrent_concept} to the end of the stream.")
            stream_parts.append(recurrent_part)
                
        # Re-concatenate the stream parts
        stream_df = pd.concat(stream_parts).reset_index(drop=True)

        return stream_df, test_dict
