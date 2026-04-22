
import gc
import os

import torch

from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from wilds import get_dataset

# Use sklearn's LabelEncoder for tabular dataframe columns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class ClusteringPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataset(self, path="/mnt/windows/Users/kevin/Downloads/"):
        import pandas.core.tools.datetimes as _pd_dt_module
        _true_orig_to_datetime = _pd_dt_module.to_datetime

        def _patched_to_datetime(*args, **kwargs):
            kwargs['format'] = 'ISO8601'
            return _true_orig_to_datetime(*args, **kwargs)

        pd.to_datetime = _patched_to_datetime
        # never use dataset, but use the metadata(it contains all IMAGES)
        dataset = get_dataset(dataset="fmow", root_dir=path, download=False)
        pd.to_datetime = _true_orig_to_datetime

        df = dataset.metadata.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        df['year_extracted'] = df['timestamp'].dt.year

        region_names_list = dataset.metadata_map['region']
        df['region_names'] = df['region'].map(lambda x: region_names_list[int(x)])

        return df

    def sample_dataset(self, df, category_col='category', samples_per_class=32, random_state=42,target_year=2016,country_code="USA"):
        """Samples a balanced subset from the main dataframe."""

        #filtering based on year and state, just to simplify
        #filtered_df = df[(df['year_extracted'] == target_year) & (df['country_code'] == country_code)].copy()
        filtered_df = df.copy()
        print(f"--- Sampling {samples_per_class} images per class ---")
        sample_indices = []

        for _, group in filtered_df.groupby(category_col):
            n_to_sample = min(len(group), samples_per_class)
            sampled_group = group.sample(n=n_to_sample, random_state=random_state)
            sample_indices.extend(sampled_group.index.tolist())

        df_sample = df.loc[sample_indices].copy()
        df_sample['index'] = sample_indices
        return df_sample

    def get_embeddings(self, df_sample, model, get_input_fn, transform_fn, batch_size=128,
                       save_path=None):
        """
        Extracts embeddings and creates a BRAND NEW, lean DataFrame
        that acts as your standalone dataset for clustering and training.
        """
        if save_path and os.path.exists(save_path):
            print(f"Found saved dataset at {save_path}. Loading from disk...")
            return pd.read_parquet(save_path)

        print("Extracting features...")
        model = model.to(self.device)
        model.eval()

        all_embeddings = []
        indices = df_sample['index'].tolist()

        with torch.no_grad():
            for i in tqdm(range(0, len(indices), batch_size), desc="Extracting"):
                batch_idx = indices[i:i + batch_size]
                batch_imgs = [transform_fn(get_input_fn(idx)) for idx in batch_idx]
                input_tensor = torch.stack(batch_imgs, dim=0).to(self.device)

                features = model(input_tensor)
                all_embeddings.append(features.cpu().numpy())

                # Strict Memory Cleanup
                del input_tensor, features, batch_imgs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        embeddings_matrix = np.concatenate(all_embeddings, axis=0)

        # ---------------------------------------------------------
        # CREATE THE NEW DATASET DATAFRAME
        # ---------------------------------------------------------
        df_dataset = pd.DataFrame({
            'original_index': indices,
            'category': df_sample['category'].values,
            'embedding': list(embeddings_matrix)
        })

        if save_path:
            print("Saving new embeddings dataset to disk...")
            df_dataset.to_parquet(save_path, engine='pyarrow')

        return df_dataset

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
        df['macro_class'] = df['category'].map(class_to_macro_mapping)

        print("Distribution of images across the new macro-classes:")
        print(df['macro_class'].value_counts())

        plt.figure(figsize=(20, 10))
        dendrogram(linkage_data, labels=name_classes, leaf_rotation=90, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
        plt.xlabel('Category Name', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.tight_layout()
        plt.show()

        return df

    def train_offline(self, model, df, criterion, optimizer, target_col='category', epochs=100, batch_size=32,
                      log_interval=5):
        """Builds DataLoaders directly from the unified dataframe and trains the model."""
        print(f"Preparing data for training on target: {target_col}")

        # 1. Prepare Data
        X = np.stack(df['embedding'].values)
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                torch.tensor(y_train, dtype=torch.long)),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                               torch.tensor(y_test, dtype=torch.long)),
                                 batch_size=batch_size, shuffle=False)

        # 2. Train Loop
        model = model.to(self.device)
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(epochs):
            model.train()
            train_loss, correct, total = 0.0, 0, 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            train_acc = 100 * correct / total
            train_loss /= len(train_loader)

            val_loss, val_acc = self._evaluate(model, test_loader, criterion)

            history['train_loss'].append(train_loss)
            history['test_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(val_acc)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        print("\n--- Final Evaluation ---")
        final_test_loss, final_test_acc = self._evaluate(model, test_loader, criterion)
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        return history

    def _evaluate(self, model, loader, criterion):
        """Internal helper for evaluation."""
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                running_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return running_loss / len(loader), 100 * correct / total

    import matplotlib.pyplot as plt
    import math

    def visualize_category_images(self,df, category_name, num_images=5, get_input_fn = None):
        """
        Visualizes a specified number of images from a chosen category.

        Args:
            df: The dataframe containing 'category' and 'index' columns.
            category_name: The string name of the category to visualize.
            num_images: The number of images to display.
            get_input_fn: The function used to load the image based on its index.
        """
        # Filter dataframe for the requested category
        subset = df[df['category'] == category_name]

        if subset.empty:
            print(f"Error: Category '{category_name}' not found in the dataframe.")
            return

        # Ensure we don't request more images than available
        actual_num_images = min(len(subset), num_images)
        if actual_num_images < num_images:
            print(f"Note: Only {actual_num_images} images available for '{category_name}'.")

        # Randomly sample the indices
        sampled_indices = subset.sample(n=actual_num_images, random_state=42)['index'].tolist()

        # Calculate grid dimensions (max 5 columns for readability)
        cols = min(actual_num_images, 5)
        rows = math.ceil(actual_num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        fig.suptitle(f"Category: {category_name} ({actual_num_images} images)", fontsize=16)

        # Flatten axes array for easy iteration, handling 1D and 2D arrays
        if actual_num_images == 1:
            axes = [axes]
        elif rows > 1 or cols > 1:
            axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < actual_num_images:
                idx = sampled_indices[i]
                img = get_input_fn(idx)  # Uses your existing image loading logic
                ax.imshow(img)
                ax.set_title(f"Idx: {idx}")
                ax.axis('off')
            else:
                # Hide empty subplots if the grid isn't perfectly filled
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust title spacing
        plt.show()
