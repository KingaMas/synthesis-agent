import h5py
import numpy as np
from pymatgen.core import Composition, Structure
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from src import ASSETS_DIR
from src.embedding import MaterialsEmbedding, InputType
from src.schema import Neighbor


class SearchAPI:
    def __init__(
        self,
        input_type: InputType,
        max_neighbors: int = 100,
    ):
        self.featurizer = MaterialsEmbedding(input_type=input_type)
        self.max_neighbors = max_neighbors

        # Load pre-computed MP dataset
        self.mp_data = self._load_mp_data()

        # Set up nearest neighbors model
        self._set_nearest_neighbors_model()

    def _load_mp_data(self):
        if self.featurizer.input_type == InputType.COMPOSITION:
            h5_file = ASSETS_DIR / "embedding" / "mp_dataset_composition_magpie.h5"
        elif self.featurizer.input_type == InputType.STRUCTURE:
            h5_file = ASSETS_DIR / "embedding" / "mp_dataset_structure_mace.h5"
        else:
            raise ValueError("Invalid input type.")
        print(f"Loading MP dataset from {h5_file}")

        with h5py.File(h5_file, "r") as f:
            features = f["features"][:]
            material_ids = f["material_ids"][:].astype("str")
            formulas = f["formulas"][:].astype("str")

        return {
            "features": features,
            "material_ids": material_ids,
            "formulas": formulas,
        }

    def _set_nearest_neighbors_model(self):
        self.scaler = StandardScaler().fit(self.mp_data["features"])
        mp_features_scaled = self.scaler.transform(self.mp_data["features"])
        self.nn_model = NearestNeighbors(
            n_neighbors=self.max_neighbors, metric="euclidean"
        ).fit(mp_features_scaled)

    def fit(
        self,
        features: np.ndarray,
        material_ids: np.ndarray,
        formulas: np.ndarray,
    ) -> None:
        """Rebuild the NearestNeighbors model on an arbitrary subset of data.

        Enables leave-one-out and train/test split evaluation without reloading
        the full HDF5 dataset.

        Args:
            features: Feature matrix of shape (n_materials, n_features).
            material_ids: Array of material ID strings, length n_materials.
            formulas: Array of formula strings, length n_materials.
        """
        self.mp_data = {
            "features": features,
            "material_ids": np.asarray(material_ids),
            "formulas": np.asarray(formulas),
        }
        self._set_nearest_neighbors_model()

    def query(
        self, input_data: Composition | Structure, n_neighbors: int = 10
    ) -> list[Neighbor]:
        input_embedding = self.featurizer.get_embedding(input_data)
        input_embedding_scaled = self.scaler.transform(input_embedding)
        distances, indices = self.nn_model.kneighbors(
            input_embedding_scaled, n_neighbors=n_neighbors
        )
        distances = distances.squeeze()
        indices = indices.squeeze()
        confidences = np.exp(-distances / 0.5)

        # Collect results
        results = []
        for i, idx in enumerate(indices):
            results.append(
                Neighbor(
                    neighbor_index=i,
                    material_id=self.mp_data["material_ids"][idx].item(),
                    formula=self.mp_data["formulas"][idx].item(),
                    distance=distances[i].item(),
                    confidence=confidences[i].item(),
                )
            )
        return results

    def query_with_exclusion(
        self,
        input_data: Composition | Structure,
        exclude_ids: list[str],
        n_neighbors: int = 10,
    ) -> list[Neighbor]:
        """Query KNN while excluding specific material IDs from results.

        Prevents self-retrieval during leave-one-out evaluation.

        Args:
            input_data: Query composition or structure.
            exclude_ids: Material IDs to remove from returned results.
            n_neighbors: Number of neighbors to return after exclusion.
        """
        exclude_set = set(exclude_ids)
        # Over-fetch to have enough after exclusion; cap at index size
        fetch_n = min(n_neighbors + len(exclude_set) + 10, len(self.mp_data["material_ids"]))

        input_embedding = self.featurizer.get_embedding(input_data)
        input_embedding_scaled = self.scaler.transform(input_embedding)
        distances, indices = self.nn_model.kneighbors(
            input_embedding_scaled, n_neighbors=fetch_n
        )
        distances = distances.squeeze()
        indices = indices.squeeze()
        confidences = np.exp(-distances / 0.5)

        results = []
        neighbor_index = 0
        for i, idx in enumerate(indices):
            mid = self.mp_data["material_ids"][idx].item()
            if mid in exclude_set:
                continue
            results.append(
                Neighbor(
                    neighbor_index=neighbor_index,
                    material_id=mid,
                    formula=self.mp_data["formulas"][idx].item(),
                    distance=distances[i].item(),
                    confidence=confidences[i].item(),
                )
            )
            neighbor_index += 1
            if len(results) >= n_neighbors:
                break
        return results
