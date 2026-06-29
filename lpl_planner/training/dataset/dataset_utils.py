from typing import Dict, List, Tuple, Type, Any
from pathlib import Path
import torch
import pickle
import gzip

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature

def load_feature_target_from_pickle(path: Path, feature_type: Type[AbstractModelFeature]) -> AbstractModelFeature:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)
    return feature_type.deserialize(data)


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, Any]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)

def _batch_abstract_features(
    initial_not_batched_features: FeaturesType, to_be_batched_features: List[FeaturesType]
) -> FeaturesType:
    """
    Batch abstract feature with custom collate function
    :param initial_not_batched_features: features from initial batch which are used only for keys
    :param to_be_batched_features: list of features which should be batched
    :return: batched features
    """
    output_features = {}
    for key in initial_not_batched_features.keys():
        list_features = [feature_single[key] for feature_single in to_be_batched_features]
        output_features[key] = initial_not_batched_features[key].collate(list_features)

    return output_features

def recursive_pin_memory(batch):
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory()
    elif isinstance(batch, dict):
        return {k: recursive_pin_memory(v) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [recursive_pin_memory(v) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(recursive_pin_memory(v) for v in batch)
    else:
        return batch

class FeatureCollate:
    """Wrapper class that collates together multiple samples into a batch."""

    def __call__(
        self, batch: List[Tuple[FeaturesType, TargetsType]]
    ) -> Tuple[FeaturesType, TargetsType]:
        """
        Collate list of [Features,Targets] into batch
        :param batch: list of tuples to be batched
        :return (features, targets) already batched
        """
        assert len(batch) > 0, "Batch size has to be greater than 0!"

        to_be_batched_features = [batch_i[0] for batch_i in batch]
        to_be_batched_targets = [batch_i[1] for batch_i in batch]

        initial_features, initial_targets = batch[0]

        out_features = _batch_abstract_features(initial_features, to_be_batched_features)
        out_targets = _batch_abstract_features(initial_targets, to_be_batched_targets)

        return (out_features, out_targets)
    
class FeatureCollatePinMemory:
    """Wrapper class that collates together multiple samples into a batch."""

    def __call__(
        self, batch: List[Tuple[FeaturesType, TargetsType]]
    ) -> Tuple[FeaturesType, TargetsType]:
        """
        Collate list of [Features,Targets] into batch
        :param batch: list of tuples to be batched
        :return (features, targets) already batched
        """
        assert len(batch) > 0, "Batch size has to be greater than 0!"

        to_be_batched_features = [batch_i[0] for batch_i in batch]
        to_be_batched_targets = [batch_i[1] for batch_i in batch]

        initial_features, initial_targets = batch[0]

        out_features = _batch_abstract_features(initial_features, to_be_batched_features)
        out_targets = _batch_abstract_features(initial_targets, to_be_batched_targets)

        return (recursive_pin_memory(out_features), recursive_pin_memory(out_targets))
    
