from pathlib import Path
import pickle
from typing import Union, List, Optional, Tuple, Dict, TypeVar
import numpy as np
import torch
import yaml

import _scalers as scalers
import _models as models

Scaler = TypeVar("Scaler")
import torch.nn as nn

class Predictor:
    def __init__(
        self,
        feature_scaler: Scaler,
        label_scaler:   Scaler,
        model:          "nn.Module",
        device:         int     = -1,
        params:         dict    = None,
    ):
        if model is not None:
            model.eval()
        self.feature_scaler = feature_scaler
        self.label_scaler   = label_scaler
        self.model          = model
        self.device         = device

    @classmethod
    def from_path(cls, path: Union[Path, str], device: int = -1):
        if type(path) is str:
            path = Path(path)
        model = cls.load_model_from_path(path,)
        params = None
        feature_scaler, label_scaler = cls.load_scalers_from_path(path=path)
        return cls(
            feature_scaler  =feature_scaler,
            label_scaler    =label_scaler,
            model           =model,
            device          =device,
            params          =params,
        )

    @classmethod
    def get_path_to_checkpoint(cls, path):
        path_to_checkpoints = path / "checkpoints"
        return [f for f in path_to_checkpoints.glob("*.ckpt")][0]

    @classmethod
    def load_hyperparams(cls, path):
        with open(path / "hparams.yaml", "r") as fp:
            hparams = yaml.safe_load(fp)
        return hparams

    @classmethod
    def load_config(cls, path):
        with open(path / "config.yaml", "r") as fp:
            config = yaml.safe_load(fp)
        return config

    @classmethod
    def load_model_from_path(
        cls, path,
    ):
        path_to_checkpoint  = cls.get_path_to_checkpoint(path=path)
        hparams             = cls.load_hyperparams(path=path)
        config              = cls.load_config(path=path)
        model               = getattr(models, config["model"]["type"])
        return model.load_from_checkpoint(path_to_checkpoint, **hparams)

   
    @classmethod
    def load_scalers_from_path(
        cls, path: Path, flax: bool = False
    ) -> Tuple[Scaler, Scaler]:
        with open(path / "scalers.pkl", "rb") as fp:
            scaler_dict = pickle.load(fp)
            scaler_dict["features"]["attrs"] = {
                k: v for k, v in scaler_dict["features"]["attrs"].items() if k != "flax"
            }
            scaler_dict["labels"]["attrs"] = {
                k: v for k, v in scaler_dict["labels"]["attrs"].items() if k != "flax"
            }
        feature_scaler = getattr(scalers, scaler_dict["features"]["type"])(
            flax=flax, **scaler_dict["features"]["attrs"]
        )
        label_scaler = getattr(scalers, scaler_dict["labels"]["type"])(
            flax=flax, **scaler_dict["labels"]["attrs"]
        )
        return feature_scaler, label_scaler

    def __call__(self, 
                 inputs: Union[np.array, torch.Tensor], 
                 return_tensors=False):
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs.astype(np.float32))


        if not hasattr(self.feature_scaler, "mean_") and type(self.feature_scaler).__name__ != "IdentityScaler":
            """
            Previous emulators had unscaled data, despite using a scalar in the config.
            Now, scaling is done in the emulator, 
            and no scaling corresponds to IdentityScaler.

            Thus, for older emulators, we don't invert scaling if the scaler has no mean_ attribute.  
            Will be removed in the future, when all emulators have been retrained properly.
            """
            transform = False 
        else:
            transform = True
        
        if transform:
            inputs = self.feature_scaler.transform(inputs)
        
        with torch.no_grad():
            """
            Not sure whether detach is needed. 
            """
            predictions = self.model.forward(inputs).detach()
                
        if transform:
            predictions = self.label_scaler.inverse_transform(predictions)

        if not return_tensors:
            return predictions.numpy()
        

        return predictions
