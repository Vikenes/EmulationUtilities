from typing import List, Optional, Union, Tuple
from pydantic import BaseModel


class DataConfig(BaseModel):
    feature_columns: List[str]
    label_columns: List[str]
    batch_size: Optional[int]
    num_workers: Optional[int]
    shuffle: Optional[bool]
    train_data_path: "str"
    val_data_path: "str"
    test_data_path: "str"
    feature_scaler: "str"
    label_scaler: Optional["str"]



class ModelConfig(BaseModel):
    type: str
    hidden_dims: Optional[List[int]]
    activation: Optional[str]
    loss: Optional[str]
    dropout: Optional[float]
    learning_rate: float
    weight_decay: Optional[float]
    batch_norm: Optional[bool]
    zero_bias: Optional[bool] = False
    positive_output: bool = False
    kernel_size: Optional[Union[int, Tuple]] = None
    kernel_size_left: Optional[int] = None
    kernel_size_right: Optional[int] = None
    n_fcn: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None
    output_filters: Optional[int] = None
    autoencoder: Optional[bool] = None
    input_height: Optional[int] = None
    latent_dim: Optional[int] = None


class TrainingConfig(BaseModel):
    gpus: Optional[List[int]]
    max_epochs: int
    patience: Optional[int] = 20
    stochastic_weight_avg: bool
    swa_start: Optional[int] = 110
    gradient_clip_val: Optional[float] = None
    default_root_dir: str
