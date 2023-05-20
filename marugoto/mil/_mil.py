from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
import os

from fastai.vision.all import (
    Learner,
    DataLoader,
    DataLoaders,
    RocAuc,
    F1Score,
    SaveModelCallback,
    CSVLogger,
)
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import pandas as pd

from marugoto.data import SKLearnEncoder

import matplotlib.pyplot as plt

from .data import make_dataset
from .transformer import Transformer
from .ViT import ViT


__all__ = ["train", "deploy"]


T = TypeVar("T")


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, npt.NDArray],
    add_features: Iterable[Tuple[SKLearnEncoder, npt.NDArray]] = [],
    valid_idxs: npt.NDArray[np.int_],
    n_epoch: int = 32,
    path: Optional[Path] = None,
    num_feats: Optional[Path] = 768,
    gpu_id: Optional[Path] = 1,
    batch_size: Optional[int] = 64,
    bag_size: Optional[int] = 1024,
    pos_enc: Optional[str] = None,
    bs_tr_only: Optional[bool] = False,
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    dev = f"cuda:{gpu_id}"
    target_enc, targs = targets
    train_ds = make_dataset(
        bags=bags[~valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[~valid_idxs]),
        add_features=[(enc, vals[~valid_idxs]) for enc, vals in add_features],
        bag_size=bag_size,
    )

    valid_ds = make_dataset(
        bags=bags[valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[valid_idxs]),
        add_features=[(enc, vals[valid_idxs]) for enc, vals in add_features],
        #bag_size=bag_size,
        bag_size=bag_size if not bs_tr_only else None,
    )

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=24, drop_last=True
    )
    valid_dl = DataLoader(
        #valid_ds, batch_size=batch_size, shuffle=False, num_workers=24,
        valid_ds, batch_size=1, shuffle=False, num_workers=24,
    )
    batch = valid_dl.one_batch()
    nr_classes = batch[-1].shape[-1]
    print(f"number of classes: {nr_classes}")
    # print(f"batch length: {len(batch)}")
    # print(f"batch[0] shape: {batch[0].shape}")
    # print(f"batch[1] shape: {batch[1].shape}")
    # print(f"batch[1]: {batch[1]}")
    # print(f"batch[2] {batch[2]}")
    # print(f"batch[3]: {batch[3]}")
    # for binary classification num_classes=2 for same output dim as normal MILModel
    model = ViT(num_classes=nr_classes,input_dim=num_feats,nr_tiles=bag_size,pos_enc=pos_enc) # Transformer(num_classes=2)
    print(f"num_feats: {num_feats}")
    model.to(torch.device(dev if torch.cuda.is_available() else 'cpu')) #

    # weigh inversely to class occurances
    counts = pd.value_counts(targs[~valid_idxs])
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32
    )
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl, device=torch.device(dev if torch.cuda.is_available() else 'cpu')) #
    learn = Learner(dls, model, loss_func=loss_func,
                    metrics=[RocAuc()], path=path) #,F1Score()

    cbs = [
        #SaveModelCallback(monitor="roc_auc_score",fname=f"best_valid"),
        SaveModelCallback(fname=f"best_valid"),
        CSVLogger(),
    ]

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs)

    #pos_feat_imp = nn.functional.softmax(learn.fc[0].weight,dim=1)[:,-2:]
    #print(f"pos feature weights: {pos_feat_imp}")
    if pos_enc:
        if "posFeat" in pos_enc:
            feat_importances = torch.abs(learn.fc[0].weight).sum(dim=0).cpu().detach().numpy()
            pos_feat_importances = feat_importances[-2:]/np.sum(feat_importances)*len(feat_importances)
            print(f"pos feat importances: {pos_feat_importances}")
    #plt.bar(range(770),feat_importances)
    #plt.xlabel("Feature index")
    #plt.ylabel("Importance")
    #plt.savefig(f"feat_importance_{np.random.randint(2000)}.pdf",dpi=250)
    
    return learn


def deploy(
    test_df: pd.DataFrame,
    learn: Learner,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None,
    cont_labels: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = 1,
    bag_size: Optional[int] = 1024,
    bs_tr_only: Optional[bool] = False,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), "duplicate patients!"
    if target_label is None:
        target_label = learn.target_label
    if cat_labels is None:
        cat_labels = learn.cat_labels
    if cont_labels is None:
        cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=bag_size if not bs_tr_only else None,
    )

    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=24
    )

    # removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict(
        {
            "PATIENT": test_df.PATIENT.values,
            target_label: test_df[target_label].values,
            **{
                f"{target_label}_{cat}": patient_preds[:, i]
                for i, cat in enumerate(categories)
            },
        }
    )

    # calculate loss
    patient_preds = patient_preds_df[
        [f"{target_label}_{cat}" for cat in categories]
    ].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1)
    )
    patient_preds_df["loss"] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs), reduction="none"
    )

    patient_preds_df["pred"] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[
        [
            "PATIENT",
            target_label,
            "pred",
            *(f"{target_label}_{cat}" for cat in categories),
            "loss",
        ]
    ]
    patient_preds_df = patient_preds_df.sort_values(by="loss")

    return patient_preds_df
