from __future__ import annotations

import argparse
import gzip
import logging
import multiprocessing as mp
import os
import pickle
import random
import re
import time
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from PIL import Image
from goatools.obo_parser import GODag
from scipy.spatial import cKDTree
from tqdm import tqdm
from anndata import AnnData
from scipy.spatial import KDTree

Image.MAX_IMAGE_PIXELS = None

def load_adata(data_path: str, dataset: str, slide: str) -> Tuple[ad.AnnData, np.ndarray]:

    slide_dir = Path(f"{data_path}/{slide}")
    h5_file = Path(f"{slide_dir}/filtered_feature_bc_matrix.h5")

    if h5_file.exists():
        adata = sc.read_visium(path=slide_dir, count_file="filtered_feature_bc_matrix.h5")
        full_img_path = Path(f"{slide_dir}/full_image.tif")
        hires_img_path = Path(f"{slide_dir}/spatial/tissue_hires_image.png")
        
        if full_img_path.exists():
            img_arr = np.array(Image.open(full_img_path))
            img_res = "fullres"
        else:
            img_arr = np.array(Image.open(hires_img_path))
            img_res = "hires"

    else:
        expr_fp = Path(f"{slide_dir}/{slide}.tsv.gz")
        coord_fp = Path(f"{slide_dir}/{slide}_selection.tsv.gz")
        img_fp = Path(f"{slide_dir}/{slide}.jpg")

        for path in [expr_fp, coord_fp, img_fp]:
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

        expr_df = pd.read_csv(expr_fp, sep="\t", index_col=0)
        coord_df = pd.read_csv(coord_fp, sep="\t")

        if all(re.match(r"^\d+x\d+$", s) for s in expr_df.index.to_list()):
            coord_df["spot"] = (coord_df["x"].astype(int).astype(str) + "x" + coord_df["y"].astype(int).astype(str))
        else:
            if "spot" not in coord_df.columns:
                raise ValueError(f"The coordinate file must contain a 'spot' column: {coord_fp}")

        shared_spot = expr_df.index.intersection(coord_df["spot"])
        expr_df = expr_df.loc[shared_spot]
        coord_df = coord_df.set_index("spot").loc[shared_spot].reset_index()
        pixels = coord_df[["pixel_x", "pixel_y"]].to_numpy()

        dists = NearestNeighbors(n_neighbors=4).fit(pixels).kneighbors(pixels)[0]
        min_dist = np.min(dists[:, 1]) * 0.85

        img_arr = np.array(Image.open(img_fp))
        img_res = "fullres"

        scalef_fp = Path(f"{slide_dir}/scalefactors.json")
        scalefactors = {
            "spot_diameter_fullres": float(min_dist),
            "tissue_hires_scalef": 1,
            "tissue_lowres_scalef": 1
        }

        if scalef_fp.exists():
            try:
                with open(scalef_fp, "r", encoding="utf-8") as f:
                    scalef_json = json.load(f)
                for key in scalefactors:
                    if key in scalef_json:
                        scalefactors[key] = float(scalef_json[key])
                logger.info(f"Successfully read scalefactors.json: {scalef_fp}")
            except Exception as e:
                logger.warning(f"Could not parse scalefactors.json: {scalef_fp}, using default scale parameters. Reason: {e}")

        adata = ad.AnnData(
            X=sp.csr_matrix(expr_df),
            obs=pd.DataFrame(index=expr_df.index),
            var=pd.DataFrame(index=expr_df.columns),
        )
        adata.obsm["spatial"] = pixels
        adata.obs["neighbor_distance"] = min_dist
        adata.uns["spatial"] = {
            slide: {
                "images": {
                    "hires": img_arr,
                    "lowres": img_arr
                },
                "scalefactors": scalefactors,
                "metadata": {
                    "chemistry_description": "raw data not provided",
                    "software_version": "raw data not provided"
                }
            }
        }
    adata.obs["slide"] = slide
    return adata, img_arr, img_res
