###  Step 0: Patch Cropping (Based on Spatial Transcriptomics Spots)

> This is the **initial step** of the pipeline. Before nuclear segmentation, you must crop image patches around spatial transcriptomics **spot locations** from high-resolution histology images.  
> Each sample (e.g., `A1`, `B1`, etc.) contains its own image and corresponding spot coordinate files.

---

####  Input Format

Each sample directory (e.g., `A1`, `B1`, etc.) should contain the following files:

| File Name                  | Description                                      |
|----------------------------|--------------------------------------------------|
| `*.jpg`                   | Histology tissue image for the sample            |
| `*.tsv.gz`                | Spot coordinate file (with x/y positions)        |
| `*_selection.tsv.gz`      | *(Optional)* A filtered list of selected spots   |

These files will be used to generate image patches centered on each spatial spot.

---

#### Example Directory Structure

```
data/
├── A1/
│   ├── A1.jpg
│   ├── A1.tsv.gz
│   └── A1_selection.tsv.gz
├── A2/
│   ├── A2.jpg
│   ├── A2.tsv.gz
│   └── A2_selection.tsv.gz
├── ...
├── B1/
│   ├── B1.jpg
│   ├── B1.tsv.gz
│   └── B1_selection.tsv.gz
```

---

####  Output

Cropped image patches will be saved to:

```
/data/bc/{SAMPLE_ID}/patches/
```

Each patch corresponds to one spatial spot, and will be used in downstream nuclear segmentation.

---


