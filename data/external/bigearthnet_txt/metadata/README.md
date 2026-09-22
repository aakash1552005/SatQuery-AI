---
license: cdla-permissive-1.0
task_categories:
- image-text-to-text
- visual-question-answering
- multiple-choice
task_ids:
- image-captioning
- multiple-choice-qa
language:
- en
pretty_name: BigEarthNet.txt
size_categories:
- 1M<n<10M
tags:
- remote sensing
- vision-language
- sentinel-1
- sentinel-2
- multispectral
configs:
- config_name: default
  data_files:
    - split: all_data
      path: BigEarthNet.txt.parquet
  default: true
---

<!-- CSS styling -->
<style>
.logo {
  height: 50px;
}
table, td {
  border: 1px solid black;
}
</style>
<!-- CSS styling End -->


<center>
  <p style="display: flex; justify-content: space-between; width: 100%;">
    <span style="display: flex; align-items: center; gap: 20px; margin-left: 20px">
      <a href="https://bifold.berlin/">
        <img src="./static/images/logos/BIFOLD_Logo_farbig.svg" alt="BIFOLD logo" class="logo">
      </a>
      <a href="https://www.tu.berlin/">
        <img src="./static/images/logos/tu-berlin-logo-long-red.svg" alt="TU Berlin Logo" class="logo">
      </a>
      <a href="https://www.rsim.berlin/">
        <img src="./static/images/logos/RSiM_Logo.png" alt="Remote Sensing Image Analysis Group Logo" class="logo">
      </a>
    </span>
    <span style="display: flex; align-items: top; gap: 10px;">
      <a href="https://txt.bigearth.net">
        <img alt="Paper Website Badge" src="https://img.shields.io/badge/Paper-Website-%237FCCE0">
      </a>
      <a href="https://arxiv.org/abs/2603.29630">
        <img alt="Paper arXiv Badge" src="https://img.shields.io/badge/Paper-arXiv-%23b31b1b">
      </a>
      <a href="https://cdla.dev/permissive-1-0/">
        <img alt="Community Data License Agreement - Permissive - Version 1.0 License Badge" src="https://img.shields.io/badge/License-CDLA%201.0-blue.svg">
      </a>
    </span>
  </p>
</center>

# BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation

`BigEarthNet.txt` is a large-scale multi-sensor image–text dataset for Earth observation, designed to advance vision–language learning on remote sensing data. It comprises <b>464,044 co-registered Sentinel-1 (SAR) and Sentinel-2 (multispectral) image pairs</b> collected over Europe, paired with approximately <b>9.6 million textual annotations</b>. The textual annotations include geographically anchored captions describing land-use/land-cover (LULC) classes and their spatial relationships, diverse visual question answering (VQA) pairs (binary and multiple-choice), and referring expression instructions for LULC localization. In addition, the dataset provides a <b>manually verified benchmark split consisting of 1,082 image pairs with 15,029 textual annotations</b>, specifically designed for reliable evaluation of vision–language models on complex multi-sensor remote sensing tasks. For more details on the dataset, please see our [paper website](https://txt.bigearth.net).

<center>
  <img src="./static/images/dataset.svg" width="80%" style="margin-bottom: 0">
  <div style="font-size: 0.8em; color: gray; width: 80%; margin-top: 0">
    The dataset supports 15 tasks (Presence, Area, Counting, Adjacency, Relative Position Country, Season, and Climate Zone, denoted as Pr, A, Cnt, Adj, RP, Loc, S, and Clt, respectively) across 4 broad categories.
  </div>
</center>

<hr>

## Parquet File Structure
The `BigEarthNet.txt.parquet` file contains multiple attributes:
- `ID`: A unique identifier for each sample in the dataset.
- `s1_name`: The name of the Sentinel-1 patch from `BigEarthNet v2.0`.
- `patch_id`: The name of the Sentinel-2 patch from `BigEarthNet v2.0`.
- `input`: The instruction or question for the VLM.
- `output`: The reference answer.
- `type`: The broader task-type of the sample, i.e., `binary`, `mcq`, `captioning`, or `bounding box`.
- `category`: The more fine-grained task-type. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/KzrmYgF) for all type-category combinations.
- `split`: The associated split of the sample, i.e., `train`, `validation`, `test`, or `bench`.
- `latitude`: The latitude coordinates of the center of the image patch.
- `longitude`: The longitude coordinates of the center of the image patch.
- `country`: The acquisition country of the image patch. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/yn1wpPS) for all available values.
- `season`: The acquisition season of the image patch. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/m59YuRc) for all available values.
- `climate_zone`: The associated [Köppen-Geiger](https://www.nature.com/articles/s41597-023-02549-6) climate zone. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/SUU1DwA) for all available values.

<hr>

## How to use
We show the recommended way to prepare the image and text data to be jointly used in the form of a custom [PyTorch Dataset](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) `BENTxTDataset` or [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) `BENTxTDataModule` provided in [ben_txt_datamodule.py](ben_txt_datamodule.py).

#### 1. Download `BigEarthNet.txt.parquet`
Download using Git.
```bash
git clone https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt
```

#### 2. Download the Image Data
Download the Sentinel-1 and Sentinel-2 image data from the [BigEarthNet v2.0 website](https://bigearth.net/).

#### 3. Preprocess the Image Data
Convert the Sentinel-1 and Sentinel-2 image data to `safetensors` stored in an `LMDB` database for higher throughput using [rico-hdl](https://github.com/rsim-tu-berlin/rico-hdl). Follow the installation instructions on [GitHub](https://github.com/rsim-tu-berlin/rico-hdl), then execute the following command to convert the Sentinel-1 and Sentinel-2 image data downloaded to `<S1_ROOT_DIR>` and `<S2_ROOT_DIR>`.
```bash
rico-hdl bigearthnet --bigearthnet-s1-dir <S1_ROOT_DIR> --bigearthnet-s2-dir <S2_ROOT_DIR> --target-dir Encoded-BigEarthNet
```

#### 4. Load the Data
Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Install the required packages via uv using the command below. You can specify if you want to use the PyTorch CPU version or PyTorch with CUDA 12.6 by choosing `cpu` or `cu126` as the `<option>`.
```bash
uv sync --extra <option>
```

<hr>

The following examples show how to jointly load text samples from `BigEarthNet.txt` with the respective image data from `BigEarthNet v2.0`.
After executing the suggested steps above, you should be able to run the following [file from this repository](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/main/example_data_loading.py):
```bash
uv run example_data_loading.py
```

or load the data manually using the [provided datamodule](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/main/ben_txt_datamodule.py) as shown in the following two examples:

This example shows how to load the Red (B04), Green (B03), and Blue (B02) band from the Sentinel-2 image data using the `BENTxTDataset` Datasets class. More details about the custom Dataset are provided in [ben_txt_datamodule.py](ben_txt_datamodule.py).
```python
from ben_txt_datamodule import BENTxTDataset

ds_rgb = BENTxTDataset(
  lmdb_file = "Encoded-BigEarthNet/",
  metadata_file = "BigEarthNet.txt.parquet",
  bands = ("B04", "B03", "B02"),
  img_size = 120
)

sample = ds_rgb[0]
print(f"RGB input image: {sample['image_input'].shape}")
print(f"Text input: {sample['text_input']}")
print(f"Reference output: {sample['reference_output']}")
```

This example shows how to load the 10m and 20m spatial resolution bands from Sentinel-1 and Sentinel-2 using the `BENTxTDataModule` Lightning DataModule class. In this example we apply multiple metadata filters on `BigEarthNet.txt`, more details about the custom DataModule are provided in [ben_txt_datamodule.py](ben_txt_datamodule.py).
```python
from ben_txt_datamodule import BENTxTDataModule

# Lightning DataModule example using the 10m and 20m spatial resolution bands from Sentinel-1 and Sentinel-2 and multiple metadata filters.
# The datamodule will create 4 dataloaders: train, val, test, and bench.
dm = BENTxTDataModule(
  image_lmdb_file = "Encoded-BigEarthNet/",
  metadata_file = "BigEarthNet.txt.parquet",
  bands = 'S1S2-10m20m',
  img_size = 120,
  batch_size = 1,
  num_workers_dataloader = 0,
  types = ['mcq'],
  categories = ['climate zone'],
  countries = ['Portugal', 'Finland'],
  seasons = ['Summer'],
  climate_zones = None,
  point_token = ['<point>', '</point>'],
  ref_token = ['<ref>', '</ref>']
)
dm.setup()
  
train_dl = dm.train_dataloader()
for batch in train_dl:
  print(f"Batch image input shape: {batch['image_input'].shape}")
  print(f"First batch sample text input: {batch['text_input'][0]}")
  print(f"First batch sample text reference output: {batch['reference_output']}")
  break
```

<hr>

### Citation
If you use the `BigEarthNet.txt` dataset, please cite:
```
J. Herzog, M. Adler, L. Hackel, Y. Shu, A. Zavras, I. Papoutsis, P. Rota, B. Demir, 
"BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation", 
Arxiv Preprint arXiv:2603.29630, 2026.
```

```bibtex
@article{Herzog2026BigEarthNetTXT,
  title={BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation},
  author={Johann-Ludwig Herzog and Mathis Jürgen Adler and Leonard Hackel and Yan Shu and Angelos Zavras and Ioannis Papoutsis and Paolo Rota and Begüm Demir},
  journal={Arxiv Preprint arXiv:2603.29630},
  year={2026},
}
```