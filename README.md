# MS-Thesis

### Installation

```shell
conda create --name thesis python=3.12.8
conda activate thesis
pip install -r requirements.txt
```

> [!IMPORTANT]
> Modify the `.env` file, updating `PROJECT_ROOT_PATH` with the correct absolute path to the project directory

### TensorBoard

To prepare the TensorBoard for a given experiment (e.g., `AC30_Wav_ITB_multimodal`):

1. Run `tensorboard --logdir "runs/Androids_Corpus_30s/Wav2Vec2_ItalianBERT/multimodal" --port 6010`
2. Access the TensorBoard on `http://localhost:6010/`

> [!TIP]
> - To check the status of previous TensorBoards on Linux, run `ps aux | grep tensorboard`
> - Mapping a remote execution to local server can be achieved by `ssh -L 6010:localhost:6010 USER@SERVER_ADDRESS`

### Project Structure

```shell
MS-Thesis/
├── src/                          # Contains the main project scripts
│   ├── dataset_downloader.py     # Script to download the DAIC-WoZ & Androids-Corpus datasets
│   ├── dataset_loader.py         # Loads and preprocesses datasets for training
│   ├── feature_extractor.py      # Extracts audio & text feature embeddings
│   ├── model.py                  # Defines the multimodal BiLSTM-based model architecture
│   ├── training.py               # Training & evaluation pipeline (loss, optimizer, logging)
│   ├── visualize.py              # Scripts for visualizing model results & metrics
├── data/                         # Contains the raw and processed data files
│   ├── processed/                # Processed cached feature embeddings
│   ├── raw/                      # Raw datasets before processing
│   │   ├── Androids_Corpus/      # Folder for the Androids Corpus dataset
│   │   ├── DAIC_WoZ/             # Folder for the DAIC-WoZ dataset
├── logs/                         # Training logs (loss, metrics, lr, execution time)
├── runs/                         # Stores the tensorboard workers for further access
├── .env                          # Environment variables (API keys, paths, etc.)
├── .gitignore                    # Files & directories to ignore in version control
├── playground.ipynb    # Jupyter notebook for quick experiments & visualization
├── LICENSE                       # License information for the project
├── main.py                       # main code block to run from the command-line
├── README.md                     # Project overview, setup instructions, and usage details
├── requirements.txt              # List of dependencies for installation
```