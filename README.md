# MS-Thesis

The following commands were run in MacOS iTerminal to install the Python packages and prerequisites:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 src/data/load_dataset.py
```


Create a `.env` file at the project root, and add the following details to the file:
```
PROJECT_ROOT_PATH=/Users/<LOCAL_PATH_TO_THE_PROJECT>/MS-Thesis
```


Suggested Project Organization:
```shell
project_name/
├── data/                      # Store raw and processed datasets
│   ├── raw/                   # Original datasets (downloaded files)
│   │   ├── eatd_corpus/
│   │   ├── daic_woz/
│   ├── processed/             # Preprocessed data (features, cleaned data)
│   │   ├── audio_features/
│   │   ├── text_features/
├── src/                       # Main source code
│   ├── data/                  # Code related to data downloading & preparation
│   │   ├── load_dataset.py   # Download and save DAIC-WoZ and EATD-Corpus
│   │   ├── preprocess_eatd.py # Preprocess EATD-Corpus
│   │   ├── preprocess_daic.py # Preprocess DAIC-WoZ
│   ├── features/              # Feature extraction for text and audio
│   │   ├── audio/
│   │   │   ├── extract_audio_eatd.py
│   │   │   ├── extract_audio_daic.py
│   │   └── text/
│   │       ├── extract_text_eatd.py
│   │       ├── extract_text_daic.py
│   ├── fusion/                # Fusion techniques for embeddings
│   │   ├── simple_concat.py   # Example: Simple concatenation fusion
│   │   ├── advanced_fusion.py # Advanced fusion methods
│   ├── classification/        # Code for classifiers
│   │   ├── classifiers.py     # Classifier implementations (e.g., SVM, MLP)
│   │   ├── train_classifier.py # Code for training and evaluation
│   ├── domain_generalization/ # Domain generalization techniques
│       ├── generalization.py  # Domain generalization implementation
│       ├── evaluate.py        # Evaluation for domain generalization
├── experiments/               # Scripts for different experiments
│   ├── exp_fusion_01.py       # Example: Experiment script for fusion method 1
│   ├── exp_classifier_01.py   # Example: Experiment script for classifier 1
├── tests/                     # Unit tests for various modules
│   ├── test_data.py           # Test for data downloading/preprocessing
│   ├── test_features.py       # Test for feature extraction
│   ├── test_fusion.py         # Test for fusion methods
├── utils/                     # Utility functions
│   ├── logging.py             # Custom logging functions
│   ├── metrics.py             # Metric calculations (e.g., accuracy, F1-score)
│   ├── helpers.py             # Reusable helper functions
├── notebooks/                 # Jupyter notebooks for exploratory analysis
│   ├── eda_eatd.ipynb         # Exploratory Data Analysis for EATD-Corpus
│   ├── eda_daic.ipynb         # Exploratory Data Analysis for DAIC-WoZ
├── configs/                   # Configuration files for different experiments
│   ├── default.yaml           # Default configuration
│   ├── exp_fusion.yaml        # Config for a specific fusion experiment
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation about the project
├── .gitignore                 # Ignored files/folders
└── setup.py                   # For creating a Python package (optional)
```