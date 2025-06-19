# African Language Emotion Classification

## Project Overview
This project examines emotion classification in African languages (Zulu and Nigerian Pidgin English) using transfer learning approaches with multilingual language models. We implement and evaluate various methods including zero-shot transfer, cross-lingual sequential fine-tuning, and adapter-based approaches to enhance emotion detection capabilities in low-resource African languages.

## Table of Contents
- [Installation](#installation)
  - [Basic Setup](#basic-setup)
  - [Handling sentencepiece Installation](#handling-sentencepiece-installation)
  - [Handling Model Files](#handling-model-files)
  - [Using Git LFS](#using-git-lfs)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Access](#data-access)
  - [Running Fine-Tuning](#running-fine-tuning)
  - [Running Jupyter Notebooks](#running-jupyter-notebooks)
  - [Running Evaluation](#running-evaluation)
  - [Running Analysis](#running-analysis)
  - [Output Directories](#output-directories)
- [Models](#models)
- [Experimental Approaches](#experimental-approaches)
- [Results](#results)

## Installation

### Basic Setup

1. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   
   # For analysis components, install additional packages:
   pip install bertviz shap seaborn matplotlib tabulate jupyter
   
   # For adapter-based approaches:
   pip install adapter-transformers peft
   ```

3. **Create a .env file with your HuggingFace token**
   ```powershell
   # Create .env file with the following content:
   echo "HUGGING_FACE_HUB_TOKEN=your_token_here" > .env
   ```
   
   You'll need a Hugging Face account and API token to access datasets:
   - Register at [HuggingFace.co](https://huggingface.co/join)
   - Create a token at [HuggingFace.co/settings/tokens](https://huggingface.co/settings/tokens)
   - This token allows downloading the BRIGHTER, AfriHate, and AfriSenti datasets

### Handling sentencepiece Installation

If you encounter issues with sentencepiece installation, follow these steps:

First, you need to install the pwsh.exe if you don't have it

```winget install --id Microsoft.Powershell --source winget
```

1. **Install build prerequisites**
   ```powershell
   pip install cmake
   ```

2. **Manual build process**
   ```powershell
   # Clone the repository
   git clone https://github.com/google/sentencepiece.git
   cd sentencepiece

   # Create build directory and compile
   mkdir build
   cd build
   
   # Configure with CMake
   cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=./root
   
   # Build and install
   cmake --build . --config Release --target install
   
   # Build Python wheel
   cd ..\python
   python setup.py bdist_wheel
   
   # Install the wheel
   pip install (Get-Item .\dist\sentencepiece*.whl).FullName
   ```

3. **Return to project directory**
   ```powershell
   cd ..\..\
   ```

### Handling Model Files

**Important:** Model files are not included in the Git repository due to their large size. These files include:

- Model weights (*.safetensors) in the `saved-models` directory
- Virtual environment files in the `.venv` directory

If you need the pre-trained models:

1. **Option 1: Download from external storage**
   Contact the project maintainers for access to the pre-trained models.

2. **Option 2: Train models from scratch**
   Use the training scripts in this repository to train the models:
   ```powershell
   # Example: Running adapter model training
   python run_adapter_fine_tuning.py
   
   # Example: Running cross-lingual training
   python run_cross_lingual_fine_tuning.py
   ```

3. **Option 3: Use Hugging Face Hub**
   Some models may be available on the Hugging Face Hub. See the Models section below for details.

### Using Git LFS

This repository uses Git Large File Storage (LFS) for managing large model files. To work with these files:

1. **Install Git LFS**
   ```powershell
   # Install Git LFS
   git lfs install
   ```

2. **Clone the repository with LFS support**
   ```powershell
   # When cloning for the first time
   git lfs clone https://github.com/your-username/COS760-group-29.git
   
   # Or if you already cloned without LFS
   git lfs pull
   ```

3. **Track large files**
   Large model files (*.safetensors, *.bin, *.pt) are automatically tracked by Git LFS.
   
4. **Push changes**
   ```powershell
   git add .
   git commit -m "Your commit message"
   git push
   ```

> **Note**: The repository primarily uses Jupyter notebooks for adapter and cross-lingual approaches. Therefore, you will not find standalone `run_adapter_fine_tuning.py` or `run_cross_lingual_fine_tuning.py` scripts.

## Project Structure

```
COS760-group-29/
├── data/                      # Data loading and preprocessing
│   └── data_loader.py         # Functions for loading and preprocessing datasets
├── results/                   # Evaluation results and analysis utilities
│   ├── attention_visualization.py  # Attention pattern visualization
│   ├── attention_visualization.ipynb  # Interactive attention visualization
│   ├── baseline_error_analysis.py  # Error analysis for baseline models
│   ├── baseline_shap_analysis.py  # SHAP analysis for baseline models
│   ├── error_analysis.py     # Error analysis for fine-tuned models
│   ├── evaluate_fine_tuned.py  # Evaluation for fine-tuned models
│   ├── model_evaluation.py  # General model evaluation functions
│   ├── shap_analysis.py     # SHAP analysis for fine-tuned models
│   └── zero_shot_shap_analysis.py  # SHAP analysis for zero-shot models
├── saved-models/             # Directory for storing model outputs
│   └── adapters/             # Subdirectory for storing adapter models
├── training/                 # Training and fine-tuning scripts
│   ├── fine_tuning.py        # Fine-tuning models on English data
│   ├── AdapterMethods.ipynb  # Notebook for adapter-based fine-tuning
│   ├── Cross_lingual_transfer_learning.ipynb  # Notebook for cross-lingual approaches
├── presentation_visuals/     # Visualizations for presentation and reporting
├── run_evaluation.py         # Main script for model evaluation
├── run_fine_tuned_evaluation.py  # Script for evaluating fine-tuned models
├── run_model_analysis.py     # Script for comprehensive model analysis
├── README.md
├── requirements.txt
└── sentencepiece/           # SentencePiece tokenizer library
```

## Usage

### Data Access

The project uses several datasets that need to be downloaded via the Hugging Face Hub:

1. **BRIGHTER Emotion Dataset**: Multi-language emotion classification dataset
   - English (eng): `brighter-dataset/BRIGHTER-emotion-categories/eng`
   - Zulu (zul): `brighter-dataset/BRIGHTER-emotion-categories/zul`
   - Nigerian Pidgin English (pcm): `brighter-dataset/BRIGHTER-emotion-categories/pcm`

2. **AfriHate Dataset**: Used for pre-training adapters
   - Zulu: `afrihate/afrihate/zul`
   - Nigerian Pidgin English: `afrihate/afrihate/pcm`

3. **AfriSenti Dataset**: Used for pre-training Nigerian Pidgin English adapters
   - Nigerian Pidgin English: `shmuhammad/AfriSenti-twitter-sentiment/pcm`

These datasets are downloaded automatically when running the scripts if you have set up your Hugging Face token correctly.

### Running Fine-Tuning

1. **Setup the environment** as described in the [Installation](#installation) section.

2. **Standard Fine-tuning** (Zero-shot approach):
   ```powershell
   python -m training.fine_tuning
   ```
   This script fine-tunes mBERT and AfriBERTa on English emotion data from the BRIGHTER dataset.

3. **Cross-lingual and Triple Sequential Fine-tuning**:
   
   Use the Jupyter notebook for these approaches:
   ```powershell
   jupyter notebook training/Cross_lingual_transfer_learning.ipynb
   ```
   
   This notebook contains:
   - Cross-lingual sequential fine-tuning (English → Zulu/PCM)
   - Triple sequential fine-tuning (English → PCM → Zulu)

4. **Adapter-based Fine-tuning**:
   
   Use the adapter methods notebook:
   ```powershell
   jupyter notebook training/AdapterMethods.ipynb
   ```
   
   This notebook implements:
   - Language-specific adapter training on AfriHate/AfriSenti
   - Adapter fine-tuning on BRIGHTER emotion data
   - Parameter-efficient training with frozen base models

### Running Jupyter Notebooks

For both the adapter and cross-lingual approaches:

1. **Start Jupyter**:
   ```powershell
   jupyter notebook
   ```

2. **Navigate** to the respective notebook:
   - `training/AdapterMethods.ipynb`
   - `training/Cross_lingual_transfer_learning.ipynb`

3. **Execute cells** in sequence, following the instructions and comments within each notebook.

4. **Save trained models** to the `saved-models` directory as specified in the notebooks.

### Running Evaluation

1. **Evaluate fine-tuned models** (Zero-shot transfer):
   ```powershell
   python run_fine_tuned_evaluation.py
   ```
   This evaluates standard fine-tuned models on Zulu and Nigerian Pidgin test sets.

2. **Run baseline model evaluation** (non-fine-tuned models):
   ```powershell
   python run_evaluation.py
   ```

3. **Evaluate notebook-trained models**:
   
   For cross-lingual and adapter models, use the evaluation cells in the respective notebooks:
   - `AdapterMethods.ipynb` contains evaluation sections for adapter-based models
   - `Cross_lingual_transfer_learning.ipynb` contains evaluation sections for cross-lingual models

### Running Analysis

1. **Baseline Model Analysis**:
   ```powershell
   # Error analysis for baseline models
   python -m results.baseline_error_analysis
   
   # SHAP analysis for baseline models
   python -m results.baseline_shap_analysis
   ```

2. **Zero-shot Model Analysis**:
   ```powershell
   # SHAP analysis for zero-shot models
   python -m results.zero_shot_shap_analysis
   ```

3. **Comprehensive Analysis**:
   ```powershell
   # Run all analysis types (attention, error, SHAP)
   python run_model_analysis.py --run-all
   
   # Or run specific analyses
   python run_model_analysis.py --shap-only  # SHAP analysis only
   python run_model_analysis.py --error-only  # Error analysis only
   python run_model_analysis.py --attention-only  # Attention visualization only
   ```

4. **Interactive Attention Visualization**:
   ```powershell
   jupyter notebook results/attention_visualization.ipynb
   ```
   This notebook provides interactive visualizations of attention patterns.

### Output Directories

The analysis scripts generate various outputs in specific directories:

- **SHAP Analysis**:
  - Baseline SHAP: `shap_analysis/baseline/`
  - Zero-shot SHAP: `zero_shot_shap_analysis/`
  - Files include prediction heatmaps, example-specific visualizations, and token importance text files

- **Error Analysis**:
  - Baseline Errors: `error_analysis/mBERT_Zulu/`, `error_analysis/mBERT_PCM/`, etc.
  - Files include confusion matrices, error types, and misclassification examples

- **Attention Visualization**:
  - HTML files in `visualizations/`
  - Interactive visualizations in the notebook: `results/attention_visualization.ipynb`

- **Model Outputs**:
  - Fine-tuned models: `saved-models/mbert_english/`, `saved-models/afriberta_english/`
  - Adapter models: `saved-models/adapters/`
  - Cross-lingual models are saved according to the paths specified in the notebooks

- **Metrics and Charts**:
  - Presentation-ready visualizations: `presentation_visuals/`
  - Performance comparison charts and metrics tables


## Models

This project utilizes two main multilingual pre-trained models:

1. **mBERT** (bert-base-multilingual-cased)
   - Trained on 104 languages using masked language modeling
   - 12-layer, 768-hidden, 12-heads, 110M parameters

2. **AfriBERTa** (castorini/afriberta_base)
   - Fine-tuned XLM-RoBERTa for African languages
   - Better support for low-resource African languages

### Fine-tuning Approaches

We implement several fine-tuning approaches:

1. **Standard Fine-tuning**
   - Full parameter updates across the entire model

2. **Sequential Fine-tuning**
   - Train on one language, then continue training on target languages

3. **Triple Sequential Fine-tuning**
   - English → Nigerian Pidgin → Zulu sequence

4. **Adapter-based Fine-tuning**
   - Language-specific adapters for Zulu and Nigerian Pidgin
   - Pre-trained on AfriHate (Zulu) and combined AfriHate/AfriSenti (Nigerian Pidgin) datasets
   - Plugged into base models and fine-tuned on BRIGHTER emotion data

This project evaluates two multilingual models:

- **mBERT** (bert-base-multilingual-cased): A multilingual version of BERT pre-trained on 104 languages.
- **AfriBERTa** (castorini/afriberta_base): A RoBERTa model specifically pre-trained on African languages.

## Experimental Approaches

### 1. Base Model Evaluation
Evaluating the base pre-trained models directly on Zulu and Nigerian Pidgin emotion classification.

### 2. Zero-shot Cross-Lingual Transfer
Fine-tune the models on English data from the BRIGHTER dataset, then test directly on Zulu and Nigerian Pidgin without any fine-tuning on these target languages.

This approach investigates how knowledge transfers from a high-resource language (English) to low-resource African languages, addressing the challenge of limited labeled data.

### 3. Sequential Fine-tuning
Fine-tune the models first on English data, then continue fine-tuning on Zulu and Nigerian Pidgin data from the BRIGHTER dataset.

### 4. Triple Sequential Fine-tuning
Fine-tune the models in sequence across three languages: English → Nigerian Pidgin → Zulu, potentially improving cross-lingual transfer.

### 5. Adapter-Based Fine-tuning
This approach uses lightweight adapters to efficiently fine-tune the models for specific languages:

1. **Stage 1**: Train language-specific adapters for Zulu and Nigerian Pidgin using:
   - AfriHate dataset for Zulu (hate speech classification)
   - Combined AfriHate and AfriSenti datasets for Nigerian Pidgin (hate speech and sentiment classification)

2. **Stage 2**: Plug these language-specific adapters into the base models and fine-tune on BRIGHTER emotion data

The method follows this process:
1. Initialize base models (mBERT and AfriBERTa)
2. Add small adapter modules with ~1-2% of base model parameters
3. Freeze the base model weights
4. Train only the adapter parameters on language-specific data (AfriHate, AfriSenti)
5. Plug trained adapters into fresh base models
6. Fine-tune the adapters on the BRIGHTER emotion data for the target languages
7. Evaluate on test sets for all languages

Benefits of the adapter approach:
- Memory efficient: Only a small set of parameters are trained (less than 2% of the full model)
- Modular: Language-specific adapters can be swapped in and out of the base model
- Practical: Easier to distribute and deploy compared to full model fine-tuning
- Cost-effective: Reduced computational requirements for training
- Knowledge preservation: Maintains the multilingual knowledge in the base model while adapting to specific languages

## Results

The results of our experiments are available in the following locations:

1. **Performance Metrics and Comparison Tables**
   - See `presentation_visuals/` for compiled charts and tables
   - Log files in the root directory with timestamps contain detailed metrics

2. **Visualizations**
   - SHAP analysis outputs: `shap_analysis/`
   - Error analysis: `error_analysis/`
   - Attention visualizations: `visualizations/`

3. **Interactive Notebooks**
   - Adapter methods results: `training/AdapterMethods.ipynb` (final cells)
   - Cross-lingual results: `training/Cross_lingual_transfer_learning.ipynb` (final cells)
   - Attention visualization: `results/attention_visualization.ipynb`

4. **Detailed Analysis**
   - For a deep dive into the results, run:
     ```powershell
     python run_model_analysis.py --run-all
     ```
   - This generates comprehensive analysis outputs in the respective directories

