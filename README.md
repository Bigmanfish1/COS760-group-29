# African Language Emotion Classification

## Project Overview
This project examines emotion classification in African languages (Zulu and Nigerian Pidgin) using transfer learning approaches with multilingual language models.

## Table of Contents
- [Installation](#installation)
  - [Basic Setup](#basic-setup)
  - [Handling sentencepiece Installation](#handling-sentencepiece-installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Team](#team)

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
   ```

3. **Create a .env file with your HuggingFace token**
   ```powershell
   # Create .env file
   "HUGGING_FACE_HUB_TOKEN=your_token_here"
   ```

### Handling sentencepiece Installation

If you encounter issues with sentencepiece installation, follow these steps:

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

## Project Structure

```
COS760-group-29/
├── data/                # Data loading and preprocessing
│   ├── __init__.py
│   └── loader.py
├── results/             # Evaluation results and utilities
│   ├── __init__.py
│   └── evaluation.py
├── saved-models/        # Directory for storing model outputs
├── README.md
├── requirements.txt
└── run_evaluation.py    # Main script to run evaluations
```

## Usage

1. **Setup the environment** as described in the [Installation](#installation) section.

2. **Run the evaluation script**:
   ```powershell
   python run_evaluation.py
   ```

3. **View results**:
   Results will be saved in the `results/` directory in JSON format with the naming convention:
   ```
   {model_name}_{language}_{data_split}_{timestamp}.json
   ```
   python run_evaluation.py
   ```

3. **View results**:
   Results will be saved in the `results/` directory in JSON format.
   
4. **Use individual modules**:
   You can also use the individual modules in your own scripts:
   
   ```python   # Example: Load datasets and models
   from data.loader import load_data
   from results.evaluation import load_models
   
   # Load datasets
   data = load_data()
   
   # Load models
   models = load_models(data['emotion_labels'])
   ```


## Models

This project evaluates two multilingual models:

- **mBERT** (bert-base-multilingual-cased): A multilingual version of BERT
- **AfriBERTa** (castorini/afriberta_base): A RoBERTa model fine-tuned on African languages

