Install a new environment:
`python -m venv venv`

Activating the environment:
`venv\Scripts\activate`

Install everything from the environment
`pip install -r requirements.txt`

## BASELINE IMPLEMENTATION

A set of data processing scripts developed in my dissertation was used as a baseline for this pipeline:

**Author**: Ismukhamedova A., Uvalieva I., Belginova S.  
**Work**: Integrating machine learning in electronic health passport based on WHO study and healthcare resources  
**Where published**: Information Fusion, 2023. DOI: [10.1016/j.imu.2023.101428](https://doi.org/10.1016/j.imu.2023.101428)  
**Source code**: https://github.com/AikoIs/Data.git  
**MIMIC version used**: MIMIC‑III  
Labevents table schema: https://mit-lcp.github.io/mimic-schema-spy/tables/labevents.html

All scripts in the `main` branch of this repository are extensions and improvements to the baseline modules described above.

**Key improvements over the baseline**:
Transition from MIMIC‑III to MIMIC‑IV (https://physionet.org/content/mimic4wdb/0.1.0/)  
- Extended EDA (module `eda_improved.py`)  
- More flexible and modular `preprocessing_improved.py`:
  - Additional IQR outlier filtering  
  - Log transformations for asymmetric features  
- Centralized scaling system (Min–Max, RobustScaler)  
- New feature engineering steps added (`feature_engineering_improved.py`)  
- Updated model training module (`model_training_improved.py`) supporting XGBoost, CatBoost, etc.

See also [CHANGELOG.md](./CHANGELOG.md) for the complete version history and differences.


## Project structure
```project/
├── data/                  # data storage directory
│   ├── raw/               # raw data, the source datasets are placed here
│   ├── processed/         # processed data, datasets are placed here after processing
│   └── eda_results/       # these are the EDA results in the form of pictures
├── models/                # a folder for saving models - only pictures of the results are presented here. In general, your prediction datasets are also stored here according to the execution of the algorithm.
├── modules/                            # modules for different parts of the project
│   ├── eda_improved.py                 # exploratory data analysis
│   ├── preprocessing_improved.py       # data preprocessing
│   ├── feature_engineering_improved.py # creation/transformation of new features, selection of features, generation of additional variables
│   ├── model_training_improved.py      # model training (Cut Boost, XGBoost, sklearn or any other library), saving the model, displaying metrics for validation, etc.
│   ├── model_tuning_improved.py        # Interpretation of the trained model: importance of features, SHAP, partial dependence plots
├── requirements.txt       
└── README.md              # description of the project
```

Explanation for launching from the modules folder:
You can run files from the modules folder one at a time by activating the virtual environment beforehand.

The order:
1) eda_improved → 2) preprocessing_improved → 3) feature_engineering_improved → 4) model_training_improved → 5) model_interpret_improved

In the project structure, there are folders that contain the result images. 
Intermediate datasets will also be stored in such folders after the algorithms are executed.

### About the data
The data is the property of the MIT project and therefore cannot be used in this repository.
But we can attach a note about this: https://physionet.org/content/mimiciv/3.1/
We will also attach a link to an article describing the study itself.:


This research was funded by the Science Committee of the Ministry of Education and Science of the Republic  of  Kazakhstan  project  number  AP22683316  “Application  of  machine  learning  algorithms  for medical decision support systems”.



