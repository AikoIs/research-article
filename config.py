"""
Конфигурационный файл для препроцессора данных
"""

config = {
    'data_raw_dir': '../data/raw',  # Путь относительно папки project
    'output_dir': '../data/processed',  # Путь относительно папки project
    'input_files': ['admissions.csv', 'patients.csv', 'labevents.csv'],
    'numeric_fill': 'median',
    'cat_fill_value': 'UNKNOWN'
} 