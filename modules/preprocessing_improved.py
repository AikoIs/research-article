import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Класс для предварительной обработки данных MIMIC.
    Включает улучшенную обработку временных данных, валидацию и обработку лабораторных показателей.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация препроцессора данных.
        
        Args:
            config: Словарь с конфигурацией, включающий пути к файлам и параметры обработки
        """
        self.config = config
        self.data = {}
        self.metrics = {
            'start_time': datetime.now(),
            'processed_records': 0,
            'missing_values': {},
            'feature_stats': {}
        }
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка системы логирования"""
        try:
            # Используем временную директорию системы для логов
            import tempfile
            log_dir = tempfile.gettempdir()
            log_file = os.path.join(log_dir, "preprocessing.log")
            
            # Настраиваем логирование
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            return logging.getLogger(__name__)
        
        except Exception as e:
            # В случае проблем используем только вывод в консоль
            print(f"Warning: Could not setup file logging: {str(e)}")
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
            return logging.getLogger(__name__)
    
    def load_data(self):
        """
        Загружает все необходимые датасеты
        """
        try:
            self.logger.info("Начало загрузки данных")
            data_files = {
                'admissions': 'admissions.csv',
                'patients': 'patients.csv',
                'labevents': 'labevents.csv'
            }
            
            for key, filename in data_files.items():
                self.logger.info(f"Загрузка файла: {filename}")
                file_path = os.path.join(self.config['data_raw_dir'], filename)
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Файл не найден: {file_path}")
                    
                # Загружаем данные и сохраняем их в словарь self.data
                self.data[key] = pd.read_csv(file_path)
                self.logger.info(f"Загружен {filename}, форма: {self.data[key].shape}")
                
            self.logger.info("Все файлы успешно загружены")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    def validate_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверяет корректность временных данных
        
        Args:
            df: DataFrame с временными данными
            
        Returns:
            DataFrame с валидированными временными данными
        """
        for col in ['ADMITTIME', 'DISCHTIME', 'DEATHTIME']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Проверка на будущие даты
                future_dates = df[col] > pd.Timestamp.now()
                if future_dates.any():
                    self.logger.warning(f"Найдены будущие даты в {col}")
                    df.loc[future_dates, col] = pd.NaT
                
                # Проверка порядка дат
                if 'ADMITTIME' in df.columns and 'DISCHTIME' in df.columns:
                    invalid_intervals = df['DISCHTIME'] < df['ADMITTIME']
                    if invalid_intervals.any():
                        self.logger.warning(f"Найдены некорректные интервалы госпитализации")
                        df.loc[invalid_intervals, 'DISCHTIME'] = pd.NaT
        
        return df
    
    def process_lab_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает лабораторные показатели
        
        Args:
            df: DataFrame с лабораторными показателями
            
        Returns:
            DataFrame с обработанными показателями
        """
        # Определение референсных значений
        lab_tests = {
            50809: {'name': 'Glucose', 'unit': 'mg/dL', 'normal_range': (70, 100)},
            50810: {'name': 'Creatinine', 'unit': 'mg/dL', 'normal_range': (0.6, 1.2)},
            50811: {'name': 'Hemoglobin', 'unit': 'g/dL', 'normal_range': (12, 16)},
            50822: {'name': 'Potassium', 'unit': 'mEq/L', 'normal_range': (3.5, 5.0)},
            50931: {'name': 'WBC', 'unit': 'K/uL', 'normal_range': (4.5, 11.0)}
        }
        
        # Проверка выхода за референсные значения
        for test_id, test_info in lab_tests.items():
            mask = df['ITEMID'] == test_id
            if mask.any():
                values = df.loc[mask, 'VALUE']
                lower, upper = test_info['normal_range']
                abnormal = (values < lower) | (values > upper)
                df.loc[mask, 'is_abnormal'] = abnormal
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает пропущенные значения
        
        Args:
            df: DataFrame с пропущенными значениями
            
        Returns:
            DataFrame с обработанными пропущенными значениями
        """
        # Сохраняем статистику до обработки
        self.metrics['missing_values']['before'] = df.isnull().sum().to_dict()
        
        # Числовые признаки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(
                strategy=self.config.get('numeric_fill', 'median')
            )
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Категориальные признаки
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].fillna(
                self.config.get('cat_fill_value', 'UNKNOWN')
            )
        
        # Сохраняем статистику после обработки
        self.metrics['missing_values']['after'] = df.isnull().sum().to_dict()
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает новые признаки
        
        Args:
            df: DataFrame для создания признаков
            
        Returns:
            DataFrame с новыми признаками
        """
        # Временные признаки
        if 'ADMITTIME' in df.columns:
            df['admission_hour'] = df['ADMITTIME'].dt.hour
            df['admission_day_of_week'] = df['ADMITTIME'].dt.dayofweek
            df['admission_month'] = df['ADMITTIME'].dt.month
            df['admission_quarter'] = df['ADMITTIME'].dt.quarter
        
        # Длительность госпитализации
        if 'ADMITTIME' in df.columns and 'DISCHTIME' in df.columns:
            df['los_days'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / (24*60*60)
        
        return df
    
    def save_results(self) -> None:
        """Сохраняет результаты обработки"""
        try:
            output_path = os.path.join(self.config['output_dir'], 'processed_data.csv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if 'processed' not in self.data:
                raise KeyError("Отсутствуют обработанные данные")
                
            self.data['processed'].to_csv(output_path, index=False)
            
            # Сохраняем метрики
            metrics_path = os.path.join(
                self.config['output_dir'],
                'preprocessing_metrics.json'
            )
            pd.DataFrame(self.metrics).to_json(metrics_path)
            
            self.logger.info(f"Результаты сохранены в {output_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            raise
    
    def process(self) -> None:
        """Основной метод обработки данных"""
        try:
            self.logger.info("Начало обработки данных")
            
            self.load_data()
            self.merge_datasets()
            self.clean_data()
            self.data['processed'] = self.merged_data
            self.save_results()
            
            # Завершение
            self.metrics['end_time'] = datetime.now()
            self.metrics['total_time'] = (
                self.metrics['end_time'] - self.metrics['start_time']
            ).total_seconds()
            
            self.logger.info("Обработка данных завершена успешно")
        except Exception as e:
            self.logger.error(f"Ошибка при обработке данных: {str(e)}")
            raise
    
    def merge_datasets(self):
        """
        Объединяет загруженные датасеты в один
        """
        try:
            self.logger.info("Начало объединения датасетов")
            
            # Проверяем наличие всех необходимых датасетов
            required_datasets = ['admissions', 'patients', 'labevents']
            for dataset in required_datasets:
                if dataset not in self.data:
                    raise KeyError(f"Отсутствует необходимый датасет: {dataset}")
            
            # Объединяем admissions и patients по subject_id
            merged = pd.merge(
                self.data['admissions'],
                self.data['patients'],
                on='subject_id',
                how='left'
            )
            
            # Объединяем с labevents
            self.merged_data = pd.merge(
                merged,
                self.data['labevents'],
                on=['subject_id', 'hadm_id'],
                how='left'
            )
            
            self.logger.info(f"Датасеты успешно объединены. Итоговая форма: {self.merged_data.shape}")
        except Exception as e:
            self.logger.error(f"Ошибка при объединении датасетов: {str(e)}")
            raise

    def clean_data(self):
        """
        Очищает и предобрабатывает объединенные данные
        """
        try:
            self.logger.info("Начало очистки данных")
            
            # Проверяем, что данные существуют
            if not hasattr(self, 'merged_data'):
                raise AttributeError("Отсутствуют данные для очистки. Сначала выполните merge_datasets()")
            
            # Удаляем дубликаты
            self.logger.info("Удаление дубликатов...")
            initial_rows = len(self.merged_data)
            self.merged_data = self.merged_data.drop_duplicates()
            self.logger.info(f"Удалено {initial_rows - len(self.merged_data)} дубликатов")
            
            # Обработка пропущенных значений
            self.logger.info("Обработка пропущенных значений...")
            # Для числовых колонок заполняем медианой
            numeric_columns = self.merged_data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                self.merged_data[col] = self.merged_data[col].fillna(self.merged_data[col].median())
            
            # Для категориальных колонок заполняем модой
            categorical_columns = self.merged_data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                self.merged_data[col] = self.merged_data[col].fillna(self.merged_data[col].mode()[0])
            
            # Удаляем строки с некорректными значениями
            self.logger.info("Удаление некорректных значений...")
            self.merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan)
            self.merged_data = self.merged_data.dropna(subset=['subject_id', 'hadm_id'])  # Важные идентификаторы
            
            # Приведение типов данных
            self.logger.info("Приведение типов данных...")
            for col in ['subject_id', 'hadm_id']:
                self.merged_data[col] = self.merged_data[col].astype('int64')
            
            self.logger.info(f"Очистка данных завершена. Итоговая форма: {self.merged_data.shape}")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке данных: {str(e)}")
            raise

if __name__ == "__main__":
    # Пример конфигурации
    config = {
        'data_raw_dir': '../data/raw',
        'output_dir': '../data/processed',
        'input_files': ['admissions.csv', 'patients.csv', 'labevents.csv'],
        'chunk_size': 50000,
        'numeric_fill': 'median',
        'cat_fill_value': 'UNKNOWN'
    }
    
    # Создаем и запускаем препроцессор
    preprocessor = DataPreprocessor(config)
    preprocessor.process() 