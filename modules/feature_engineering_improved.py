import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import holidays
import json
from scipy import stats
import warnings
import gc
gc.enable()  # Включаем сборщик мусора
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Класс для создания признаков из медицинских данных
    
    Attributes:
        config (Dict[str, Any]): Конфигурация
        features (Dict[str, List[str]]): Созданные признаки по категориям
        metrics (Dict[str, Any]): Метрики качества признаков
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация инженера признаков
        
        Args:
            config: Словарь с настройками:
                - input_path: путь к входному файлу
                - output_path: путь для сохранения результатов
                - n_components: число компонент для PCA
                - n_interactions: число взаимодействий для отбора
                - text_max_features: максимальное число текстовых признаков
        """
        self.config = config
        self.features = {
            'temporal': [],
            'lab': [],
            'categorical': [],
            'text': [],
            'interactions': []
        }
        self.metrics = {
            'start_time': datetime.now(),
            'feature_importance': {},
            'stability_metrics': {},
            'correlation_metrics': {},
            'selected_features': []
        }
        
        # Создаем директории
        os.makedirs(os.path.dirname(self.config['output_path']), exist_ok=True)
        
        # Настройка логирования
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Загружаем праздники
        self.holidays_us = holidays.US()
        
        # Инициализация преобразователей
        self.scalers = {}
        self.encoders = {}
    
    def setup_logging(self) -> None:
        """Настройка системы логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('feature_engineering.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
        """
        Загружает предобработанные данные
        
        Returns:
            DataFrame с загруженными данными
        """
        try:
            input_path = os.path.join('..', 'data', 'processed', 'processed_data.csv')
            
            self.logger.info(f"Попытка загрузки файла из: {os.path.abspath(input_path)}")
            
            if not os.path.exists(input_path):
                dir_path = os.path.dirname(input_path)
                if not os.path.exists(dir_path):
                    self.logger.error(f"Директория не существует: {dir_path}")
                    os.makedirs(dir_path, exist_ok=True)
                    self.logger.info(f"Создана директория: {dir_path}")
                raise FileNotFoundError(f"Файл не найден: {input_path}")
            
            df = pd.read_csv(input_path)
            self.logger.info(f"Данные успешно загружены, форма: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает временные признаки
        
        Args:
            df: DataFrame с временными колонками
            
        Returns:
            DataFrame с новыми признаками
        """
        time_cols = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME']
        
        for col in time_cols:
            if col in df.columns:
                # Базовые временные признаки
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                
                # Циклические признаки
                df[f'{col}_hour_sin'] = np.sin(2 * np.pi * df[f'{col}_hour'] / 24)
                df[f'{col}_hour_cos'] = np.cos(2 * np.pi * df[f'{col}_hour'] / 24)
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12)
                
                # Праздники и выходные
                df[f'{col}_is_holiday'] = df[col].apply(
                    lambda x: 1 if x.date() in self.holidays_us else 0)
                df[f'{col}_is_weekend'] = df[f'{col}_dayofweek'].isin([5, 6]).astype(int)
                
                # Сохраняем имена созданных признаков
                self.features['temporal'].extend([
                    f'{col}_hour', f'{col}_day', f'{col}_month', f'{col}_year',
                    f'{col}_dayofweek', f'{col}_hour_sin', f'{col}_hour_cos',
                    f'{col}_month_sin', f'{col}_month_cos', f'{col}_is_holiday',
                    f'{col}_is_weekend'
                ])
        
        # Интервалы между событиями
        if 'ADMITTIME' in df.columns and 'DISCHTIME' in df.columns:
            df['los_days'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / (24*60*60)
            self.features['temporal'].append('los_days')
        
        if 'ADMITTIME' in df.columns and 'DEATHTIME' in df.columns:
            df['time_to_death'] = (df['DEATHTIME'] - df['ADMITTIME']).dt.total_seconds() / (24*60*60)
            self.features['temporal'].append('time_to_death')
        
        return df
    
    def create_lab_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает признаки из лабораторных данных
        
        Args:
            df: DataFrame с лабораторными показателями
            
        Returns:
            DataFrame с новыми признаками
        """
        lab_cols = [col for col in df.columns if 'lab_' in col.lower()]
        
        for col in lab_cols:
            # Базовая статистика
            stats = df.groupby('HADM_ID')[col].agg([
                'mean', 'std', 'min', 'max', 'median',
                ('range', lambda x: x.max() - x.min()),
                ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))
            ]).add_prefix(f'{col}_')
            
            # Добавляем статистики в датафрейм
            df = df.merge(stats, on='HADM_ID', how='left')
            
            # Тренды
            df[f'{col}_trend'] = df.groupby('HADM_ID')[col].diff()
            df[f'{col}_trend_sign'] = np.sign(df[f'{col}_trend'])
            
            # Выход за референсные значения (если есть)
            if f'{col}_lower' in df.columns and f'{col}_upper' in df.columns:
                df[f'{col}_is_abnormal'] = ((df[col] < df[f'{col}_lower']) | 
                                          (df[col] > df[f'{col}_upper'])).astype(int)
            
            # Сохраняем имена признаков
            self.features['lab'].extend([
                f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max',
                f'{col}_median', f'{col}_range', f'{col}_iqr',
                f'{col}_trend', f'{col}_trend_sign'
            ])
            if f'{col}_is_abnormal' in df.columns:
                self.features['lab'].append(f'{col}_is_abnormal')
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает категориальные признаки с оптимизацией памяти
        """
        try:
            self.logger.info("Создание категориальных признаков...")
            
            # Создаем признаки частями
            chunk_size = 1000000  # Обрабатываем по 1 миллиону строк
            num_chunks = len(df) // chunk_size + 1
            
            for col in ['insurance', 'admission_type', 'admission_location']:
                self.logger.info(f"Обработка колонки: {col}")
                
                # Получаем все уникальные значения для создания признаков
                unique_values = df[col].unique()
                
                # Создаем новые колонки с оптимизированным типом данных
                for value in unique_values:
                    col_name = f"{col}_{value}"
                    # Используем uint8 вместо float64 для бинарных признаков
                    df[col_name] = 0
                    df[col_name] = df[col_name].astype('uint8')
                    
                    # Заполняем значения по частям
                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(df))
                        mask = df.iloc[start_idx:end_idx][col] == value
                        df.loc[df.index[start_idx:end_idx], col_name] = mask.astype('uint8')
                
                # Удаляем исходную колонку для экономии памяти
                df = df.drop(columns=[col])
            
            self.logger.info("Категориальные признаки созданы успешно")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании категориальных признаков: {str(e)}")
            raise
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает признаки из текстовых данных
        
        Args:
            df: DataFrame с текстовыми колонками
            
        Returns:
            DataFrame с новыми признаками
        """
        text_cols = ['ICD_CODE_LIST', 'ALL_DRUGS', 'DEPARTMENTS']
        
        for col in text_cols:
            if col in df.columns:
                # TF-IDF
                tfidf = TfidfVectorizer(
                    max_features=self.config['text_max_features'],
                    stop_words='english'
                )
                text_features = tfidf.fit_transform(df[col].fillna(''))
                
                # Преобразуем в DataFrame
                feature_names = [f'{col}_tfidf_{i}' for i in range(text_features.shape[1])]
                text_df = pd.DataFrame(
                    text_features.toarray(),
                    columns=feature_names,
                    index=df.index
                )
                
                df = pd.concat([df, text_df], axis=1)
                self.features['text'].extend(feature_names)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает признаки взаимодействия с оптимизацией памяти
        """
        try:
            self.logger.info("Создание признаков взаимодействия...")
            
            # Выбираем только числовые колонки
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Ограничиваем количество признаков для взаимодействия
            if len(numeric_cols) > 10:
                self.logger.info("Выбираем только 10 наиболее важных числовых признаков")
                numeric_cols = numeric_cols[:10]  # Берем только первые 10 признаков
            
            # Обрабатываем данные чанками
            chunk_size = 100000
            num_chunks = len(df) // chunk_size + 1
            
            # Инициализируем PolynomialFeatures только для выбранных признаков
            poly = PolynomialFeatures(degree=2, include_bias=False)
            
            # Создаем пустой DataFrame для результатов
            feature_names = []
            first_chunk = True
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                
                # Обрабатываем чанк данных
                chunk = df.iloc[start_idx:end_idx][numeric_cols]
                
                # Для первого чанка получаем имена признаков
                if first_chunk:
                    # Получаем имена признаков из первого чанка
                    poly.fit(chunk)
                    feature_names = poly.get_feature_names_out(numeric_cols)
                    # Создаем пустые колонки в исходном DataFrame
                    for name in feature_names[len(numeric_cols):]:  # Пропускаем исходные признаки
                        df[f'interaction_{name}'] = 0
                    first_chunk = False
                
                # Трансформируем чанк
                poly_features = poly.transform(chunk)
                
                # Обновляем только новые признаки (пропускаем исходные)
                for j, name in enumerate(feature_names[len(numeric_cols):], len(numeric_cols)):
                    df.iloc[start_idx:end_idx, df.columns.get_loc(f'interaction_{name}')] = poly_features[:, j]
                
                # Очищаем память
                del chunk, poly_features
                gc.collect()
                
                if i % 10 == 0:
                    self.logger.info(f"Обработано {end_idx}/{len(df)} строк")
            
            self.logger.info("Признаки взаимодействия созданы успешно")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании признаков взаимодействия: {str(e)}")
            raise
    
    def calculate_feature_importance(self, df: pd.DataFrame) -> None:
        """
        Вычисляет важность признаков
        
        Args:
            df: DataFrame с признаками и целевой переменной
        """
        if 'target' not in df.columns:
            return
        
        # Получаем все созданные признаки
        all_features = []
        for feature_list in self.features.values():
            all_features.extend(feature_list)
        
        # Вычисляем Mutual Information
        mi_scores = mutual_info_classif(df[all_features], df['target'])
        
        # Сохраняем результаты
        self.metrics['feature_importance'] = {
            feature: score for feature, score in zip(all_features, mi_scores)
        }
    
    def calculate_stability_metrics(self, df: pd.DataFrame) -> None:
        """
        Вычисляет метрики стабильности признаков
        
        Args:
            df: DataFrame с признаками
        """
        # Получаем все созданные признаки
        all_features = []
        for feature_list in self.features.values():
            all_features.extend(feature_list)
        
        stability_metrics = {}
        for feature in all_features:
            if feature in df.columns:
                # Базовые статистики
                stats = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'skew': stats.skew(df[feature].dropna()),
                    'kurtosis': stats.kurtosis(df[feature].dropna()),
                    'missing_ratio': df[feature].isnull().mean()
                }
                stability_metrics[feature] = stats
        
        self.metrics['stability_metrics'] = stability_metrics
    
    def calculate_correlation_metrics(self, df: pd.DataFrame) -> None:
        """
        Вычисляет корреляционные метрики
        
        Args:
            df: DataFrame с признаками
        """
        # Получаем все созданные признаки
        all_features = []
        for feature_list in self.features.values():
            all_features.extend(feature_list)
        
        # Вычисляем корреляционную матрицу
        corr_matrix = df[all_features].corr()
        
        # Находим сильно коррелирующие признаки
        high_corr = np.where(np.abs(corr_matrix) > 0.8)
        high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                     for x, y in zip(*high_corr) if x != y]
        
        self.metrics['correlation_metrics'] = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr
        }
    
    def select_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Выбирает наиболее важные признаки с оптимизацией памяти
        """
        try:
            self.logger.info("Начало отбора признаков...")
            self.logger.info(f"Целевая переменная: {target_col}")
            
            # Проверяем наличие целевой переменной
            if target_col not in df.columns:
                available_columns = ', '.join(df.columns)
                self.logger.error(f"Целевая переменная {target_col} не найдена. Доступные колонки: {available_columns}")
                raise KeyError(f"Целевая переменная {target_col} отсутствует в данных")
            
            # Обрабатываем данные чанками
            chunk_size = 100000
            num_chunks = len(df) // chunk_size + 1
            
            # Получаем список числовых колонок без загрузки всех данных
            numeric_cols = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    if col != target_col:  # Исключаем целевую переменную
                        numeric_cols.append(col)
            
            self.logger.info(f"Найдено {len(numeric_cols)} числовых признаков")
            
            # Вычисляем корреляции по чанкам
            correlations = {}
            for col in numeric_cols:
                correlations[col] = 0
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk = df.iloc[start_idx:end_idx]
                
                # Вычисляем корреляции для текущего чанка
                for col in numeric_cols:
                    corr = abs(np.corrcoef(chunk[col], chunk[target_col])[0, 1])
                    correlations[col] += corr / num_chunks
                
                if i % 10 == 0:
                    self.logger.info(f"Обработано {end_idx}/{len(df)} строк при вычислении корреляций")
            
            # Сортируем признаки по корреляции
            sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Выбираем топ-30 признаков
            selected_features = [feat[0] for feat in sorted_features[:30]]
            selected_features.append(target_col)
            
            # Оставляем только выбранные колонки
            df_selected = df[selected_features]
            
            self.logger.info(f"Отобрано {len(selected_features)-1} признаков")
            return df_selected
            
        except Exception as e:
            self.logger.error(f"Ошибка при отборе признаков: {str(e)}")
            raise
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Масштабирует числовые признаки
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            DataFrame с масштабированными признаками
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            self.scalers['standard'] = StandardScaler()
            df[numeric_cols] = self.scalers['standard'].fit_transform(df[numeric_cols])
        
        return df
    
    def save_results(self, df: pd.DataFrame) -> None:
        """
        Сохраняет результаты и метрики
        
        Args:
            df: DataFrame с результатами
        """
        try:
            # Создаем директорию для результатов
            os.makedirs(os.path.dirname(self.config['output_path']), exist_ok=True)
            
            # Сохраняем данные с признаками
            output_path = self.config['output_path']
            df.to_csv(output_path, index=False)
            
            # Сохраняем метрики
            self.metrics['end_time'] = datetime.now()
            self.metrics['total_time'] = (
                self.metrics['end_time'] - self.metrics['start_time']
            ).total_seconds()
            
            metrics_path = os.path.join(
                os.path.dirname(self.config['output_path']),
                'feature_engineering_metrics.json'
            )
            
            # Преобразуем numpy типы в обычные Python типы
            metrics_dict = {}
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    metrics_dict[key] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in value.items()
                    }
                else:
                    metrics_dict[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=4, default=str)
            
            # Сохраняем список признаков без ID-полей
            self.extract_features(df)
            
            self.logger.info(f"Результаты сохранены в {os.path.dirname(self.config['output_path'])}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            raise
    
    def extract_features(self, df: pd.DataFrame) -> None:
        """
        Извлекает и сохраняет список признаков без ID-полей
        
        Args:
            df: DataFrame с признаками
        """
        try:
            self.logger.info("Извлечение списка признаков без ID-полей...")
            
            # Получаем все колонки из датафрейма
            all_columns = df.columns.tolist()
            
            # Определяем паттерны для ID-полей
            id_patterns = [
                'ID', '_id', 'TIME', 'DATE', 'ADMIT', 'DISCH', 
                'DEATH', 'SUBJECT', 'HADM', 'ICUSTAY', 'ROW'
            ]
            
            # Фильтруем ID-поля
            filtered_features = []
            id_fields = []
            
            for col in all_columns:
                is_id_field = False
                for pattern in id_patterns:
                    if pattern.upper() in col.upper():
                        is_id_field = True
                        id_fields.append(col)
                        break
                
                if not is_id_field:
                    filtered_features.append(col)
            
            # Сохраняем список признаков в CSV
            features_path = os.path.join(
                os.path.dirname(self.config['output_path']),
                'model_features.csv'
            )
            
            features_df = pd.DataFrame({'feature': filtered_features})
            features_df.to_csv(features_path, index=False)
            
            # Логируем информацию
            self.logger.info(f"Всего признаков: {len(all_columns)}")
            self.logger.info(f"ID-полей отфильтровано: {len(id_fields)}")
            self.logger.info(f"Сохранено признаков без ID-полей: {len(filtered_features)}")
            self.logger.info(f"Список признаков сохранен в {features_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении признаков: {str(e)}")
            raise
    
    def process(self) -> None:
        """Основной метод создания признаков"""
        try:
            self.logger.info("Начало создания признаков")
            
            # Загрузка данных
            df = self.load_data()
            
            # Создание признаков
            df = self.create_temporal_features(df)
            df = self.create_lab_features(df)
            df = self.create_categorical_features(df)
            df = self.create_text_features(df)
            df = self.create_interaction_features(df)
            
            # Отбор признаков
            if 'target_col' in self.config:
                df = self.select_features(df, self.config['target_col'])
            
            # Масштабирование признаков
            df = self.scale_features(df)
            
            # Сохранение результатов
            self.save_results(df)
            
            self.logger.info("Создание признаков завершено")
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании признаков: {str(e)}")
            raise

if __name__ == "__main__":
    # Пример конфигурации
    config = {
        'input_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "..", "data", "processed", "preprocessed_data.csv"),
        'output_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "..", "data", "processed", "engineered_features.csv"),
        'n_components': 10,
        'n_interactions': 20,
        'text_max_features': 100,
        'target_col': 'hospital_expire_flag',
        'n_features': 20
    }
    
    # Создаем и запускаем инженера признаков
    engineer = FeatureEngineer(config)
    engineer.process() 