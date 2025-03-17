import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                           precision_score, recall_score, f1_score, fbeta_score,
                           precision_recall_curve, auc, accuracy_score, roc_curve)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Generator, Optional
import joblib
import json
import logging
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout
import psutil
import warnings
import gc
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Класс для обучения и оценки моделей машинного обучения
    
    Attributes:
        config (Dict[str, Any]): Конфигурация
        models (Dict[str, Any]): Словарь моделей
        results (Dict[str, Any]): Результаты обучения
        best_model (Any): Лучшая модель
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация тренера моделей
        
        Args:
            config: Словарь с настройками:
                - input_path: путь к входному файлу
                - output_path: путь для сохранения результатов
                - chunk_size: размер чанка для обработки
                - n_trials: число попыток для оптимизации
                - cv_folds: число фолдов для кросс-валидации
                - balance_strategy: стратегия балансировки классов
        """
        self.config = config
        self.models = {}
        self.results = {
            'training_metrics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'hyperparameters': {},
            'resource_usage': {}
        }
        self.best_model = None
        
        # Создаем директории
        os.makedirs(os.path.dirname(self.config['output_path']), exist_ok=True)
        
        # Настройка логирования
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _create_neural_net(self, input_dim: int) -> Sequential:
        """
        Создает архитектуру нейронной сети
        
        Args:
            input_dim: Размерность входного слоя
            
        Returns:
            Модель Keras
        """
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def prepare_models(self) -> Dict[str, Any]:
        """
        Создает набор моделей для обучения
        
        Returns:
            Словарь моделей
        """
        self.logger.info("Подготовка моделей...")
        
        # Базовые модели
        models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42),
            'catboost': cb.CatBoostClassifier(random_state=42, verbose=False)
        }
        
        # Нейронная сеть будет создана позже, когда будет известна размерность входа
        self.models = models
        return models
    
    def create_balanced_pipeline(self, model: Any) -> ImbPipeline:
        """
        Создает пайплайн с балансировкой классов
        
        Args:
            model: Базовая модель
            
        Returns:
            Pipeline с балансировкой
        """
        try:
            balance_strategy = self.config.get('balance_strategy', 'class_weight')
            
            if balance_strategy == 'class_weight':
                # Используем веса классов
                if isinstance(model, RandomForestClassifier):
                    model.set_params(class_weight='balanced')
                elif isinstance(model, xgb.XGBClassifier):
                    # Для XGBoost используем scale_pos_weight
                    model.set_params(scale_pos_weight=self.class_weights[1]/self.class_weights[0])
                
                return ImbPipeline([
                    ('classifier', model)
                ])
                
            elif balance_strategy == 'sampling':
                # Используем SMOTE для балансировки
                return ImbPipeline([
                    ('sampler', SMOTE(random_state=self.config['random_state'])),
                    ('classifier', model)
                ])
                
            else:
                # По умолчанию используем веса классов
                if isinstance(model, RandomForestClassifier):
                    model.set_params(class_weight='balanced')
                elif isinstance(model, xgb.XGBClassifier):
                    model.set_params(scale_pos_weight=self.class_weights[1]/self.class_weights[0])
                
                return ImbPipeline([
                    ('classifier', model)
                ])
                
        except Exception as e:
            self.logger.error(f"Ошибка при создании пайплайна: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Оптимизирует гиперпараметры модели
        """
        self.logger.info(f"Оптимизация гиперпараметров для {model_name}...")
        
        def objective(trial):
            params = {}
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
                classifier = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            
            elif model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                classifier = GradientBoostingClassifier(**params, random_state=42)
            
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                classifier = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
            
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
                }
                classifier = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
            
            elif model_name == 'catboost':
                # Более тщательная предобработка данных для CatBoost
                X_processed = X.copy()
                for col in X_processed.columns:
                    # Преобразуем все столбцы в числовой формат
                    if X_processed[col].dtype == 'object':
                        try:
                            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                        except:
                            # Если не удается преобразовать в числа, используем факторизацию
                            X_processed[col] = pd.factorize(X_processed[col])[0]
                
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'verbose': False,
                    'allow_writing_files': False
                }
                classifier = cb.CatBoostClassifier(**params, random_state=42)
                
                # Используем упрощенный пайплайн для CatBoost
                pipeline = ImbPipeline([
                    ('classifier', classifier)
                ])
                
                try:
                    scores = cross_val_score(pipeline, X_processed, y, scoring='roc_auc', cv=cv, n_jobs=-1)
                    return scores.mean()
                except Exception as e:
                    self.logger.error(f"Ошибка в кросс-валидации CatBoost: {str(e)}")
                    return float('-inf')
            
            else:
                # Код для других моделей остается без изменений
                pipeline = ImbPipeline([
                    ('scaler', StandardScaler()),
                    ('sampler', SMOTE(random_state=42)),
                    ('classifier', classifier)
                ])
                
                try:
                    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
                    return scores.mean()
                except Exception as e:
                    self.logger.error(f"Ошибка в кросс-валидации: {str(e)}")
                    return float('-inf')
        
        # Создаем исследование Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['n_trials'])
        
        # Сохраняем результаты
        self.results['hyperparameters'][model_name] = study.best_params
        
        return study.best_params
    
    def train_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """
        Обучает модель и собирает метрики
        
        Args:
            model: Модель для обучения
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Обученная модель и метрики
        """
        # Замеряем время и память
        start_time = datetime.now()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Обучаем модель
        model.fit(X_train, y_train)
        
        # Получаем предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Собираем метрики
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Добавляем метрики ресурсов
        end_time = datetime.now()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics['resource_usage'] = {
            'training_time': (end_time - start_time).total_seconds(),
            'memory_usage_mb': final_memory - initial_memory
        }
        
        return model, metrics
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет расширенный набор метрик
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            y_pred_proba: Вероятности предсказаний
            
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        # Базовые метрики
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        
        # Precision-Recall кривая
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)
        
        # F-beta score
        metrics['f2'] = fbeta_score(y_true, y_pred, beta=2)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1]
        }
        
        # Калибровка
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        metrics['calibration'] = {
            'true_probs': prob_true.tolist(),
            'predicted_probs': prob_pred.tolist()
        }
        
        return metrics
    
    def create_stacking_model(self, X: pd.DataFrame) -> StackingClassifier:
        """
        Создает модель стекинга
        
        Args:
            X: Признаки для определения размерности входа
            
        Returns:
            Модель стекинга
        """
        # Добавляем нейронную сеть
        input_dim = X.shape[1]
        self.models['neural_net'] = KerasClassifier(
            build_fn=lambda: self._create_neural_net(input_dim),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Создаем стекинг
        estimators = [
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting']),
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm']),
            ('cb', self.models['catboost']),
            ('nn', self.models['neural_net'])
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
    
    def train_and_evaluate(self) -> None:
        """
        Обучает и оценивает модели
        """
        try:
            # Загружаем данные
            df = self.load_data()
            
            # Уменьшаем размер датасета
            sample_size = 1000000  # берем первый миллион записей
            df = df.head(sample_size)
            self.logger.info(f"Используем уменьшенный датасет: {len(df)} записей")
            
            # Проверяем и преобразуем метки классов
            target_col = 'hospital_expire_flag'
            unique_labels = sorted(df[target_col].unique())
            self.logger.info(f"Уникальные метки в данных: {unique_labels}")
            
            # Преобразуем метки в 0 и 1
            label_map = {
                unique_labels[0]: 0,  # меньшее значение -> 0
                unique_labels[1]: 1   # большее значение -> 1
            }
            df[target_col] = df[target_col].map(label_map)
            self.logger.info(f"Метки преобразованы: {label_map}")
            
            # Проверяем результат
            self.logger.info(f"Метки после преобразования: {sorted(df[target_col].unique())}")
            
            # Разделяем на признаки и целевую переменную
            y = df[target_col]
            
            # Исключаем ID-поля из признаков
            id_patterns = ['TIME', 'DATE', 'ID', '_id', 'ADMIT', 'DISCH', 
                          'DEATH', 'SUBJECT', 'HADM', 'ICUSTAY', 'ROW']
            
            # Проверяем наличие файла с отфильтрованными признаками
            model_features_path = os.path.join(os.path.dirname(self.config['input_path']), 'model_features.csv')
            
            if os.path.exists(model_features_path):
                self.logger.info(f"Найден файл с отфильтрованными признаками: {model_features_path}")
                feature_df = pd.read_csv(model_features_path)
                feature_list = feature_df['feature'].tolist()
                
                # Проверяем, что все признаки есть в датафрейме
                available_features = [f for f in feature_list if f in df.columns]
                self.logger.info(f"Доступно {len(available_features)} из {len(feature_list)} признаков")
                
                # Используем только доступные признаки
                X = df[available_features]
            else:
                self.logger.info("Файл с отфильтрованными признаками не найден, выполняем фильтрацию вручную")
                
                # Фильтруем ID-поля
                filtered_columns = []
                id_fields = []
                
                for col in df.columns:
                    if col == target_col:
                        continue
                        
                    is_id_field = False
                    for pattern in id_patterns:
                        if pattern.upper() in col.upper():
                            is_id_field = True
                            id_fields.append(col)
                            break
                    
                    if not is_id_field:
                        filtered_columns.append(col)
                
                self.logger.info(f"Отфильтровано {len(id_fields)} ID-полей")
                self.logger.info(f"Используется {len(filtered_columns)} признаков")
                
                # Используем отфильтрованные признаки
                X = df[filtered_columns]
            
            # Сохраняем список признаков для последующей интерпретации
            features_path = os.path.join(self.config['output_path'], 'training_features.csv')
            pd.DataFrame({'feature': X.columns}).to_csv(features_path, index=False)
            self.logger.info(f"Сохранен список признаков в {features_path}")
            
            # Проверяем баланс классов
            class_counts = y.value_counts()
            self.logger.info(f"Распределение классов:\n{class_counts}")
            
            # Вычисляем веса классов
            self.class_weights = dict(enumerate(class_counts.sum() / (len(class_counts) * class_counts)))
            self.logger.info(f"Веса классов: {self.class_weights}")
            
            # Разделяем на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=self.config['random_state'],
                stratify=y
            )
            
            # Проверяем распределение в обучающей выборке
            train_counts = pd.Series(y_train).value_counts()
            self.logger.info(f"Распределение классов в обучающей выборке:\n{train_counts}")
            
            # Создаем и обучаем модели
            self.models = {}
            best_test_score = 0
            best_model_name = None
            
            for name, model in self.prepare_models().items():
                try:
                    self.logger.info(f"Обучение модели {name}...")
                    
                    # Оптимизация гиперпараметров
                    self.logger.info(f"Оптимизация гиперпараметров для {name}...")
                    best_params = self.optimize_hyperparameters(name, X_train, y_train)
                    
                    if best_params:
                        # Обновляем параметры модели
                        model.set_params(**best_params)
                        
                        # Создаем пайплайн
                        pipeline = self.create_balanced_pipeline(model)
                        
                        # Обучаем модель
                        pipeline.fit(X_train, y_train)
                        
                        # Сохраняем модель
                        self.models[name] = pipeline
                        
                        # Оцениваем модель
                        train_score = pipeline.score(X_train, y_train)
                        test_score = pipeline.score(X_test, y_test)
                        
                        self.logger.info(f"Модель {name}:")
                        self.logger.info(f"Train score: {train_score:.4f}")
                        self.logger.info(f"Test score: {test_score:.4f}")
                        
                        # Сохраняем модель и важность признаков
                        self.save_model(name, pipeline, X)
                        
                        # Оцениваем модель и сохраняем метрики
                        metrics = self.evaluate_model(name, pipeline, X_test, y_test)
                        
                        # Обновляем лучшую модель
                        if test_score > best_test_score:
                            best_test_score = test_score
                            best_model_name = name
                            
                except Exception as e:
                    self.logger.error(f"Ошибка при обучении модели {name}: {str(e)}")
                    continue
            
            # Сохраняем лучшую модель
            if best_model_name:
                best_model = self.models[best_model_name]
                joblib.dump(best_model, os.path.join(self.config['output_path'], 'best_model.joblib'))
                self.logger.info(f"Сохранена лучшая модель: {best_model_name} (test score: {best_test_score:.4f})")
            
            self.logger.info("Обучение моделей завершено")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении моделей: {str(e)}")
            raise
    
    def save_results(self) -> None:
        """
        Сохраняет результаты обучения
        """
        # Сохраняем метрики
        results_path = os.path.join(
            self.config['output_path'],
            'training_results.json'
        )
        
        # Преобразуем numpy типы в обычные Python типы
        results_dict = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_dict[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            else:
                results_dict[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=4, default=str)
        
        # Создаем визуализации
        self.create_visualizations()
        
        self.logger.info(f"Результаты сохранены в {self.config['output_path']}")
    
    def create_visualizations(self) -> None:
        """
        Создает визуализации результатов
        """
        # ROC кривые
        plt.figure(figsize=(10, 6))
        for name, metrics in self.results['training_metrics'].items():
            if 'roc_curve' in metrics:
                plt.plot(
                    metrics['roc_curve']['fpr'],
                    metrics['roc_curve']['tpr'],
                    label=f"{name} (AUC = {metrics['roc_auc']:.3f})"
                )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(os.path.join(self.config['output_path'], 'roc_curves.png'))
        plt.close()
        
        # Сравнение метрик
        metrics_comparison = pd.DataFrame(
            {name: metrics for name, metrics in self.results['training_metrics'].items()}
        ).T[['accuracy', 'roc_auc', 'precision', 'recall', 'f1']]
        
        plt.figure(figsize=(12, 6))
        metrics_comparison.plot(kind='bar')
        plt.title('Model Metrics Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_path'], 'metrics_comparison.png'))
        plt.close()
        
        # Время обучения и память
        resource_usage = pd.DataFrame(
            {name: metrics['resource_usage'] 
             for name, metrics in self.results['training_metrics'].items()}
        ).T
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        resource_usage['training_time'].plot(kind='bar', ax=ax1)
        ax1.set_title('Training Time (seconds)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        resource_usage['memory_usage_mb'].plot(kind='bar', ax=ax2)
        ax2.set_title('Memory Usage (MB)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_path'], 'resource_usage.png'))
        plt.close()

    def load_data(self) -> pd.DataFrame:
        """
        Загружает данные из CSV файла
        """
        try:
            self.logger.info(f"Загрузка данных из {self.config['input_path']}")
            
            # Проверяем существование файла
            if not os.path.exists(self.config['input_path']):
                self.logger.error(f"Файл {self.config['input_path']} не найден")
                raise FileNotFoundError(f"Файл {self.config['input_path']} не существует")
            
            # Загружаем данные
            df = pd.read_csv(self.config['input_path'])
            
            # Проверяем, что данные не пустые
            if df.empty:
                self.logger.error("Загруженный датасет пуст")
                raise ValueError("Датасет не содержит данных")
            
            self.logger.info(f"Загружено {len(df)} записей и {len(df.columns)} признаков")
            
            # Удаляем строки с пропущенными значениями
            df_cleaned = df.dropna()
            
            if len(df_cleaned) < len(df):
                self.logger.warning(f"Удалено {len(df) - len(df_cleaned)} строк с пропущенными значениями")
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise

    def evaluate_model(self, name: str, model, X_test, y_test) -> Dict:
        """
        Оценивает модель на тестовых данных
        
        Args:
            name: Название модели
            model: Обученная модель
            X_test: Тестовые признаки
            y_test: Тестовые метки
        
        Returns:
            Dict: Метрики качества модели
        """
        try:
            self.logger.info(f"Оценка модели {name}...")
            
            # Получаем предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Вычисляем метрики
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Логируем результаты
            self.logger.info(f"Метрики для модели {name}:")
            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
            
            # Сохраняем метрики
            metrics_path = os.path.join(self.config['output_path'], f'{name}_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Создаем и сохраняем ROC-кривую
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc="lower right")
            
            # Сохраняем график
            plt.savefig(os.path.join(self.config['output_path'], f'{name}_roc_curve.png'))
            plt.close()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка при оценке модели {name}: {str(e)}")
            raise

    def save_model(self, name: str, model, X: pd.DataFrame) -> None:
        """
        Сохраняет модель и важность признаков
        
        Args:
            name: Название модели
            model: Обученная модель
            X: Признаки
        """
        try:
            self.logger.info(f"Сохранение модели {name}...")
            
            # Сохраняем модель
            model_path = os.path.join(self.config['output_path'], f'{name}_model.joblib')
            joblib.dump(model, model_path)
            self.logger.info(f"Модель {name} сохранена в {model_path}")
            
            # Получаем важность признаков из правильного компонента пайплайна
            if hasattr(model, 'named_steps'):
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.named_steps['classifier'].feature_importances_
                    })
                    feature_importance_path = os.path.join(
                        self.config['output_path'], 
                        f'{name}_feature_importance.csv'
                    )
                    feature_importance.to_csv(feature_importance_path, index=False)
                    self.logger.info(f"Важность признаков сохранена в {feature_importance_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели {name}: {str(e)}")
            raise

    def train_lightgbm(self, X, y):
        try:
            # ... existing code ...
            
            # Изменяем сохранение важности признаков
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['lightgbm'].named_steps['classifier'].feature_importances_
            })
            feature_importance.to_csv('data/processed/lightgbm_feature_importance.csv', index=False)
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели lightgbm: {str(e)}")

    def train_catboost(self, X, y):
        try:
            # Добавляем обработку кодировки данных
            X = X.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
            
            # ... rest of the catboost training code ...
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели catboost: {str(e)}")

if __name__ == "__main__":
    # Получаем абсолютный путь к текущей директории
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # Конфигурация с абсолютными путями
    config = {
        'input_path': os.path.join(project_dir, 'data', 'processed', 'engineered_features.csv'),
        'output_path': os.path.join(project_dir, 'models'),
        'random_state': 42,
        'test_size': 0.2,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'early_stopping_rounds': 10,
        'n_trials': 10,
        'cv_folds': 3,
        'timeout': 600,
        'balance_strategy': 'class_weight',
        'class_weight': 'balanced',
        'model_params': {
            'random_forest': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 4)
            },
            'xgboost': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.5, 1.0)
            },
            'neural_network': {
                'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
                'activation': ['relu', 'tanh'],
                'learning_rate': (0.001, 0.1)
            }
        }
    }
    
    # Создаем директории, если их нет
    os.makedirs(os.path.dirname(config['input_path']), exist_ok=True)
    os.makedirs(config['output_path'], exist_ok=True)
    
    # Создаем и запускаем тренер
    trainer = ModelTrainer(config)
    trainer.train_and_evaluate() 