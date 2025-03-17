import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from alibi.explainers import AnchorTabular
from alibi.explainers import CounterfactualProto
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from typing import Any, List, Tuple, Dict, Optional
from datetime import datetime
import json
from scipy.stats import spearmanr
import logging
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_interpret.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelInterpreter:
    """
    Класс для интерпретации моделей машинного обучения
    
    Attributes:
        config (Dict[str, Any]): Конфигурация
        model (Any): Модель для интерпретации
        data (pd.DataFrame): Данные для интерпретации
        explanations (Dict[str, Any]): Результаты объяснений
        visualizations (Dict[str, Any]): Визуализации
    """
    
    def __init__(self, config_path: str):
        """Инициализация интерпретатора"""
        try:
            self.logger = self._setup_logging()
            
            # Проверяем тип входного параметра
            if isinstance(config_path, dict):
                self.config = config_path
            else:
                self.config = self._load_config(config_path)
            
            self.model = self._load_model()
            self.X, self.y = self._load_data()
            self.feature_names = list(self.X.columns)
            self.logger.info("Интерпретатор успешно инициализирован")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации: {str(e)}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации"""
        try:
            if isinstance(config_path, dict):
                return config_path
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Конфигурация загружена из {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
            raise
    
    def _load_model(self) -> Any:
        """Загрузка модели"""
        model_path = os.path.join(self.config['model_path'], 'best_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
        
        model = joblib.load(model_path)
        self.logger.info(f"Модель загружена из {model_path}")
        
        return model
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка данных для интерпретации"""
        try:
            data = pd.read_csv(self.config['data_path'])
            self.logger.info(f"Загружено {len(data)} записей из {self.config['data_path']}")
            
            # Определяем признаки для использования
            if hasattr(self.model, 'feature_names_in_'):
                features = self.model.feature_names_in_
            elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['classifier'], 'feature_names_in_'):
                features = self.model.named_steps['classifier'].feature_names_in_
            else:
                features = [col for col in data.columns if col != self.config['target_column']]
            
            X = data[features]
            y = data[self.config['target_column']]
            
            return X, y
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    def explain_with_shap(self) -> None:
        """
        Объясняет модель с помощью SHAP values только для sklearn и XGBoost моделей
        """
        try:
            self.logger.info("Вычисление важности признаков...")
            
            # Получаем базовую модель из пайплайна
            if isinstance(self.model, Pipeline):
                model = self.model.named_steps['classifier']
            else:
                model = self.model
            
            # Получаем важность признаков напрямую из модели
            if hasattr(model, 'feature_importances_'):
                # Получаем имена признаков
                feature_names = list(self.data.columns)
                importances = model.feature_importances_
                
                # Проверяем соответствие длин
                if len(feature_names) != len(importances):
                    self.logger.warning(f"Несоответствие длин: признаков {len(feature_names)}, важностей {len(importances)}")
                    # Обрезаем более длинный список
                    min_len = min(len(feature_names), len(importances))
                    feature_names = feature_names[:min_len]
                    importances = importances[:min_len]
                
                # Создаем DataFrame
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
                # Сортируем по важности
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                # Создаем график важности признаков
                plt.figure(figsize=(12, 8))
                plt.bar(range(len(feature_importance)), feature_importance['importance'])
                plt.xticks(
                    range(len(feature_importance)), 
                    feature_importance['feature'], 
                    rotation=45, 
                    ha='right'
                )
                plt.title('Feature Importances')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()
                
                # Сохраняем график
                plt.savefig(os.path.join(self.config['output_path'], 'feature_importance.png'))
                plt.close()
                
                # Сохраняем важность признаков в CSV
                feature_importance.to_csv(
                    os.path.join(self.config['output_path'], 'feature_importance.csv'),
                    index=False
                )
                
                self.logger.info(f"Топ 5 важных признаков:\n{feature_importance.head()}")
                self.logger.info("Анализ важности признаков завершен")
                
            else:
                self.logger.warning("Модель не поддерживает прямой расчет важности признаков")
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе важности признаков: {str(e)}")
            raise
    
    def explain_with_lime(self, instance_idx: int) -> Dict[str, Any]:
        """
        Объясняет конкретный пример с помощью LIME
        
        Args:
            instance_idx: Индекс примера для объяснения
            
        Returns:
            Словарь с объяснением LIME
        """
        self.logger.info(f"Объяснение примера {instance_idx} с помощью LIME...")
        
        # Создаем объяснитель LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.data[self.feature_names].values,
            feature_names=self.feature_names,
            class_names=['Negative', 'Positive'],
            mode='classification'
        )
        
        # Получаем объяснение
        exp = explainer.explain_instance(
            self.data[self.feature_names].iloc[instance_idx].values,
            self.model.predict_proba if hasattr(self.model, 'predict_proba')
            else self.model.predict
        )
        
        # Сохраняем результаты
        self.explanations['lime'][instance_idx] = {
            'local_importance': dict(exp.local_exp[1]),
            'prediction': exp.predict_proba[1] if hasattr(exp, 'predict_proba')
            else exp.predict[0],
            'explanation': exp.as_list()
        }
        
        return self.explanations['lime'][instance_idx]
    
    def explain_with_anchor(self, instance_idx: int) -> Dict[str, Any]:
        """
        Объясняет пример с помощью Anchors
        
        Args:
            instance_idx: Индекс примера для объяснения
            
        Returns:
            Словарь с объяснением Anchor
        """
        self.logger.info(f"Объяснение примера {instance_idx} с помощью Anchor...")
        
        # Создаем объяснитель Anchor
        explainer = AnchorTabular(
            predictor=self.model.predict,
            feature_names=self.feature_names
        )
        
        # Получаем объяснение
        explanation = explainer.explain(
            self.data[self.feature_names].iloc[instance_idx].values
        )
        
        # Сохраняем результаты
        self.explanations['anchor'][instance_idx] = {
            'anchor': explanation.anchor,
            'precision': explanation.precision,
            'coverage': explanation.coverage
        }
        
        return self.explanations['anchor'][instance_idx]
    
    def explain_with_counterfactual(self, instance_idx: int) -> Dict[str, Any]:
        """
        Генерирует контрфактуальные объяснения
        
        Args:
            instance_idx: Индекс примера для объяснения
            
        Returns:
            Словарь с контрфактуальными объяснениями
        """
        self.logger.info(f"Генерация контрфактуальных объяснений для примера {instance_idx}...")
        
        # Создаем объяснитель
        explainer = CounterfactualProto(
            predictor=self.model.predict,
            shape=(1, len(self.feature_names)),
            feature_names=self.feature_names
        )
        
        # Получаем объяснение
        explanation = explainer.explain(
            self.data[self.feature_names].iloc[instance_idx].values.reshape(1, -1)
        )
        
        # Сохраняем результаты
        self.explanations['counterfactual'][instance_idx] = {
            'counterfactual': explanation.cf,
            'distance': explanation.distance,
            'prediction': explanation.cf_pred
        }
        
        return self.explanations['counterfactual'][instance_idx]
    
    def explain_with_permutation_importance(self):
        """Вычисление permutation importance"""
        try:
            self.logger.info("Вычисление permutation importance...")
            
            # Проверяем наличие признаков
            if not hasattr(self, 'feature_names') or len(self.feature_names) == 0:
                self.feature_names = list(self.X.columns)
            
            # Вычисляем permutation importance
            result = permutation_importance(
                self.model,
                self.X,
                self.y,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Создаем DataFrame с результатами
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            })
            
            # Сортируем по важности
            importance_df = importance_df.sort_values('importance_mean', ascending=False)
            
            # Сохраняем результаты
            output_path = os.path.join(self.config['output_path'], 'permutation_importance.csv')
            importance_df.to_csv(output_path, index=False)
            self.logger.info(f"Результаты permutation importance сохранены в {output_path}")
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении permutation importance: {str(e)}")
            raise
    
    def create_pdp_plots(self) -> None:
        """
        Создает графики частичных зависимостей
        """
        self.logger.info("Создание PDP plots...")
        
        # Выбираем топ признаки по важности
        if 'shap' in self.explanations:
            importance = self.explanations['shap']['importance']
            top_features = [self.feature_names[i] 
                          for i in np.argsort(importance)[-self.config['n_features']:]]
        else:
            top_features = self.feature_names[:self.config['n_features']]
        
        # Создаем графики
        for feature in top_features:
            fig = go.Figure()
            
            # Вычисляем PDP
            pdp_result = partial_dependence(
                self.model,
                self.data[self.feature_names],
                [feature]
            )
            
            # Добавляем линию
            fig.add_trace(
                go.Scatter(
                    x=pdp_result[1][0],
                    y=pdp_result[0][0],
                    mode='lines',
                    name=feature
                )
            )
            
            fig.update_layout(
                title=f'Partial Dependence Plot for {feature}',
                xaxis_title=feature,
                yaxis_title='Partial dependence'
            )
            
            self.visualizations[f'pdp_{feature}'] = fig
    
    def create_shap_plots(self) -> None:
        """
        Создает визуализации SHAP значений
        """
        self.logger.info("Создание SHAP визуализаций...")
        
        if 'shap' not in self.explanations:
            self.logger.warning("SHAP значения не найдены")
            return
        
        # Summary plot
        fig = go.Figure()
        for i, feature in enumerate(self.feature_names):
            fig.add_trace(
                go.Box(
                    y=self.explanations['shap']['values'][:, i],
                    name=feature,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                )
            )
        
        fig.update_layout(
            title='SHAP Summary Plot',
            yaxis_title='SHAP value',
            showlegend=False
        )
        
        self.visualizations['shap_summary'] = fig
        
        # Dependence plots для топ признаков
        importance = np.abs(self.explanations['shap']['values']).mean(axis=0)
        top_features = np.argsort(importance)[-self.config['n_features']:]
        
        for idx in top_features:
            feature = self.feature_names[idx]
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=self.data[feature],
                    y=self.explanations['shap']['values'][:, idx],
                    mode='markers',
                    name=feature
                )
            )
            
            fig.update_layout(
                title=f'SHAP Dependence Plot for {feature}',
                xaxis_title=feature,
                yaxis_title='SHAP value'
            )
            
            self.visualizations[f'shap_dependence_{feature}'] = fig
    
    def analyze_explanation_stability(self) -> Dict[str, Any]:
        """
        Анализирует стабильность объяснений
        
        Returns:
            Словарь с метриками стабильности
        """
        self.logger.info("Анализ стабильности объяснений...")
        
        stability_metrics = {}
        
        # Bootstrap для SHAP values
        if 'shap' in self.explanations:
            n_bootstrap = 100
            bootstrap_values = []
            
            for _ in range(n_bootstrap):
                idx = np.random.choice(
                    len(self.data),
                    size=min(len(self.data), self.config['n_samples']),
                    replace=True
                )
                shap_result = self.explain_with_shap()
                bootstrap_values.append(np.abs(shap_result['values']).mean(axis=0))
            
            stability_metrics['shap'] = {
                'feature_importance_std': np.std(bootstrap_values, axis=0),
                'feature_importance_ci': np.percentile(bootstrap_values, [2.5, 97.5], axis=0)
            }
        
        return stability_metrics
    
    def compare_explanation_methods(self) -> Dict[str, Any]:
        """
        Сравнивает различные методы интерпретации
        
        Returns:
            Словарь с результатами сравнения
        """
        self.logger.info("Сравнение методов интерпретации...")
        
        comparison = {}
        
        # Выбираем тестовый набор
        test_indices = np.random.choice(
            len(self.data),
            size=min(100, len(self.data)),
            replace=False
        )
        
        # Получаем объяснения разными методами
        for idx in test_indices:
            self.explain_with_lime(idx)
            self.explain_with_anchor(idx)
        
        # Сравниваем важность признаков
        if all(method in self.explanations for method in ['shap', 'permutation']):
            comparison['feature_importance_correlation'] = {
                'shap_permutation': spearmanr(
                    self.explanations['shap']['importance'],
                    self.explanations['permutation']['importances_mean']
                )
            }
        
        return comparison
    
    def save_results(self) -> None:
        """
        Сохраняет результаты интерпретации
        """
        self.logger.info("Сохранение результатов...")
        
        # Сохраняем объяснения
        explanations_path = os.path.join(
            self.config['output_path'],
            'explanations.json'
        )
        
        # Преобразуем numpy типы в обычные Python типы
        explanations_dict = {}
        for method, results in self.explanations.items():
            if isinstance(results, dict):
                explanations_dict[method] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in results.items()
                }
            else:
                explanations_dict[method] = float(results) if isinstance(
                    results, (np.float32, np.float64)) else results
        
        with open(explanations_path, 'w') as f:
            json.dump(explanations_dict, f, indent=4, default=str)
        
        # Сохраняем визуализации
        for name, fig in self.visualizations.items():
            # Сохраняем в HTML для интерактивности
            html_path = os.path.join(self.config['output_path'], f'{name}.html')
            fig.write_html(html_path)
            
            # Сохраняем в PNG для статичного просмотра
            png_path = os.path.join(self.config['output_path'], f'{name}.png')
            fig.write_image(png_path)
        
        self.logger.info(f"Результаты сохранены в {self.config['output_path']}")
    
    def interpret(self) -> None:
        """
        Выполняет интерпретацию модели
        """
        try:
            if self.data is None:
                raise ValueError("Данные не загружены")
                
            self.explain_with_shap()
            self.explain_with_permutation_importance()
            self.generate_report()
            self.logger.info("Интерпретация модели завершена")
        except Exception as e:
            self.logger.error(f"Ошибка при интерпретации модели: {str(e)}")
            raise

    def generate_report(self) -> None:
        """
        Генерирует итоговый отчет в формате Markdown
        """
        try:
            self.logger.info("Генерация итогового отчета...")
            
            report_path = os.path.join(self.config['output_path'], '..', 'docs', 'model_interpretation_report.md')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Загружаем результаты анализа
            feature_importance = pd.read_csv(os.path.join(self.config['output_path'], 'feature_importance.csv'))
            permutation_importance = pd.read_csv(os.path.join(self.config['output_path'], 'permutation_importance.csv'))
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # Заголовок
                f.write("# Отчет по интерпретации модели\n\n")
                f.write(f"Дата создания: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Информация о модели
                f.write("## Информация о модели\n\n")
                if isinstance(self.model, Pipeline):
                    model = self.model.named_steps['classifier']
                else:
                    model = self.model
                f.write(f"Тип модели: {type(model).__name__}\n\n")
                
                # Важность признаков
                f.write("## Важность признаков\n\n")
                f.write("### Top-10 важных признаков (Feature Importance)\n\n")
                f.write("| Признак | Важность |\n")
                f.write("|---------|----------|\n")
                for _, row in feature_importance.head(10).iterrows():
                    f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
                f.write("\n")
                
                # Permutation Importance
                f.write("### Top-10 важных признаков (Permutation Importance)\n\n")
                f.write("| Признак | Важность (среднее) | Стд. отклонение |\n")
                f.write("|---------|-------------------|------------------|\n")
                for _, row in permutation_importance.head(10).iterrows():
                    f.write(f"| {row['feature']} | {row['importance_mean']:.4f} | {row['importance_std']:.4f} |\n")
                f.write("\n")
                
                # Визуализации
                f.write("## Визуализации\n\n")
                f.write("### График важности признаков\n\n")
                f.write("![Feature Importance](../models/feature_importance.png)\n\n")
                f.write("### График Permutation Importance\n\n")
                f.write("![Permutation Importance](../models/permutation_importance.png)\n\n")
                
                # Выводы
                f.write("## Выводы\n\n")
                
                # Общие признаки в топ-5 обоих методов
                top5_feature = set(feature_importance['feature'].head().tolist())
                top5_permutation = set(permutation_importance['feature'].head().tolist())
                common_features = top5_feature.intersection(top5_permutation)
                
                f.write("### Наиболее важные признаки\n\n")
                f.write("Признаки, которые показали высокую важность по обоим методам:\n\n")
                for feature in common_features:
                    f.write(f"- {feature}\n")
                f.write("\n")
                
                # Рекомендации
                f.write("### Рекомендации\n\n")
                f.write("1. Особое внимание следует уделить признакам, которые показали высокую важность по обоим методам\n")
                f.write("2. Рассмотреть возможность создания новых признаков на основе наиболее важных\n")
                f.write("3. Провести дополнительный анализ признаков с низкой важностью\n")
            
            self.logger.info(f"Отчет сохранен в {report_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета: {str(e)}")
            raise

if __name__ == "__main__":
    # Получаем абсолютный путь к текущей директории
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # Конфигурация с правильными путями и именем целевой переменной
    config = {
        'model_path': os.path.join(project_dir, 'models'),
        'data_path': os.path.join(project_dir, 'data', 'processed', 'engineered_features.csv'),
        'output_path': os.path.join(project_dir, 'models'),
        'target_column': 'label',  # Изменено с 'target' на 'label' - убедитесь, что это правильное имя
        'n_features': 10
    }
    
    try:
        # Проверяем существование файла данных
        if not os.path.exists(config['data_path']):
            raise FileNotFoundError(f"Файл данных не найден: {config['data_path']}")
            
        # Проверяем наличие целевой переменной в данных
        df = pd.read_csv(config['data_path'])
        if config['target_column'] not in df.columns:
            available_columns = ', '.join(df.columns)
            raise ValueError(f"Целевая переменная '{config['target_column']}' не найдена в данных. "
                           f"Доступные столбцы: {available_columns}")
        
        # Создаем директории с проверкой прав доступа
        for path in [config['model_path'], config['output_path']]:
            try:
                os.makedirs(path, exist_ok=True)
            except PermissionError:
                # Если нет прав доступа, используем временную директорию
                import tempfile
                temp_dir = tempfile.gettempdir()
                if path == config['model_path']:
                    config['model_path'] = os.path.join(temp_dir, 'models')
                if path == config['output_path']:
                    config['output_path'] = os.path.join(temp_dir, 'output')
                os.makedirs(path, exist_ok=True)
        
        interpreter = ModelInterpreter(config)
        interpreter.interpret()
        
    except Exception as e:
        print(f"Ошибка при выполнении интерпретации: {str(e)}")
        # Выводим доступные столбцы, если есть проблема с целевой переменной
        try:
            df = pd.read_csv(config['data_path'])
            print(f"\nДоступные столбцы в данных:")
            for col in df.columns:
                print(f"- {col}")
        except Exception:
            pass 