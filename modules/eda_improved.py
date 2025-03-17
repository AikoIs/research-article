import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import json
import warnings
import gc

warnings.filterwarnings('ignore')
gc.enable()

# Используем базовый стиль matplotlib вместо seaborn
plt.style.use('default')  # Заменили 'seaborn' на 'default'

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExploratoryAnalysis:
    """
    Класс для проведения разведочного анализа данных
    
    Attributes:
        config (Dict[str, Any]): Конфигурация анализа
        data (pd.DataFrame): Анализируемые данные
        results (Dict[str, Any]): Результаты анализа
        figures (Dict[str, Any]): Созданные визуализации
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация анализатора
        
        Args:
            config: Словарь с настройками:
                - input_path: путь к входному файлу
                - output_path: путь для сохранения результатов
                - n_clusters: число кластеров для кластерного анализа
                - n_components: число компонент для PCA
                - significance_level: уровень значимости для тестов
        """
        self.config = config
        self.data = None
        self.results = {
            'summary': {},
            'statistical_tests': {},
            'relationships': {},
            'clusters': {}
        }
        self.figures = {}
        
        # Создаем директории
        os.makedirs(os.path.dirname(self.config['output_path']), exist_ok=True)
        
        # Настройка логирования
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_data(self) -> None:
        """
        Загружает данные из файла
        """
        try:
            self.logger.info(f"Загрузка данных из {self.config['input_path']}")
            self.data = pd.read_csv(self.config['input_path'])
            
            # Преобразуем временные колонки
            time_cols = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME']
            for col in time_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_datetime(self.data[col])
            
            self.logger.info(f"Загружено {len(self.data)} записей")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    def analyze_missing_values(self) -> Dict:
        """
        Анализирует пропущенные значения с оптимизацией памяти
        """
        try:
            self.logger.info("Анализ пропущенных значений...")
            
            # Вычисляем статистику пропущенных значений по чанкам
            chunk_size = 1000000
            missing_stats = {}
            total_rows = len(self.data)
            
            # Инициализируем счетчики для каждой колонки
            for col in self.data.columns:
                missing_stats[col] = {
                    'missing_count': 0,
                    'missing_percentage': 0
                }
            
            # Обрабатываем данные чанками
            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                chunk = self.data.iloc[start:end]
                
                # Подсчитываем пропущенные значения для каждой колонки
                for col in chunk.columns:
                    missing_count = chunk[col].isna().sum()
                    missing_stats[col]['missing_count'] += missing_count
                
                # Очищаем память
                del chunk
                gc.collect()
                
                if start % (chunk_size * 10) == 0:
                    self.logger.info(f"Обработано {end}/{total_rows} строк")
            
            # Вычисляем финальные проценты
            for col in missing_stats:
                missing_stats[col]['missing_percentage'] = (
                    missing_stats[col]['missing_count'] / total_rows * 100
                )
            
            # Создаем график с помощью matplotlib вместо plotly
            plt.figure(figsize=(12, 6))
            plt.bar(
                [col[0] for col in sorted(missing_stats.items(), key=lambda x: x[1]['missing_percentage'], reverse=True)[:20]],
                [col[1]['missing_percentage'] for col in sorted(missing_stats.items(), key=lambda x: x[1]['missing_percentage'], reverse=True)[:20]]
            )
            plt.title('Топ-20 признаков по количеству пропущенных значений')
            plt.xlabel('Признаки')
            plt.ylabel('Процент пропущенных значений (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()  # Автоматически регулирует размеры
            
            # Сохраняем график в PNG
            output_dir = os.path.join('..', 'data', 'eda_results')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'missing_values.png'))
            plt.close()
            
            self.logger.info("Анализ пропущенных значений завершен")
            return missing_stats
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе пропущенных значений: {str(e)}")
            raise
    
    def analyze_distributions(self) -> Dict:
        """
        Анализирует распределения числовых признаков с оптимизацией памяти
        """
        try:
            self.logger.info("Анализ распределений...")
            
            # Получаем числовые колонки
            numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
            
            # Размер чанка
            chunk_size = 100000  # Уменьшили размер чанка
            
            distribution_stats = {}
            for col in numeric_cols:
                self.logger.info(f"Анализ распределения признака: {col}")
                
                try:
                    # Вычисляем базовые статистики по чанкам
                    sum_values = 0
                    sum_squares = 0
                    count = 0
                    min_val = float('inf')
                    max_val = float('-inf')
                    values = []  # Временное хранение значений для гистограммы
                    
                    # Обрабатываем данные чанками
                    for start in range(0, len(self.data), chunk_size):
                        end = min(start + chunk_size, len(self.data))
                        chunk = self.data[col].iloc[start:end]
                        
                        # Собираем выборку для гистограммы
                        if len(values) < 10000:  # Ограничиваем размер выборки
                            sample_size = min(1000, len(chunk))
                            values.extend(chunk.sample(n=sample_size).tolist())
                        
                        # Обновляем статистики
                        sum_values += chunk.sum()
                        sum_squares += (chunk ** 2).sum()
                        count += len(chunk)
                        min_val = min(min_val, chunk.min())
                        max_val = max(max_val, chunk.max())
                        
                        # Очищаем память
                        del chunk
                        gc.collect()
                    
                    # Вычисляем финальные статистики
                    mean = sum_values / count
                    variance = (sum_squares / count) - (mean ** 2)
                    std = np.sqrt(variance)
                    
                    distribution_stats[col] = {
                        'mean': mean,
                        'std': std,
                        'min': min_val,
                        'max': max_val
                    }
                    
                    # Создаем гистограмму на основе выборки
                    plt.figure(figsize=(10, 6))
                    plt.hist(values, bins=50, density=True, alpha=0.7)
                    plt.title(f'Распределение {col} (на основе выборки)')
                    plt.xlabel(col)
                    plt.ylabel('Плотность')
                    
                    # Добавляем среднее значение и стандартное отклонение
                    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Среднее = {mean:.2f}')
                    plt.axvline(mean + std, color='g', linestyle='dashed', linewidth=1, label=f'Ст. откл. = {std:.2f}')
                    plt.axvline(mean - std, color='g', linestyle='dashed', linewidth=1)
                    plt.legend()
                    
                    # Сохраняем график
                    output_dir = os.path.join('..', 'data', 'eda_results')
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))
                    plt.close()
                    
                    # Очищаем память
                    del values
                    gc.collect()
                    
                except Exception as e:
                    self.logger.error(f"Ошибка при анализе распределения {col}: {str(e)}")
                    continue
            
            self.logger.info("Анализ распределений завершен")
            return distribution_stats
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе распределений: {str(e)}")
            raise
    
    def analyze_relationships(self) -> Dict[str, Any]:
        """
        Анализирует взаимосвязи между переменными
        
        Returns:
            Словарь с результатами анализа
        """
        relationship_analysis = {}
        
        # Корреляционный анализ для числовых признаков
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Корреляция Пирсона
            pearson_corr = self.data[numeric_cols].corr(method='pearson')
            # Корреляция Спирмена
            spearman_corr = self.data[numeric_cols].corr(method='spearman')
            
            relationship_analysis['correlations'] = {
                'pearson': pearson_corr.to_dict(),
                'spearman': spearman_corr.to_dict()
            }
            
            # Визуализация корреляций
            fig = make_subplots(rows=1, cols=2,
                              subplot_titles=('Корреляция Пирсона', 
                                            'Корреляция Спирмена'))
            
            fig.add_trace(
                go.Heatmap(z=pearson_corr.values,
                          x=pearson_corr.columns,
                          y=pearson_corr.columns,
                          colorscale='RdBu'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Heatmap(z=spearman_corr.values,
                          x=spearman_corr.columns,
                          y=spearman_corr.columns,
                          colorscale='RdBu'),
                row=1, col=2
            )
            
            fig.update_layout(title='Корреляционные матрицы')
            self.figures['correlation_matrices'] = fig
        
        # Анализ для целевых переменных
        target_cols = ['HOSPITAL_EXPIRE_FLAG', 'was_in_icu']
        for target in target_cols:
            if target in self.data.columns:
                target_analysis = {}
                
                # Для числовых признаков
                for col in numeric_cols:
                    if col != target:
                        # t-test
                        t_stat, p_value = stats.ttest_ind(
                            self.data[self.data[target] == 1][col].dropna(),
                            self.data[self.data[target] == 0][col].dropna(),
                            equal_var=False
                        )
                        
                        # Эффект размера (Cohen's d)
                        d = (self.data[self.data[target] == 1][col].mean() - 
                             self.data[self.data[target] == 0][col].mean()) / \
                            np.sqrt((self.data[self.data[target] == 1][col].var() + 
                                    self.data[self.data[target] == 0][col].var()) / 2)
                        
                        target_analysis[col] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'effect_size': d
                        }
                        
                        # Визуализация
                        fig = go.Figure()
                        fig.add_trace(
                            go.Box(y=self.data[self.data[target] == 0][col],
                                  name='0',
                                  boxpoints='outliers')
                        )
                        fig.add_trace(
                            go.Box(y=self.data[self.data[target] == 1][col],
                                  name='1',
                                  boxpoints='outliers')
                        )
                        fig.update_layout(
                            title=f'{col} by {target}',
                            yaxis_title=col,
                            xaxis_title=target
                        )
                        self.figures[f'{col}_by_{target}'] = fig
                
                # Для категориальных признаков
                cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
                for col in cat_cols:
                    if col != target:
                        # Chi-square test
                        contingency = pd.crosstab(self.data[col], self.data[target])
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                        
                        # Cramer's V
                        n = contingency.sum().sum()
                        min_dim = min(contingency.shape) - 1
                        cramer_v = np.sqrt(chi2 / (n * min_dim))
                        
                        target_analysis[col] = {
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'cramer_v': cramer_v
                        }
                        
                        # Визуализация
                        fig = px.bar(contingency, 
                                   title=f'{col} by {target}',
                                   barmode='group')
                        self.figures[f'{col}_by_{target}'] = fig
                
                relationship_analysis[f'{target}_relationships'] = target_analysis
        
        return relationship_analysis
    
    def perform_dimension_reduction(self) -> Dict[str, Any]:
        """
        Выполняет снижение размерности
        
        Returns:
            Словарь с результатами анализа
        """
        dimension_analysis = {}
        
        # Отбираем числовые признаки
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            # Стандартизация данных
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.data[numeric_cols])
            
            # PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            # Сохраняем результаты
            dimension_analysis['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components_95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
            }
            
            # Визуализация
            fig = make_subplots(rows=1, cols=2,
                              subplot_titles=('Объясненная дисперсия', 
                                            'Проекция на первые 2 компоненты'))
            
            # График объясненной дисперсии
            fig.add_trace(
                go.Bar(x=np.arange(1, len(pca.explained_variance_ratio_) + 1),
                      y=pca.explained_variance_ratio_,
                      name='Individual'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=np.arange(1, len(pca.explained_variance_ratio_) + 1),
                          y=np.cumsum(pca.explained_variance_ratio_),
                          name='Cumulative',
                          mode='lines+markers'),
                row=1, col=1
            )
            
            # Проекция на первые 2 компоненты
            fig.add_trace(
                go.Scatter(x=pca_result[:, 0],
                          y=pca_result[:, 1],
                          mode='markers',
                          name='Samples'),
                row=1, col=2
            )
            
            fig.update_layout(title='Анализ главных компонент')
            self.figures['pca_analysis'] = fig
        
        return dimension_analysis
    
    def perform_clustering(self) -> Dict[str, Any]:
        """
        Выполняет кластерный анализ
        
        Returns:
            Словарь с результатами анализа
        """
        clustering_analysis = {}
        
        # Отбираем числовые признаки
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Стандартизация данных
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.data[numeric_cols])
            
            # K-means
            kmeans = KMeans(n_clusters=self.config['n_clusters'], 
                          random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Сохраняем результаты
            clustering_analysis['kmeans'] = {
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_,
                'cluster_sizes': np.bincount(clusters).tolist()
            }
            
            # Визуализация в пространстве первых двух главных компонент
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            fig = go.Figure()
            for i in range(self.config['n_clusters']):
                mask = clusters == i
                fig.add_trace(
                    go.Scatter(x=pca_result[mask, 0],
                              y=pca_result[mask, 1],
                              mode='markers',
                              name=f'Cluster {i}')
                )
            
            # Добавляем центры кластеров
            centers_pca = pca.transform(kmeans.cluster_centers_)
            fig.add_trace(
                go.Scatter(x=centers_pca[:, 0],
                          y=centers_pca[:, 1],
                          mode='markers',
                          marker=dict(size=15, symbol='x', color='black'),
                          name='Centroids')
            )
            
            fig.update_layout(title='Кластерный анализ (проекция на 2 главные компоненты)')
            self.figures['clustering'] = fig
            
            # Анализ характеристик кластеров
            cluster_stats = {}
            for col in numeric_cols:
                cluster_stats[col] = {
                    'means': [self.data[clusters == i][col].mean() 
                             for i in range(self.config['n_clusters'])],
                    'stds': [self.data[clusters == i][col].std() 
                            for i in range(self.config['n_clusters'])]
                }
            clustering_analysis['cluster_statistics'] = cluster_stats
        
        return clustering_analysis
    
    def save_results(self) -> None:
        """
        Сохраняет результаты анализа
        """
        try:
            # Сохраняем метрики
            results_path = os.path.join(
                self.config['output_path'],
                'eda_results.json'
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
            
            # Сохраняем визуализации в PNG
            for name, fig in self.figures.items():
                plt.figure(figsize=(12, 8))
                
                if name == 'correlation_matrices':
                    try:
                        # Получаем корреляционные матрицы
                        correlations = self.results.get('relationship_analysis', {}).get('correlations', {})
                        
                        if correlations and isinstance(correlations.get('pearson'), (np.ndarray, list)):
                            # Создаем корреляционные матрицы
                            plt.subplot(1, 2, 1)
                            pearson_corr = np.array(correlations['pearson'])
                            plt.imshow(pearson_corr, cmap='RdBu', vmin=-1, vmax=1)
                            plt.title('Корреляция Пирсона')
                            plt.colorbar()
                            
                            plt.subplot(1, 2, 2)
                            spearman_corr = np.array(correlations['spearman'])
                            plt.imshow(spearman_corr, cmap='RdBu', vmin=-1, vmax=1)
                            plt.title('Корреляция Спирмена')
                            plt.colorbar()
                        else:
                            self.logger.warning("Корреляционные матрицы отсутствуют или имеют неверный формат")
                            continue
                            
                    except Exception as e:
                        self.logger.error(f"Ошибка при создании корреляционных матриц: {str(e)}")
                        continue
                    
                elif name == 'pca_analysis':
                    try:
                        pca_results = self.results.get('dimension_analysis', {}).get('pca', {})
                        
                        if pca_results:
                            # График PCA
                            plt.subplot(1, 2, 1)
                            variance_ratio = np.array(pca_results['explained_variance_ratio'])
                            plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)
                            plt.title('Объясненная дисперсия')
                            plt.xlabel('Компонента')
                            plt.ylabel('Доля объясненной дисперсии')
                            
                            plt.subplot(1, 2, 2)
                            cumulative_ratio = np.array(pca_results['cumulative_variance_ratio'])
                            plt.plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio)
                            plt.title('Кумулятивная объясненная дисперсия')
                            plt.xlabel('Число компонент')
                            plt.ylabel('Доля объясненной дисперсии')
                        else:
                            self.logger.warning("Результаты PCA отсутствуют")
                            continue
                            
                    except Exception as e:
                        self.logger.error(f"Ошибка при создании графиков PCA: {str(e)}")
                        continue
                    
                elif name == 'clustering':
                    try:
                        clustering_results = self.results.get('clustering_analysis', {}).get('kmeans', {})
                        
                        if clustering_results:
                            cluster_labels = np.array(clustering_results['cluster_labels'])
                            pca_coords = np.array(clustering_results['pca_coords'])
                            
                            for i in range(self.config['n_clusters']):
                                mask = cluster_labels == i
                                plt.scatter(pca_coords[mask, 0],
                                          pca_coords[mask, 1],
                                          label=f'Кластер {i}')
                            plt.title('Кластерный анализ (проекция на 2 главные компоненты)')
                            plt.xlabel('PC1')
                            plt.ylabel('PC2')
                            plt.legend()
                        else:
                            self.logger.warning("Результаты кластеризации отсутствуют")
                            continue
                            
                    except Exception as e:
                        self.logger.error(f"Ошибка при создании графика кластеризации: {str(e)}")
                        continue
                
                plt.tight_layout()
                
                # Сохраняем график
                output_path = os.path.join(self.config['output_path'], f'{name}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Результаты сохранены в {self.config['output_path']}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            raise
    
    def analyze(self) -> None:
        """
        Проводит полный анализ данных
        """
        try:
            # Загружаем данные
            self.load_data()
            
            # Анализ пропущенных значений
            self.logger.info("Анализ пропущенных значений...")
            self.results['missing_analysis'] = self.analyze_missing_values()
            
            # Анализ распределений
            self.logger.info("Анализ распределений...")
            self.results['distribution_analysis'] = self.analyze_distributions()
            
            # Анализ взаимосвязей
            self.logger.info("Анализ взаимосвязей...")
            self.results['relationship_analysis'] = self.analyze_relationships()
            
            # Снижение размерности
            self.logger.info("Снижение размерности...")
            self.results['dimension_analysis'] = self.perform_dimension_reduction()
            
            # Кластерный анализ
            self.logger.info("Кластерный анализ...")
            self.results['clustering_analysis'] = self.perform_clustering()
            
            # Сохраняем результаты
            self.logger.info("Сохранение результатов...")
            self.save_results()
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе данных: {str(e)}")
            raise

if __name__ == "__main__":
    # Пример конфигурации
    config = {
        'input_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "..", "data", "processed", "engineered_features.csv"),
        'output_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "..", "data", "eda_results"),
        'n_clusters': 3,
        'n_components': 2,
        'significance_level': 0.05
    }
    
    # Создаем и запускаем анализатор
    analyzer = ExploratoryAnalysis(config)
    analyzer.analyze() 