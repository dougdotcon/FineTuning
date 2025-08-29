# FT-ECO-001: Fine-Tuning para IA em Ecologia Computacional

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados em ecologia computacional, integrando princípios ecológicos com métodos computacionais avançados para modelar, analisar e prever sistemas ambientais complexos.

### Contexto Filosófico
A ecologia computacional representa a síntese entre a compreensão holística dos ecossistemas e o rigor analítico da computação. O objetivo é transcender a mera descrição de padrões ecológicos para alcançar a predição e gestão sustentável dos sistemas naturais.

### Metodologia de Aprendizado Recomendada
1. **Abordagem Sistêmica**: Integrar múltiplas escalas espaciais e temporais
2. **Modelagem Interdisciplinar**: Conectar ecologia com matemática, estatística e computação
3. **Validação Empírica**: Basear modelos em dados observacionais robustos
4. **Pensamento Adaptativo**: Incorporar mudança e incerteza nos modelos
5. **Aplicação Prática**: Focar em soluções para problemas ambientais reais

---

## 1. FUNDAMENTOS TEÓRICOS ECOLÓGICOS

### 1.1 Dinâmica de Populações
```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lotka_volterra_model(X, t, alpha, beta, gamma, delta):
    """
    Modelo Lotka-Volterra para interação presa-predador
    X = [presa, predador]
    dX/dt = [alpha*X[0] - beta*X[0]*X[1], delta*X[0]*X[1] - gamma*X[1]]
    """
    prey, predator = X
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

def simulate_ecosystem(initial_conditions, time_span, params):
    """
    Simula dinâmica de ecossistema usando modelo Lotka-Volterra
    """
    t = np.linspace(0, time_span, 1000)
    solution = odeint(lotka_volterra_model, initial_conditions, t, args=params)

    plt.figure(figsize=(10, 6))
    plt.plot(t, solution[:, 0], 'b-', label='Presas')
    plt.plot(t, solution[:, 1], 'r-', label='Predadores')
    plt.xlabel('Tempo')
    plt.ylabel('População')
    plt.title('Dinâmica Presa-Predador (Lotka-Volterra)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return solution
```

**Conceitos Críticos:**
- Equações diferenciais para crescimento populacional
- Capacidade de suporte e fatores limitantes
- Interações interespecíficas (competição, predação, mutualismo)
- Efeitos densidade-dependentes e independentes

### 1.2 Estrutura e Funcionamento de Ecossistemas
```python
import networkx as nx
import matplotlib.pyplot as plt

class EcosystemNetwork:
    """
    Representação de rede trófica usando grafos
    """

    def __init__(self):
        self.network = nx.DiGraph()
        self.trophic_levels = {}

    def add_species(self, species_name, trophic_level):
        """Adiciona espécie à rede trófica"""
        self.network.add_node(species_name, trophic_level=trophic_level)
        self.trophic_levels[species_name] = trophic_level

    def add_interaction(self, predator, prey, strength=1.0):
        """Adiciona interação trófica"""
        self.network.add_edge(predator, prey, weight=strength)

    def calculate_trophic_metrics(self):
        """Calcula métricas da rede trófica"""
        # Comprimento médio da cadeia alimentar
        chain_lengths = []
        for node in self.network.nodes():
            if self.network.in_degree(node) == 0:  # Produtores
                for target in nx.descendants(self.network, node):
                    path_length = nx.shortest_path_length(
                        self.network, node, target
                    )
                    chain_lengths.append(path_length)

        avg_chain_length = np.mean(chain_lengths) if chain_lengths else 0

        # Conectância
        possible_links = len(self.network.nodes()) ** 2
        actual_links = len(self.network.edges())
        connectance = actual_links / possible_links

        return {
            'average_chain_length': avg_chain_length,
            'connectance': connectance,
            'species_richness': len(self.network.nodes())
        }

    def visualize_network(self):
        """Visualiza a rede trófica"""
        pos = nx.spring_layout(self.network)

        # Colorir nós por nível trófico
        colors = [self.trophic_levels[node] for node in self.network.nodes()]

        plt.figure(figsize=(12, 8))
        nx.draw(self.network, pos, with_labels=True,
                node_color=colors, cmap=plt.cm.viridis,
                node_size=500, font_size=10,
                arrows=True, arrowsize=20)
        plt.title("Rede Trófica")
        plt.show()
```

**Tópicos Essenciais:**
- Fluxos de energia e matéria através de teias alimentares
- Ciclos biogeoquímicos (carbono, nitrogênio, fósforo)
- Estrutura da biomassa e produtividade primária
- Decomposição e ciclagem de nutrientes

### 1.3 Ecologia de Paisagens
```python
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

class LandscapeEcologist:
    """
    Análise de padrões espaciais em ecologia de paisagens
    """

    def __init__(self, landscape_data):
        self.landscape = landscape_data
        self.patches = []

    def identify_patches(self, habitat_type):
        """Identifica manchas de habitat"""
        # Algoritmo simplificado de identificação de manchas
        patches = []
        visited = set()

        for i, cell in enumerate(self.landscape):
            if cell == habitat_type and i not in visited:
                patch = self._grow_patch(i, habitat_type, visited)
                if len(patch) > 0:
                    patches.append(patch)

        self.patches = patches
        return patches

    def _grow_patch(self, start_idx, habitat_type, visited):
        """Cresce uma mancha a partir de um ponto inicial"""
        patch = []
        queue = [start_idx]
        rows, cols = self.landscape.shape

        while queue:
            idx = queue.pop(0)
            if idx in visited:
                continue

            row, col = idx // cols, idx % cols
            if (0 <= row < rows and 0 <= col < cols and
                self.landscape[row, col] == habitat_type):
                patch.append((row, col))
                visited.add(idx)

                # Adicionar vizinhos
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < rows and 0 <= new_col < cols):
                        new_idx = new_row * cols + new_col
                        if new_idx not in visited:
                            queue.append(new_idx)

        return patch

    def calculate_patch_metrics(self):
        """Calcula métricas de manchas"""
        metrics = []

        for patch in self.patches:
            # Área
            area = len(patch)

            # Perímetro
            perimeter = self._calculate_perimeter(patch)

            # Forma (índice de forma)
            shape_index = perimeter / (2 * np.sqrt(np.pi * area))

            # Coordenadas do centróide
            rows = [p[0] for p in patch]
            cols = [p[1] for p in patch]
            centroid = (np.mean(rows), np.mean(cols))

            metrics.append({
                'area': area,
                'perimeter': perimeter,
                'shape_index': shape_index,
                'centroid': centroid
            })

        return metrics

    def _calculate_perimeter(self, patch):
        """Calcula perímetro de uma mancha"""
        perimeter = 0
        for row, col in patch:
            # Verificar se cada lado é fronteira
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if ((new_row < 0 or new_row >= self.landscape.shape[0]) or
                    (new_col < 0 or new_col >= self.landscape.shape[1]) or
                    self.landscape[new_row, new_col] != self.landscape[row, col]):
                    perimeter += 1

        return perimeter
```

**Conceitos Fundamentais:**
- Heterogeneidade espacial e padrões de distribuição
- Conectividade e fragmentação de habitats
- Fluxos entre manchas e corredores ecológicos
- Escala e resolução em análises espaciais

---

## 2. MÉTODOS COMPUTACIONAIS

### 2.1 Modelagem Baseada em Agentes
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EcosystemAgent:
    """
    Agente individual em modelo de ecossistema baseado em agentes
    """

    def __init__(self, species_type, position, energy=100):
        self.species_type = species_type  # 'producer', 'consumer', 'decomposer'
        self.position = np.array(position)
        self.energy = energy
        self.age = 0
        self.alive = True

    def move(self, environment):
        """Movimento do agente no ambiente"""
        # Movimento browniano simples
        direction = np.random.normal(0, 1, 2)
        direction = direction / np.linalg.norm(direction)

        new_position = self.position + direction * 0.1

        # Manter dentro dos limites
        new_position = np.clip(new_position, 0, 1)

        # Verificar se movimento é válido
        if environment.is_habitable(new_position):
            self.position = new_position

    def interact(self, other_agents, environment):
        """Interação com outros agentes"""
        interactions = []

        for other in other_agents:
            distance = np.linalg.norm(self.position - other.position)

            if distance < 0.05:  # Raio de interação
                interaction = self._calculate_interaction(other, distance)
                interactions.append(interaction)

        return interactions

    def _calculate_interaction(self, other, distance):
        """Calcula efeito da interação"""
        if self.species_type == 'consumer' and other.species_type == 'producer':
            # Herbivoria
            energy_gain = min(other.energy * 0.1, 20)
            self.energy += energy_gain
            other.energy -= energy_gain
            return 'herbivory'

        elif self.species_type == 'consumer' and other.species_type == 'consumer':
            # Competição
            if self.energy > other.energy:
                self.energy += 5
                other.energy -= 5
            return 'competition'

        return 'neutral'

class EcosystemModel:
    """
    Modelo completo de ecossistema baseado em agentes
    """

    def __init__(self, size=100, num_agents=500):
        self.size = size
        self.environment = self._create_environment()
        self.agents = self._initialize_agents(num_agents)

    def _create_environment(self):
        """Cria ambiente heterogêneo"""
        # Gradiente de produtividade
        x = np.linspace(0, 1, self.size)
        y = np.linspace(0, 1, self.size)
        X, Y = np.meshgrid(x, y)

        # Produtividade sinusoidal
        productivity = 0.5 + 0.3 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        return productivity

    def _initialize_agents(self, num_agents):
        """Inicializa população de agentes"""
        agents = []

        for _ in range(num_agents):
            position = np.random.random(2)

            # Distribuição de tipos de espécie
            species_types = ['producer', 'consumer', 'decomposer']
            weights = [0.6, 0.3, 0.1]  # Maioria produtores

            species_type = np.random.choice(species_types, p=weights)

            agent = EcosystemAgent(species_type, position)
            agents.append(agent)

        return agents

    def step(self):
        """Executa um passo temporal do modelo"""
        # Movimento
        for agent in self.agents:
            if agent.alive:
                agent.move(self)
                agent.age += 1

        # Interações
        for i, agent1 in enumerate(self.agents):
            if agent1.alive:
                others = [a for j, a in enumerate(self.agents)
                         if j != i and a.alive]
                interactions = agent1.interact(others, self)

        # Metabolismo e mortalidade
        for agent in self.agents:
            if agent.alive:
                # Consumo basal de energia
                agent.energy -= 1

                # Mortalidade por inanição
                if agent.energy <= 0:
                    agent.alive = False

                # Morte por velhice
                if agent.age > 1000:
                    agent.alive = False

        # Reprodução
        self._reproduce_agents()

        # Atualizar ambiente
        self._update_environment()

    def _reproduce_agents(self):
        """Reprodução de agentes"""
        new_agents = []

        for agent in self.agents:
            if (agent.alive and agent.energy > 150 and
                np.random.random() < 0.1):  # Taxa de reprodução

                # Energia dividida entre pai e filho
                offspring_energy = agent.energy * 0.4
                agent.energy *= 0.6

                # Posição próxima ao pai
                offspring_position = agent.position + np.random.normal(0, 0.05, 2)
                offspring_position = np.clip(offspring_position, 0, 1)

                offspring = EcosystemAgent(
                    agent.species_type,
                    offspring_position,
                    offspring_energy
                )

                new_agents.append(offspring)

        self.agents.extend(new_agents)

    def _update_environment(self):
        """Atualiza estado do ambiente"""
        # Regeneração baseada em produtividade
        # Implementação simplificada
        pass

    def get_statistics(self):
        """Retorna estatísticas do ecossistema"""
        alive_agents = [a for a in self.agents if a.alive]

        if not alive_agents:
            return {'total_agents': 0}

        species_counts = {}
        for agent in alive_agents:
            species_counts[agent.species_type] = species_counts.get(
                agent.species_type, 0) + 1

        avg_energy = np.mean([a.energy for a in alive_agents])
        avg_age = np.mean([a.age for a in alive_agents])

        return {
            'total_agents': len(alive_agents),
            'species_counts': species_counts,
            'average_energy': avg_energy,
            'average_age': avg_age
        }
```

**Princípios de Modelagem Baseada em Agentes:**
- Comportamentos individuais emergem em padrões populacionais
- Heterogeneidade e estocasticidade nos processos
- Interações locais levam a dinâmicas globais
- Adaptação e aprendizagem dos agentes

### 2.2 Análise de Séries Temporais Ecológicas
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

class TimeSeriesEcologist:
    """
    Análise de séries temporais em ecologia
    """

    def __init__(self, data, time_column='date', value_column='abundance'):
        self.data = data.copy()
        self.time_col = time_column
        self.value_col = value_column

        # Preparar dados
        self._prepare_data()

    def _prepare_data(self):
        """Prepara dados para análise"""
        # Converter coluna de tempo
        self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
        self.data = self.data.set_index(self.time_col)

        # Remover valores ausentes
        self.data = self.data.dropna()

        # Resample para frequência regular
        self.data = self.data.resample('M').mean()

    def test_stationarity(self):
        """Testa estacionariedade da série"""
        # Teste Augmented Dickey-Fuller
        adf_result = adfuller(self.data[self.value_col], autolag='AIC')

        # Teste KPSS
        kpss_result = kpss(self.data[self.value_col], regression='c')

        results = {
            'adf': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'stationary': adf_result[1] < 0.05
            },
            'kpss': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'stationary': kpss_result[1] > 0.05
            }
        }

        return results

    def decompose_series(self):
        """Decompõe série em tendência, sazonalidade e resíduo"""
        decomposition = seasonal_decompose(
            self.data[self.value_col],
            model='additive',
            period=12  # Assumindo sazonalidade mensal
        )

        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'original': self.data[self.value_col]
        }

    def calculate_autocorrelation(self, lags=50):
        """Calcula funções de autocorrelação"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # ACF
        plot_acf(self.data[self.value_col], lags=lags, ax=ax1)
        ax1.set_title('Função de Autocorrelação (ACF)')

        # PACF
        plot_pacf(self.data[self.value_col], lags=lags, ax=ax2)
        ax2.set_title('Função de Autocorrelação Parcial (PACF)')

        plt.tight_layout()
        plt.show()

    def detect_anomalies(self, threshold=3):
        """Detecta anomalias na série temporal"""
        # Calcular z-score
        mean = self.data[self.value_col].mean()
        std = self.data[self.value_col].std()

        z_scores = (self.data[self.value_col] - mean) / std

        # Identificar anomalias
        anomalies = self.data[abs(z_scores) > threshold].copy()
        anomalies['z_score'] = z_scores[abs(z_scores) > threshold]

        return anomalies

    def forecast_population(self, steps=24):
        """Previsão de população usando modelo ARIMA simples"""
        from statsmodels.tsa.arima.model import ARIMA

        # Ajustar modelo ARIMA
        model = ARIMA(self.data[self.value_col], order=(1, 1, 1))
        model_fit = model.fit()

        # Fazer previsões
        forecast = model_fit.forecast(steps=steps)

        # Criar índice de datas para previsões
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq='M'
        )

        forecast_series = pd.Series(
            forecast.values,
            index=forecast_dates,
            name='forecast'
        )

        return forecast_series

    def analyze_regime_shifts(self, window_size=12):
        """Detecta mudanças de regime na série"""
        # Calcular médias móveis
        rolling_mean = self.data[self.value_col].rolling(window=window_size).mean()
        rolling_std = self.data[self.value_col].rolling(window=window_size).std()

        # Detectar pontos de inflexão
        diff_mean = rolling_mean.diff()
        regime_shifts = []

        for i in range(window_size, len(diff_mean)):
            if abs(diff_mean.iloc[i]) > 2 * rolling_std.iloc[i]:
                regime_shifts.append({
                    'date': diff_mean.index[i],
                    'change': diff_mean.iloc[i],
                    'significance': abs(diff_mean.iloc[i]) / rolling_std.iloc[i]
                })

        return regime_shifts
```

**Técnicas de Análise de Séries Temporais:**
- Testes de estacionariedade (ADF, KPSS)
- Decomposição sazonal
- Funções de autocorrelação
- Detecção de anomalias e mudanças de regime
- Modelos de previsão (ARIMA, SARIMA)

### 2.3 Modelagem Estatística Ecológica
```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import matplotlib.pyplot as plt

class EcologicalStatistics:
    """
    Métodos estatísticos para análise ecológica
    """

    def __init__(self, data):
        self.data = data

    def species_accumulation_curve(self, species_column='species', sample_column='sample'):
        """Calcula curva de acumulação de espécies"""
        samples = self.data[sample_column].unique()
        accumulation = []

        for i in range(1, len(samples) + 1):
            subset_samples = np.random.choice(samples, i, replace=False)
            subset_data = self.data[self.data[sample_column].isin(subset_samples)]
            unique_species = subset_data[species_column].unique()
            accumulation.append(len(unique_species))

        return np.array(accumulation)

    def diversity_indices(self, abundance_column='abundance', group_column='site'):
        """Calcula índices de diversidade"""
        results = []

        for site in self.data[group_column].unique():
            site_data = self.data[self.data[group_column] == site]

            # Riqueza de espécies
            richness = len(site_data)

            # Índice de Shannon
            proportions = site_data[abundance_column] / site_data[abundance_column].sum()
            shannon = -np.sum(proportions * np.log(proportions))

            # Índice de Simpson
            simpson = 1 - np.sum(proportions ** 2)

            # Índice de Pielou (equitabilidade)
            pielou = shannon / np.log(richness) if richness > 1 else 0

            results.append({
                'site': site,
                'richness': richness,
                'shannon': shannon,
                'simpson': simpson,
                'pielou': pielou
            })

        return pd.DataFrame(results)

    def species_area_relationship(self, area_column='area', species_column='species'):
        """Analisa relação espécies-área"""
        # Logaritmizar dados
        log_area = np.log(self.data[area_column])
        log_species = np.log(self.data[species_column])

        # Regressão linear
        X = sm.add_constant(log_area)
        model = sm.OLS(log_species, X).fit()

        # Resultados
        slope = model.params[1]
        intercept = model.params[0]
        r_squared = model.rsquared

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'model': model
        }

    def mantel_test(self, matrix1, matrix2, permutations=999):
        """Teste de Mantel para correlação entre matrizes"""
        # Implementação simplificada do teste de Mantel
        def mantel_statistic(mat1, mat2):
            # Converter para vetores (triângulo superior)
            vec1 = mat1[np.triu_indices_from(mat1, k=1)]
            vec2 = mat2[np.triu_indices_from(mat2, k=1)]

            # Correlação de Pearson
            return stats.pearsonr(vec1, vec2)[0]

        # Estatística observada
        observed_r = mantel_statistic(matrix1, matrix2)

        # Distribuição nula
        null_distribution = []
        for _ in range(permutations):
            # Permutar linhas/colunas de matrix2
            perm_indices = np.random.permutation(matrix2.shape[0])
            perm_matrix2 = matrix2[perm_indices][:, perm_indices]
            null_r = mantel_statistic(matrix1, perm_matrix2)
            null_distribution.append(null_r)

        # Calcular p-valor
        p_value = (np.sum(np.abs(null_distribution) >= np.abs(observed_r)) + 1) / (permutations + 1)

        return {
            'observed_r': observed_r,
            'p_value': p_value,
            'null_distribution': null_distribution
        }

    def canonical_correspondence_analysis(self, species_matrix, environmental_matrix):
        """Análise de Correspondência Canônica (CCA)"""
        from sklearn.decomposition import PCA
        from sklearn.cross_decomposition import CCA

        # Padronizar dados
        species_std = (species_matrix - species_matrix.mean()) / species_matrix.std()
        env_std = (environmental_matrix - environmental_matrix.mean()) / environmental_matrix.std()

        # CCA
        cca = CCA(n_components=2)
        species_scores, env_scores = cca.fit_transform(species_std, env_std)

        # Calcular eigenvalues
        X_scores = cca.x_scores_
        Y_scores = cca.y_scores_

        return {
            'species_scores': species_scores,
            'environmental_scores': env_scores,
            'canonical_correlations': cca.score(species_std, env_std)
        }
```

**Métodos Estatísticos Ecológicos:**
- Índices de diversidade (Shannon, Simpson, Pielou)
- Curvas de acumulação de espécies
- Relação espécies-área
- Teste de Mantel para autocorrelação espacial
- Análise de correspondência canônica

---

## 3. FERRAMENTAS E BIBLIOTECAS

### 3.1 Ecologia Computacional em Python
```python
# Ambiente recomendado para ecologia computacional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Análise estatística
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Modelagem
from scipy.integrate import odeint
from scipy.optimize import minimize

# Aprendizado de máquina
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Análise espacial
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium

# Redes e grafos
import networkx as nx

# Séries temporais
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Visualização avançada
import plotly.express as px
import plotly.graph_objects as go

print("Bibliotecas essenciais para ecologia computacional carregadas com sucesso!")
```

### 3.2 Softwares Especializados
- **R**: Pacotes vegan, BiodiversityR, ade4 para análise multivariada
- **QGIS**: Análise espacial e SIG para ecologia de paisagens
- **NetLogo**: Modelagem baseada em agentes
- **MCMCglmm**: Modelos mistos generalizados
- **JAGS/Stan**: Inferência bayesiana

### 3.3 Recursos Computacionais
- **High-performance computing** para modelos complexos
- **Google Earth Engine** para análise de dados geoespaciais
- **Cloud computing** (AWS, Google Cloud) para processamento de grandes volumes
- **Databases especializados** para dados ecológicos (PostgreSQL com PostGIS)

---

## 4. HIPÓTESES E APLICAÇÕES PRÁTICAS

### 4.1 Hipóteses Centrais
1. **Teoria da Complexidade**: Sistemas ecológicos exibem comportamento complexo emergente
2. **Princípio da Hierarquia**: Processos ocorrem em múltiplas escalas organizacionais
3. **Conectividade Ecológica**: Interações espaciais determinam dinâmica de metapopulações
4. **Resiliência Adaptativa**: Ecossistemas se adaptam a perturbações mantendo funções essenciais

### 4.2 Aplicações em Conservação
```python
class ConservationPlanner:
    """
    Planejamento de conservação baseado em modelagem computacional
    """

    def __init__(self, ecosystem_data):
        self.ecosystem = ecosystem_data
        self.threats = []
        self.protected_areas = []

    def assess_biodiversity_hotspots(self):
        """Identifica hotspots de biodiversidade"""
        # Análise de endemismo e riqueza de espécies
        species_richness = self.ecosystem.groupby('grid_cell')['species'].nunique()
        endemism = self._calculate_endemism()

        # Combinar métricas
        biodiversity_index = species_richness * endemism

        # Identificar hotspots (top 10%)
        threshold = np.percentile(biodiversity_index, 90)
        hotspots = biodiversity_index[biodiversity_index >= threshold]

        return hotspots

    def _calculate_endemism(self):
        """Calcula índice de endemismo"""
        species_distribution = {}

        for _, row in self.ecosystem.iterrows():
            species = row['species']
            grid_cell = row['grid_cell']

            if species not in species_distribution:
                species_distribution[species] = set()
            species_distribution[species].add(grid_cell)

        # Endemismo por célula
        endemism_scores = {}
        total_species = len(species_distribution)

        for cell in self.ecosystem['grid_cell'].unique():
            endemic_species = 0

            for species, cells in species_distribution.items():
                if len(cells) == 1 and cell in cells:
                    endemic_species += 1

            endemism_scores[cell] = endemic_species / total_species if total_species > 0 else 0

        return pd.Series(endemism_scores)

    def design_protected_area_network(self, target_coverage=0.3):
        """Projeta rede de áreas protegidas"""
        hotspots = self.assess_biodiversity_hotspots()

        # Algoritmo guloso para seleção de áreas
        selected_cells = []
        covered_species = set()

        while len(selected_cells) / len(self.ecosystem['grid_cell'].unique()) < target_coverage:
            best_cell = None
            best_gain = 0

            for cell in hotspots.index:
                if cell not in selected_cells:
                    cell_species = set(self.ecosystem[
                        self.ecosystem['grid_cell'] == cell
                    ]['species'].unique())

                    new_species = cell_species - covered_species
                    if len(new_species) > best_gain:
                        best_gain = len(new_species)
                        best_cell = cell

            if best_cell:
                selected_cells.append(best_cell)
                cell_species = set(self.ecosystem[
                    self.ecosystem['grid_cell'] == best_cell
                ]['species'].unique())
                covered_species.update(cell_species)
            else:
                break

        return selected_cells

    def predict_impacts(self, scenario):
        """Prevê impactos de cenários de mudança"""
        # Modelos de impacto climático
        # Implementação simplificada

        if scenario == 'climate_change':
            # Mudanças na distribuição de espécies
            range_shifts = self._calculate_range_shifts()

            # Perdas de habitat
            habitat_loss = self._estimate_habitat_loss(range_shifts)

            return {
                'range_shifts': range_shifts,
                'habitat_loss': habitat_loss,
                'extinction_risk': self._assess_extinction_risk(habitat_loss)
            }

        return {}

    def _calculate_range_shifts(self):
        """Calcula deslocamentos de distribuição"""
        # Modelo simplificado baseado em gradiente climático
        shifts = {}

        for species in self.ecosystem['species'].unique():
            species_data = self.ecosystem[self.ecosystem['species'] == species]

            # Gradiente latitudinal simplificado
            current_range = species_data['latitude'].agg(['min', 'max'])
            shift_distance = 0.5  # graus por grau de aquecimento

            new_range = {
                'min': current_range['min'] + shift_distance,
                'max': current_range['max'] + shift_distance
            }

            shifts[species] = new_range

        return shifts

    def _estimate_habitat_loss(self, range_shifts):
        """Estima perda de habitat"""
        habitat_loss = {}

        for species, new_range in range_shifts.items():
            # Assumir perda se nova distribuição sai dos limites
            if new_range['max'] > 90 or new_range['min'] < -90:
                habitat_loss[species] = 1.0  # Perda total
            else:
                # Perda proporcional
                overlap = min(90, new_range['max']) - max(-90, new_range['min'])
                original_range = 180  # Range total possível
                habitat_loss[species] = 1 - (overlap / original_range)

        return habitat_loss

    def _assess_extinction_risk(self, habitat_loss):
        """Avalia risco de extinção"""
        extinction_risk = {}

        for species, loss in habitat_loss.items():
            if loss > 0.8:
                risk = 'critical'
            elif loss > 0.5:
                risk = 'high'
            elif loss > 0.2:
                risk = 'medium'
            else:
                risk = 'low'

            extinction_risk[species] = risk

        return extinction_risk
```

### 4.3 Monitoramento Ambiental Inteligente
```python
class EnvironmentalMonitoring:
    """
    Sistema de monitoramento ambiental usando IoT e IA
    """

    def __init__(self):
        self.sensors = {}
        self.data_streams = {}
        self.alerts = []

    def deploy_sensor_network(self, locations, sensor_types):
        """Implanta rede de sensores"""
        for i, location in enumerate(locations):
            sensor_id = f"sensor_{i}"
            sensor = {
                'id': sensor_id,
                'location': location,
                'type': sensor_types[i % len(sensor_types)],
                'active': True,
                'last_reading': None
            }
            self.sensors[sensor_id] = sensor

    def collect_data(self):
        """Coleta dados dos sensores"""
        for sensor_id, sensor in self.sensors.items():
            if sensor['active']:
                # Simulação de leitura de sensor
                reading = self._simulate_sensor_reading(sensor)

                if sensor_id not in self.data_streams:
                    self.data_streams[sensor_id] = []

                self.data_streams[sensor_id].append({
                    'timestamp': pd.Timestamp.now(),
                    'value': reading,
                    'sensor_type': sensor['type']
                })

                # Verificar alertas
                self._check_alerts(sensor_id, reading)

    def _simulate_sensor_reading(self, sensor):
        """Simula leitura de sensor baseada no tipo"""
        sensor_type = sensor['type']

        if sensor_type == 'temperature':
            # Temperatura com variação diurna
            base_temp = 25 + 5 * np.sin(2 * np.pi * pd.Timestamp.now().hour / 24)
            noise = np.random.normal(0, 1)
            return base_temp + noise

        elif sensor_type == 'humidity':
            # Umidade relativa
            base_humidity = 60 + 20 * np.sin(2 * np.pi * pd.Timestamp.now().hour / 24)
            noise = np.random.normal(0, 5)
            return np.clip(base_humidity + noise, 0, 100)

        elif sensor_type == 'co2':
            # Concentração de CO2
            base_co2 = 400 + 50 * np.random.random()
            return base_co2

        else:
            return np.random.random() * 100

    def _check_alerts(self, sensor_id, reading):
        """Verifica condições de alerta"""
        sensor = self.sensors[sensor_id]
        sensor_type = sensor['type']

        # Definições de limites de alerta
        alert_limits = {
            'temperature': {'min': 10, 'max': 40},
            'humidity': {'min': 20, 'max': 90},
            'co2': {'max': 1000}
        }

        if sensor_type in alert_limits:
            limits = alert_limits[sensor_type]

            if 'min' in limits and reading < limits['min']:
                self.alerts.append({
                    'sensor_id': sensor_id,
                    'type': 'low_value',
                    'value': reading,
                    'limit': limits['min'],
                    'timestamp': pd.Timestamp.now()
                })

            if 'max' in limits and reading > limits['max']:
                self.alerts.append({
                    'sensor_id': sensor_id,
                    'type': 'high_value',
                    'value': reading,
                    'limit': limits['max'],
                    'timestamp': pd.Timestamp.now()
                })

    def analyze_trends(self, sensor_id, window_days=7):
        """Analisa tendências nos dados"""
        if sensor_id not in self.data_streams:
            return None

        data = pd.DataFrame(self.data_streams[sensor_id])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')

        # Análise de tendência recente
        recent_data = data.last(f'{window_days}D')

        if len(recent_data) < 2:
            return {'trend': 'insufficient_data'}

        # Regressão linear simples para tendência
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data['value'].values

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        trend = 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'

        return {
            'trend': trend,
            'slope': slope,
            'r_squared': model.score(X, y),
            'data_points': len(recent_data)
        }

    def predict_anomalies(self, sensor_id, forecast_hours=24):
        """Prevê anomalias usando aprendizado de máquina"""
        if sensor_id not in self.data_streams or len(self.data_streams[sensor_id]) < 50:
            return None

        data = pd.DataFrame(self.data_streams[sensor_id])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')

        # Preparar dados para ML
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek

        # Features
        features = ['hour', 'day_of_week']
        X = data[features]

        # Target: desvio da média histórica
        hourly_means = data.groupby('hour')['value'].mean()
        data['expected_value'] = data['hour'].map(hourly_means)
        data['deviation'] = data['value'] - data['expected_value']
        y = (abs(data['deviation']) > data['deviation'].std() * 2).astype(int)

        # Treinar classificador
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Previsões para próximas horas
        future_times = pd.date_range(
            start=data.index[-1] + pd.Timedelta(hours=1),
            periods=forecast_hours,
            freq='H'
        )

        future_features = pd.DataFrame({
            'hour': future_times.hour,
            'day_of_week': future_times.dayofweek
        })

        predictions = clf.predict_proba(future_features)[:, 1]  # Probabilidade de anomalia

        return {
            'predictions': predictions,
            'future_times': future_times,
            'model_accuracy': clf.score(X_test, y_test)
        }
```

---

## 5. METODOLOGIA DE DESENVOLVIMENTO

### 5.1 Estrutura do Projeto de Pesquisa
1. **Definição do Problema**: Questão ecológica específica
2. **Revisão de Literatura**: Estado da arte em modelagem
3. **Coleta de Dados**: Dados empíricos ou simulações
4. **Desenvolvimento do Modelo**: Implementação computacional
5. **Validação**: Comparação com dados observacionais
6. **Análise de Sensibilidade**: Robustez dos resultados
7. **Interpretação**: Implicações ecológicas e aplicações

### 5.2 Boas Práticas de Programação
```python
# Exemplo de estrutura de projeto bem organizada
ecology_project/
├── data/
│   ├── raw/           # Dados brutos
│   ├── processed/     # Dados processados
│   └── metadata/      # Metadados e documentação
├── src/
│   ├── models/        # Modelos ecológicos
│   ├── analysis/      # Scripts de análise
│   ├── visualization/ # Gráficos e visualizações
│   └── utils/         # Funções utilitárias
├── tests/             # Testes unitários
├── docs/              # Documentação
├── requirements.txt   # Dependências
└── README.md         # Documentação principal
```

### 5.3 Controle de Qualidade
- **Testes Estatísticos**: Validação de pressupostos
- **Análise de Sensibilidade**: Robustez dos modelos
- **Validação Cruzada**: Generalização dos resultados
- **Revisão por Pares**: Feedback de especialistas

---

## 6. PROJETOS PRÁTICOS

### 6.1 Projeto 1: Modelagem de Floresta Amazônica
**Objetivo**: Modelar dinâmica de regeneração florestal

**Metodologia**:
1. Coletar dados de inventário florestal
2. Implementar modelo de sucessão ecológica
3. Simular cenários de desmatamento
4. Analisar resiliência do ecossistema

### 6.2 Projeto 2: Rede Alimentar Marinha
**Objetivo**: Analisar estrutura de rede trófica

**Metodologia**:
1. Construir matriz de interações tróficas
2. Calcular métricas de rede (conectância, níveis tróficos)
3. Simular extinções em cascata
4. Identificar espécies-chave

### 6.3 Projeto 3: Monitoramento de Biodiversidade Urbana
**Objetivo**: Avaliar impacto da urbanização

**Metodologia**:
1. Implantar rede de sensores ambientais
2. Coletar dados de biodiversidade
3. Analisar padrões espaço-temporais
4. Desenvolver estratégias de conservação

---

## 7. RECURSOS ADICIONAIS

### 7.1 Livros e Referências
- **Gotelli, N.J. & Colwell, R.K. (2001)**: Quantifying Biodiversity
- **Jørgensen, S.E. (2002)**: Integration of Ecosystem Theories
- **Levin, S.A. (1999)**: Fragile Dominion
- **May, R.M. (1976)**: Theoretical Ecology

### 7.2 Cursos Online
- Coursera: Ecological Modeling
- edX: Introduction to Computational Biology
- DataCamp: Environmental Data Science

### 7.3 Comunidades e Conferências
- Ecological Society of America (ESA)
- International Society for Ecological Modelling
- Society for Conservation Biology

### 7.4 Datasets Públicos
- Global Biodiversity Information Facility (GBIF)
- NASA Earth Observations
- World Wildlife Fund Living Planet Index

---

## 8. CONSIDERAÇÕES FINAIS

A ecologia computacional representa uma ponte crucial entre a teoria ecológica e aplicações práticas. Os métodos apresentados fornecem ferramentas poderosas para:

1. **Compreensão de Sistemas Complexos**: Modelagem de interações ecológicas em múltiplas escalas
2. **Previsão de Mudanças**: Antecipação de impactos ambientais e climáticos
3. **Conservação Estratégica**: Planejamento baseado em evidências científicas
4. **Monitoramento Inteligente**: Sistemas automatizados de vigilância ambiental

O sucesso na aplicação desses métodos depende da integração entre conhecimento ecológico profundo, rigor computacional e validação empírica contínua.

**Próximos Passos Recomendados**:
1. Dominar fundamentos matemáticos e ecológicos
2. Desenvolver proficiência em ferramentas computacionais
3. Participar de projetos de pesquisa colaborativos
4. Contribuir para o avanço da ciência ecológica aplicada

---

*Documento preparado para fine-tuning de IA em Ecologia Computacional*
*Versão 1.0 - Preparado para implementação prática*
