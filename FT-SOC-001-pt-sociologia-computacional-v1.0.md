# FT-SOC-001: Fine-Tuning para IA em Sociologia Computacional

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados em sociologia computacional, integrando análise de redes sociais, modelagem de sistemas sociais complexos e simulação de dinâmica social com métodos computacionais avançados.

### Contexto Filosófico
A sociologia computacional representa a convergência entre a compreensão sociológica tradicional e o poder analítico da computação, permitindo explorar padrões sociais emergentes que transcendem a observação individual para revelar leis gerais do comportamento coletivo.

### Metodologia de Aprendizado Recomendada
1. **Abordagem Sistêmica**: Compreensão de sociedades como sistemas complexos
2. **Análise de Redes**: Modelagem de conexões e interações sociais
3. **Simulação Computacional**: Modelos baseados em agentes e dinâmica social
4. **Análise Empírica**: Validação com dados sociais reais
5. **Interdisciplinaridade**: Integração com psicologia, economia e ciência política

---

## 1. ANÁLISE DE REDES SOCIAIS

### 1.1 Estrutura e Propriedades de Redes
```python
import networkx as nx
import numpy as np
from scipy.stats import powerlaw
import matplotlib.pyplot as plt

class SocialNetworkAnalysis:
    """
    Análise estrutural de redes sociais
    """

    def __init__(self, network_data=None):
        self.G = nx.Graph()
        if network_data:
            self.load_network(network_data)

    def load_network(self, network_data):
        """Carregar rede social"""
        if isinstance(network_data, dict):
            # Formato: {node: [neighbors]}
            for node, neighbors in network_data.items():
                self.G.add_node(node)
                for neighbor in neighbors:
                    self.G.add_edge(node, neighbor)
        else:
            # Assumir que já é um grafo NetworkX
            self.G = network_data

    def network_properties(self):
        """Calcular propriedades básicas da rede"""
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        density = nx.density(self.G)

        # Graus
        degrees = [d for n, d in self.G.degree()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)

        # Clustering
        avg_clustering = nx.average_clustering(self.G)

        # Componentes
        n_components = nx.number_connected_components(self.G)
        largest_component = max(nx.connected_components(self.G), key=len)
        largest_component_size = len(largest_component)

        # Diâmetro (se rede pequena)
        try:
            diameter = nx.diameter(self.G)
        except:
            diameter = None

        # Centralidades
        degree_centrality = nx.degree_centrality(self.G)
        betweenness_centrality = nx.betweenness_centrality(self.G)
        closeness_centrality = nx.closeness_centrality(self.G)

        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'avg_clustering': avg_clustering,
            'n_components': n_components,
            'largest_component_size': largest_component_size,
            'diameter': diameter,
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality
        }

    def degree_distribution_analysis(self):
        """Análise da distribuição de graus"""
        degrees = [d for n, d in self.G.degree()]

        # Distribuição empírica
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        empirical_dist = counts / np.sum(counts)

        # Ajuste a lei de potência (scale-free)
        try:
            # Método de máxima verossimilhança para lei de potência
            alpha = 1 + len(degrees) / np.sum(np.log(degrees / np.min(degrees)))

            # Kolmogorov-Smirnov test
            from scipy.stats import kstest
            # Gerar distribuição teórica
            theoretical_dist = powerlaw.rvs(alpha, size=len(degrees))

            ks_stat, p_value = kstest(degrees, lambda x: powerlaw.cdf(x, alpha))

            power_law_fit = {
                'alpha': alpha,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'is_power_law': p_value > 0.05
            }
        except:
            power_law_fit = None

        return {
            'degrees': degrees,
            'empirical_distribution': dict(zip(unique_degrees, empirical_dist)),
            'power_law_fit': power_law_fit
        }

    def community_detection(self, method='louvain'):
        """Detecção de comunidades"""
        if method == 'louvain':
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(self.G)

                # Modularidade
                modularity = community_louvain.modularity(partition, self.G)

                # Propriedades das comunidades
                communities = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities:
                        communities[comm_id] = []
                    communities[comm_id].append(node)

            except ImportError:
                # Implementação simplificada de Louvain
                partition = self._simple_louvain()
                modularity = self._calculate_modularity(partition)
                communities = self._partition_to_communities(partition)

        elif method == 'girvan_newman':
            communities_generator = nx.algorithms.community.girvan_newman(self.G)
            communities = next(communities_generator)
            modularity = self._calculate_modularity_from_communities(communities)
            partition = self._communities_to_partition(communities)

        return {
            'partition': partition,
            'communities': communities,
            'modularity': modularity,
            'n_communities': len(set(partition.values()))
        }

    def _simple_louvain(self):
        """Implementação simplificada do algoritmo Louvain"""
        # Atribuir cada nó à sua própria comunidade inicialmente
        partition = {node: i for i, node in enumerate(self.G.nodes())}

        # Iterativamente melhorar modularidade
        improved = True
        while improved:
            improved = False
            for node in self.G.nodes():
                best_community = partition[node]
                best_modularity_gain = 0

                # Tentar mover para comunidade dos vizinhos
                neighbor_communities = set(partition[neighbor] for neighbor in self.G.neighbors(node))

                for comm in neighbor_communities:
                    if comm != partition[node]:
                        # Calcular ganho de modularidade
                        gain = self._modularity_gain(node, comm, partition)

                        if gain > best_modularity_gain:
                            best_modularity_gain = gain
                            best_community = comm

                if best_community != partition[node]:
                    partition[node] = best_community
                    improved = True

        return partition

    def _modularity_gain(self, node, community, partition):
        """Calcular ganho de modularidade ao mover nó para comunidade"""
        # Implementação simplificada
        return 0.1  # Valor fixo para demonstração

    def _calculate_modularity(self, partition):
        """Calcular modularidade da partição"""
        # Implementação simplificada
        return 0.3

    def _partition_to_communities(self, partition):
        """Converter partição em dicionário de comunidades"""
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        return communities

    def _calculate_modularity_from_communities(self, communities):
        """Calcular modularidade a partir de comunidades"""
        # Implementação simplificada
        return 0.4

    def _communities_to_partition(self, communities):
        """Converter comunidades em partição"""
        partition = {}
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = comm_id
        return partition

    def small_world_analysis(self):
        """Análise de propriedades small-world"""
        # Coeficiente de clustering médio
        avg_clustering = nx.average_clustering(self.G)

        # Comprimento médio do caminho mais curto
        try:
            avg_shortest_path = nx.average_shortest_path_length(self.G)
        except:
            # Para redes desconectadas
            components = nx.connected_components(self.G)
            avg_shortest_path = np.mean([nx.average_shortest_path_length(self.G.subgraph(c))
                                        for c in components if len(c) > 1])

        # Comparar com rede aleatória
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        p_random = 2 * n_edges / (n_nodes * (n_nodes - 1))

        # Clustering esperado em rede aleatória
        expected_clustering_random = p_random

        # Small-world coefficient
        small_world_coefficient = (avg_clustering / expected_clustering_random) / avg_shortest_path

        return {
            'avg_clustering': avg_clustering,
            'avg_shortest_path': avg_shortest_path,
            'expected_clustering_random': expected_clustering_random,
            'small_world_coefficient': small_world_coefficient,
            'is_small_world': small_world_coefficient > 1
        }

    def influence_propagation(self, seed_nodes, propagation_model='independent_cascade'):
        """Modelagem de propagação de influência"""
        if propagation_model == 'independent_cascade':
            return self._independent_cascade(seed_nodes)
        elif propagation_model == 'linear_threshold':
            return self._linear_threshold(seed_nodes)
        else:
            raise ValueError("Modelo de propagação não suportado")

    def _independent_cascade(self, seed_nodes, propagation_prob=0.1):
        """Modelo Independent Cascade"""
        activated = set(seed_nodes)
        newly_activated = set(seed_nodes)

        while newly_activated:
            next_newly_activated = set()

            for node in newly_activated:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in activated:
                        if np.random.random() < propagation_prob:
                            next_newly_activated.add(neighbor)
                            activated.add(neighbor)

            newly_activated = next_newly_activated

        return {
            'activated_nodes': list(activated),
            'total_activated': len(activated),
            'activation_rate': len(activated) / self.G.number_of_nodes(),
            'propagation_prob': propagation_prob
        }

    def _linear_threshold(self, seed_nodes, threshold=0.5):
        """Modelo Linear Threshold"""
        activated = set(seed_nodes)
        node_thresholds = {node: threshold for node in self.G.nodes()}

        # Calcular influência dos vizinhos
        influence = {}
        for node in self.G.nodes():
            if node not in activated:
                activated_neighbors = [n for n in self.G.neighbors(node) if n in activated]
                influence[node] = len(activated_neighbors) / self.G.degree(node)

        # Ativar nós acima do threshold
        newly_activated = set()
        for node, inf in influence.items():
            if inf >= node_thresholds[node]:
                newly_activated.add(node)

        activated.update(newly_activated)

        return {
            'activated_nodes': list(activated),
            'total_activated': len(activated),
            'average_influence': np.mean(list(influence.values())),
            'threshold': threshold
        }
```

**Análise de Redes Sociais:**
- Propriedades estruturais (densidade, clustering, centralidade)
- Distribuição de graus e leis de potência
- Detecção de comunidades
- Propriedades small-world
- Propagação de influência

### 1.2 Dinâmica Social em Redes
```python
import numpy as np
from scipy.integrate import odeint
import networkx as nx

class SocialDynamics:
    """
    Modelagem da dinâmica social em redes
    """

    def __init__(self, social_network):
        self.network = social_network
        self.opinions = {}
        self.behaviors = {}

    def opinion_dynamics(self, initial_opinions, model='voter', time_steps=100):
        """
        Dinâmica de formação de opinião
        """
        if model == 'voter':
            return self._voter_model(initial_opinions, time_steps)
        elif model == 'majority':
            return self._majority_model(initial_opinions, time_steps)
        elif model == 'bounded_confidence':
            return self._bounded_confidence_model(initial_opinions, time_steps)
        else:
            raise ValueError("Modelo de opinião não suportado")

    def _voter_model(self, initial_opinions, time_steps):
        """Modelo do eleitor (Voter Model)"""
        opinions = np.array(initial_opinions.copy())
        nodes = list(self.network.nodes())

        opinion_history = [opinions.copy()]

        for t in range(time_steps):
            # Escolher nó aleatoriamente
            node = np.random.choice(nodes)

            # Se não tiver vizinhos, pular
            neighbors = list(self.network.neighbors(node))
            if not neighbors:
                continue

            # Copiar opinião de um vizinho aleatório
            neighbor = np.random.choice(neighbors)
            opinions[node] = opinions[neighbor]

            opinion_history.append(opinions.copy())

        # Análise de consenso
        final_opinions = opinion_history[-1]
        consensus_reached = len(np.unique(final_opinions)) == 1

        return {
            'opinion_history': opinion_history,
            'final_opinions': final_opinions,
            'consensus_reached': consensus_reached,
            'consensus_time': time_steps if consensus_reached else None,
            'dominant_opinion': np.argmax(np.bincount(final_opinions.astype(int)))
        }

    def _majority_model(self, initial_opinions, time_steps):
        """Modelo da maioria"""
        opinions = np.array(initial_opinions.copy())
        nodes = list(self.network.nodes())

        opinion_history = [opinions.copy()]

        for t in range(time_steps):
            # Escolher nó aleatoriamente
            node = np.random.choice(nodes)

            neighbors = list(self.network.neighbors(node))
            if not neighbors:
                continue

            # Contar opiniões dos vizinhos
            neighbor_opinions = opinions[neighbors]
            majority_opinion = np.argmax(np.bincount(neighbor_opinions.astype(int)))

            # Adotar opinião da maioria
            opinions[node] = majority_opinion

            opinion_history.append(opinions.copy())

        return {
            'opinion_history': opinion_history,
            'final_opinions': opinion_history[-1],
            'opinion_distribution': np.bincount(opinion_history[-1].astype(int))
        }

    def _bounded_confidence_model(self, initial_opinions, time_steps, confidence_threshold=0.3):
        """Modelo de confiança limitada (Deffuant-Weisbuch)"""
        opinions = np.array(initial_opinions.copy(), dtype=float)
        nodes = list(self.network.nodes())

        opinion_history = [opinions.copy()]

        for t in range(time_steps):
            # Escolher par de vizinhos
            node_i = np.random.choice(nodes)
            neighbors_i = list(self.network.neighbors(node_i))

            if not neighbors_i:
                continue

            node_j = np.random.choice(neighbors_i)

            # Diferença de opiniões
            opinion_diff = abs(opinions[node_i] - opinions[node_j])

            # Se diferença menor que threshold, interagir
            if opinion_diff < confidence_threshold:
                # Atualizar opiniões
                mu = 0.5  # Parâmetro de convergência
                new_opinion_i = opinions[node_i] + mu * (opinions[node_j] - opinions[node_i])
                new_opinion_j = opinions[node_j] + mu * (opinions[node_i] - opinions[node_j])

                opinions[node_i] = new_opinion_i
                opinions[node_j] = new_opinion_j

            opinion_history.append(opinions.copy())

        # Análise de clusters de opinião
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        opinion_clusters = kmeans.fit_predict(opinions.reshape(-1, 1))

        return {
            'opinion_history': opinion_history,
            'final_opinions': opinion_history[-1],
            'opinion_clusters': opinion_clusters,
            'cluster_centers': kmeans.cluster_centers_.flatten(),
            'confidence_threshold': confidence_threshold
        }

    def social_contagion(self, initial_infected, infection_rate=0.1, recovery_rate=0.05):
        """
        Modelos de contágio social
        """
        nodes = list(self.network.nodes())
        infected = set(initial_infected)
        susceptible = set(nodes) - infected
        recovered = set()

        infection_history = [len(infected)]
        susceptible_history = [len(susceptible)]
        recovered_history = [len(recovered)]

        max_steps = 1000
        for step in range(max_steps):
            new_infections = set()
            new_recoveries = set()

            # Infecção
            for node in infected.copy():
                for neighbor in self.network.neighbors(node):
                    if neighbor in susceptible and np.random.random() < infection_rate:
                        new_infections.add(neighbor)

            # Recuperação
            for node in infected:
                if np.random.random() < recovery_rate:
                    new_recoveries.add(node)

            # Atualizar estados
            infected.update(new_infections)
            infected -= new_recoveries

            susceptible -= new_infections
            recovered.update(new_recoveries)

            # Registrar histórico
            infection_history.append(len(infected))
            susceptible_history.append(len(susceptible))
            recovered_history.append(len(recovered))

            # Verificar equilíbrio
            if len(new_infections) == 0 and len(new_recoveries) == 0:
                break

        return {
            'infection_history': infection_history,
            'susceptible_history': susceptible_history,
            'recovered_history': recovered_history,
            'final_infected': len(infected),
            'total_infected': len(infected) + len(recovered),
            'infection_rate': infection_rate,
            'recovery_rate': recovery_rate
        }

    def segregation_dynamics(self, initial_groups, similarity_threshold=0.5):
        """
        Dinâmica de segregação (Modelo de Schelling)
        """
        nodes = list(self.network.nodes())

        # Atribuir grupos iniciais
        groups = np.array(initial_groups)
        positions = np.random.rand(len(nodes), 2)  # Posições espaciais

        # Simulação
        max_steps = 100
        moved_agents = []

        for step in range(max_steps):
            moved_in_step = 0

            for i, node in enumerate(nodes):
                neighbors = list(self.network.neighbors(node))
                if not neighbors:
                    continue

                # Calcular satisfação
                neighbor_groups = groups[[nodes.index(n) for n in neighbors]]
                similar_neighbors = np.sum(neighbor_groups == groups[i])
                satisfaction = similar_neighbors / len(neighbors)

                # Se insatisfeito, mover para posição aleatória
                if satisfaction < similarity_threshold:
                    # Encontrar novo vizinho similar
                    potential_moves = []
                    for j, other_node in enumerate(nodes):
                        if i != j and groups[j] == groups[i]:
                            other_neighbors = list(self.network.neighbors(other_node))
                            if other_neighbors:
                                other_neighbor_groups = groups[[nodes.index(n) for n in other_neighbors]]
                                other_similar = np.sum(other_neighbor_groups == groups[j])
                                other_satisfaction = other_similar / len(other_neighbors)

                                if other_satisfaction >= satisfaction:
                                    potential_moves.append(j)

                    # Mover se possível
                    if potential_moves:
                        move_to = np.random.choice(potential_moves)
                        # Trocar posições
                        positions[i], positions[move_to] = positions[move_to], positions[i]
                        moved_in_step += 1

            moved_agents.append(moved_in_step)

            # Parar se não houve movimentos
            if moved_in_step == 0:
                break

        # Calcular índice de segregação
        segregation_index = self._calculate_segregation_index(groups, positions)

        return {
            'final_positions': positions,
            'moved_agents_history': moved_agents,
            'segregation_index': segregation_index,
            'similarity_threshold': similarity_threshold,
            'simulation_steps': step + 1
        }

    def _calculate_segregation_index(self, groups, positions):
        """Calcular índice de segregação"""
        # Índice de dissimilaridade
        total_pairs = 0
        dissimilar_pairs = 0

        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                total_pairs += 1
                if groups[i] != groups[j]:
                    dissimilar_pairs += 1

        dissimilarity_index = dissimilar_pairs / total_pairs if total_pairs > 0 else 0

        return dissimilarity_index

    def cooperation_evolution(self, initial_strategies, benefit=2, cost=1):
        """
        Evolução da cooperação (Dilema do Prisioneiro)
        """
        nodes = list(self.network.nodes())
        strategies = np.array(initial_strategies)  # 0: Defect, 1: Cooperate

        strategy_history = [strategies.copy()]
        cooperation_rates = [np.mean(strategies)]

        # Parâmetros do jogo
        R = benefit - cost  # Recompensa mútua
        T = benefit         # Tentação
        S = -cost          # Punição
        P = 0              # Punição mútua

        payoff_matrix = np.array([
            [P, S],  # Defect vs Defect, Defect vs Cooperate
            [T, R]   # Cooperate vs Defect, Cooperate vs Cooperate
        ])

        # Simulação
        for generation in range(50):
            new_strategies = strategies.copy()
            payoffs = np.zeros(len(nodes))

            # Calcular payoffs
            for i, node in enumerate(nodes):
                neighbors = list(self.network.neighbors(node))

                if neighbors:
                    for neighbor in neighbors:
                        j = nodes.index(neighbor)
                        payoffs[i] += payoff_matrix[strategies[i], strategies[j]]

            # Reprodução proporcional ao payoff
            for i in range(len(nodes)):
                # Probabilidade de imitar estratégia de vizinho bem-sucedido
                if neighbors:
                    neighbor_payoffs = payoffs[[nodes.index(n) for n in neighbors]]
                    best_neighbor_idx = np.argmax(neighbor_payoffs)
                    best_neighbor = neighbors[best_neighbor_idx]

                    if payoffs[nodes.index(best_neighbor)] > payoffs[i]:
                        new_strategies[i] = strategies[nodes.index(best_neighbor)]

            strategies = new_strategies
            strategy_history.append(strategies.copy())
            cooperation_rates.append(np.mean(strategies))

        return {
            'strategy_history': strategy_history,
            'cooperation_rates': cooperation_rates,
            'final_cooperation_rate': cooperation_rates[-1],
            'payoff_matrix': payoff_matrix,
            'game_parameters': {'benefit': benefit, 'cost': cost}
        }
```

**Dinâmica Social:**
- Formação de opinião (Voter, Majority, Bounded Confidence)
- Contágio social e difusão de informação
- Segregação e dinâmica espacial
- Evolução da cooperação

---

## 2. MODELAGEM BASEADA EM AGENTES

### 2.1 Sistemas Sociais Baseados em Agentes
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class AgentBasedSocialSystems:
    """
    Modelagem de sistemas sociais baseada em agentes
    """

    def __init__(self, n_agents=100, environment_size=(10, 10)):
        self.n_agents = n_agents
        self.environment_size = environment_size
        self.agents = []
        self.initialize_agents()

    def initialize_agents(self):
        """Inicializar agentes com propriedades sociais"""
        for i in range(self.n_agents):
            agent = {
                'id': i,
                'position': np.random.rand(2) * np.array(self.environment_size),
                'wealth': np.random.exponential(100),  # Distribuição exponencial
                'social_status': np.random.uniform(0, 1),
                'education': np.random.uniform(0, 1),
                'opinion': np.random.choice([-1, 1]),  # Polarizado
                'happiness': np.random.uniform(0, 1),
                'social_network': [],  # Conexões sociais
                'behavioral_traits': {
                    'risk_aversion': np.random.uniform(0, 1),
                    'altruism': np.random.uniform(0, 1),
                    'conformity': np.random.uniform(0, 1)
                }
            }
            self.agents.append(agent)

    def social_interaction_model(self, interaction_radius=1.0, time_steps=100):
        """
        Modelo de interações sociais espaciais
        """
        positions = np.array([agent['position'] for agent in self.agents])

        for step in range(time_steps):
            # Calcular distâncias entre agentes
            distances = squareform(pdist(positions))

            # Interações sociais
            for i in range(self.n_agents):
                # Encontrar vizinhos próximos
                neighbors = np.where(distances[i] < interaction_radius)[0]
                neighbors = neighbors[neighbors != i]

                if len(neighbors) > 0:
                    # Escolher vizinho aleatório
                    j = np.random.choice(neighbors)

                    # Interação social
                    self._social_interaction(i, j)

                    # Possível formação de conexão social
                    if np.random.random() < 0.1:  # Probabilidade de conexão
                        if j not in self.agents[i]['social_network']:
                            self.agents[i]['social_network'].append(j)
                            self.agents[j]['social_network'].append(i)

            # Movimento dos agentes
            self._agent_movement()

        return self._analyze_social_structure()

    def _social_interaction(self, i, j):
        """Interação entre dois agentes"""
        agent_i = self.agents[i]
        agent_j = self.agents[j]

        # Influência na opinião
        opinion_diff = agent_j['opinion'] - agent_i['opinion']
        conformity_i = agent_i['behavioral_traits']['conformity']
        conformity_j = agent_j['behavioral_traits']['conformity']

        # Mudança de opinião baseada em conformidade
        if abs(opinion_diff) > 0:
            change_i = conformity_i * opinion_diff * 0.1
            change_j = conformity_j * (-opinion_diff) * 0.1

            agent_i['opinion'] = np.clip(agent_i['opinion'] + change_i, -1, 1)
            agent_j['opinion'] = np.clip(agent_j['opinion'] + change_j, -1, 1)

        # Troca econômica
        if np.random.random() < 0.05:  # Probabilidade de troca
            transfer_amount = min(agent_i['wealth'], agent_j['wealth']) * 0.1
            agent_i['wealth'] -= transfer_amount
            agent_j['wealth'] += transfer_amount

    def _agent_movement(self):
        """Movimento dos agentes no ambiente"""
        for agent in self.agents:
            # Movimento browniano simples
            displacement = np.random.normal(0, 0.1, 2)
            agent['position'] += displacement

            # Manter dentro dos limites
            agent['position'] = np.clip(agent['position'], 0, np.array(self.environment_size))

    def _analyze_social_structure(self):
        """Analisar estrutura social emergente"""
        # Rede social
        social_network = {}
        for i, agent in enumerate(self.agents):
            social_network[i] = agent['social_network']

        # Distribuição de riqueza
        wealths = [agent['wealth'] for agent in self.agents]
        gini_coefficient = self._calculate_gini_coefficient(wealths)

        # Polarização de opinião
        opinions = [agent['opinion'] for agent in self.agents]
        opinion_polarization = np.abs(np.mean(opinions))

        # Clustering social
        positions = np.array([agent['position'] for agent in self.agents])
        social_clusters = self._identify_social_clusters(positions)

        return {
            'social_network': social_network,
            'wealth_distribution': {
                'gini_coefficient': gini_coefficient,
                'mean_wealth': np.mean(wealths),
                'wealth_std': np.std(wealths)
            },
            'opinion_distribution': {
                'polarization': opinion_polarization,
                'opinion_std': np.std(opinions)
            },
            'social_clusters': social_clusters
        }

    def _calculate_gini_coefficient(self, values):
        """Calcular coeficiente de Gini"""
        values = np.array(values)
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)

        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _identify_social_clusters(self, positions):
        """Identificar clusters sociais espaciais"""
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=1.0, min_samples=3)
        clusters = clustering.fit_predict(positions)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        return {
            'n_clusters': n_clusters,
            'cluster_labels': clusters,
            'noise_points': np.sum(clusters == -1)
        }

    def urban_growth_simulation(self, initial_population=50, max_steps=200):
        """
        Simulação de crescimento urbano baseado em agentes
        """
        # Inicializar com população menor
        self.n_agents = initial_population
        self.agents = self.agents[:initial_population]

        population_history = [initial_population]
        urban_density = []

        for step in range(max_steps):
            # Nascimento de novos agentes
            if np.random.random() < 0.02:  # Taxa de nascimento
                new_agent = self._create_new_agent()
                self.agents.append(new_agent)
                self.n_agents += 1

            # Morte de agentes
            if np.random.random() < 0.01:  # Taxa de mortalidade
                if self.n_agents > 10:  # Manter população mínima
                    deceased_idx = np.random.randint(self.n_agents)
                    self.agents.pop(deceased_idx)
                    self.n_agents -= 1

            # Migração
            for agent in self.agents:
                if np.random.random() < 0.005:  # Probabilidade de migração
                    # Migrar para área com maior densidade social
                    self._migrate_agent(agent)

            population_history.append(self.n_agents)

            # Calcular densidade urbana
            positions = np.array([agent['position'] for agent in self.agents])
            density = self.n_agents / (self.environment_size[0] * self.environment_size[1])
            urban_density.append(density)

        return {
            'population_history': population_history,
            'urban_density_history': urban_density,
            'final_population': self.n_agents,
            'urban_expansion_rate': (self.n_agents - initial_population) / max_steps
        }

    def _create_new_agent(self):
        """Criar novo agente"""
        return {
            'id': self.n_agents,
            'position': np.random.rand(2) * np.array(self.environment_size),
            'wealth': np.random.exponential(50),  # Menos rico inicialmente
            'social_status': np.random.uniform(0, 0.5),  # Status mais baixo
            'education': np.random.uniform(0, 0.5),
            'opinion': np.random.choice([-1, 1]),
            'happiness': np.random.uniform(0.3, 0.7),
            'social_network': [],
            'behavioral_traits': {
                'risk_aversion': np.random.uniform(0, 1),
                'altruism': np.random.uniform(0, 1),
                'conformity': np.random.uniform(0, 1)
            }
        }

    def _migrate_agent(self, agent):
        """Migrar agente para área mais densa"""
        # Encontrar centroide dos agentes existentes
        positions = np.array([a['position'] for a in self.agents])
        centroid = np.mean(positions, axis=0)

        # Mover em direção ao centroide
        direction = centroid - agent['position']
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            agent['position'] += direction * 0.5

        # Manter dentro dos limites
        agent['position'] = np.clip(agent['position'], 0, np.array(self.environment_size))

    def cultural_evolution_model(self, n_generations=50, mutation_rate=0.01):
        """
        Modelo de evolução cultural
        """
        # Representar cultura como vetor de características
        culture_dimensions = ['values', 'norms', 'beliefs', 'practices']

        # Inicializar culturas dos agentes
        for agent in self.agents:
            agent['culture'] = {
                dim: np.random.uniform(0, 1) for dim in culture_dimensions
            }

        culture_history = []

        for generation in range(n_generations):
            # Interações culturais
            for i in range(self.n_agents):
                # Escolher parceiro de interação
                j = np.random.randint(self.n_agents)
                if i != j:
                    self._cultural_interaction(i, j)

            # Seleção cultural
            self._cultural_selection()

            # Mutação
            self._cultural_mutation(mutation_rate)

            # Registrar diversidade cultural
            cultures = [agent['culture'] for agent in self.agents]
            diversity = self._calculate_cultural_diversity(cultures)
            culture_history.append(diversity)

        return {
            'culture_history': culture_history,
            'final_cultures': [agent['culture'] for agent in self.agents],
            'cultural_diversity': culture_history[-1]
        }

    def _cultural_interaction(self, i, j):
        """Interação cultural entre dois agentes"""
        culture_i = self.agents[i]['culture']
        culture_j = self.agents[j]['culture']

        # Difusão cultural
        for dim in culture_i.keys():
            # Convergência cultural
            diff = culture_j[dim] - culture_i[dim]
            culture_i[dim] += 0.1 * diff
            culture_j[dim] -= 0.1 * diff

    def _cultural_selection(self):
        """Seleção cultural baseada em adequação"""
        # Agentes com cultura mais "adequada" têm mais probabilidade de influenciar outros
        fitness_scores = []

        for agent in self.agents:
            # Adequação baseada em riqueza e felicidade
            fitness = agent['wealth'] * 0.6 + agent['happiness'] * 0.4
            fitness_scores.append(fitness)

        # Normalizar
        fitness_scores = np.array(fitness_scores)
        fitness_scores = fitness_scores / np.sum(fitness_scores)

        # Reamostragem baseada em adequação
        selected_indices = np.random.choice(
            self.n_agents, self.n_agents, p=fitness_scores
        )

        new_agents = [self.agents[i].copy() for i in selected_indices]
        self.agents = new_agents

    def _cultural_mutation(self, mutation_rate):
        """Mutação cultural"""
        for agent in self.agents:
            if np.random.random() < mutation_rate:
                # Mutar dimensão cultural aleatória
                dim = np.random.choice(list(agent['culture'].keys()))
                agent['culture'][dim] += np.random.normal(0, 0.1)
                agent['culture'][dim] = np.clip(agent['culture'][dim], 0, 1)

    def _calculate_cultural_diversity(self, cultures):
        """Calcular diversidade cultural"""
        culture_vectors = np.array([[c[dim] for dim in cultures[0].keys()] for c in cultures])
        diversity = np.mean(np.std(culture_vectors, axis=0))
        return diversity
```

**Modelagem Baseada em Agentes:**
- Interações sociais espaciais
- Simulação de crescimento urbano
- Evolução cultural
- Dinâmica de opinião e comportamento

### 2.2 Análise de Sistemas Sociais Complexos
```python
import numpy as np
from scipy.integrate import odeint
import networkx as nx

class ComplexSocialSystems:
    """
    Análise de sistemas sociais complexos
    """

    def __init__(self, system_parameters=None):
        self.params = system_parameters or {}
        self.system_state = {}

    def epidemiological_social_model(self, initial_conditions, time_span=100):
        """
        Modelo epidemiológico para difusão social (SIS, SIR)
        """
        def sir_model(y, t, beta, gamma, contact_rate):
            """Modelo SIR: Susceptible-Infected-Recovered"""
            S, I, R = y

            dS_dt = -beta * S * I * contact_rate
            dI_dt = beta * S * I * contact_rate - gamma * I
            dR_dt = gamma * I

            return [dS_dt, dI_dt, dR_dt]

        # Parâmetros
        beta = self.params.get('infection_rate', 0.3)      # Taxa de infecção
        gamma = self.params.get('recovery_rate', 0.1)      # Taxa de recuperação
        contact_rate = self.params.get('contact_rate', 1.0) # Taxa de contato

        # Integração
        t = np.linspace(0, time_span, 1000)
        solution = odeint(sir_model, initial_conditions, t, args=(beta, gamma, contact_rate))

        S, I, R = solution.T

        # Análise de pontos críticos
        r0 = beta / gamma  # Número básico de reprodução

        # Ponto de equilíbrio
        if r0 > 1:
            equilibrium_S = 1/r0
            equilibrium_I = 0
        else:
            equilibrium_S = 1
            equilibrium_I = 0

        return {
            'time': t,
            'susceptible': S,
            'infected': I,
            'recovered': R,
            'r0': r0,
            'equilibrium_S': equilibrium_S,
            'equilibrium_I': equilibrium_I,
            'total_infected_peak': np.max(I),
            'final_recovered': R[-1]
        }

    def economic_inequality_dynamics(self, initial_wealth_distribution, time_steps=100):
        """
        Dinâmica de desigualdade econômica
        """
        n_agents = len(initial_wealth_distribution)
        wealth = np.array(initial_wealth_distribution)

        wealth_history = [wealth.copy()]
        gini_history = [self._gini_coefficient(wealth)]

        # Parâmetros
        savings_rate = self.params.get('savings_rate', 0.1)
        investment_return = self.params.get('investment_return', 0.05)
        redistribution_rate = self.params.get('redistribution_rate', 0.01)

        for t in range(time_steps):
            # Crescimento econômico
            wealth *= (1 + investment_return)

            # Poupança (efeito multiplicador)
            savings = wealth * savings_rate
            wealth += savings * investment_return

            # Interações econômicas (troca)
            for _ in range(n_agents // 10):  # Algumas interações por período
                i, j = np.random.choice(n_agents, 2, replace=False)

                # Troca proporcional à diferença de riqueza
                transfer = (wealth[j] - wealth[i]) * 0.01
                wealth[i] += transfer
                wealth[j] -= transfer

            # Redistribuição (impostos/progressividade)
            mean_wealth = np.mean(wealth)
            for i in range(n_agents):
                if wealth[i] > mean_wealth:
                    tax = (wealth[i] - mean_wealth) * redistribution_rate
                    wealth[i] -= tax

                    # Redistribuir igualmente
                    wealth += tax / n_agents

            # Evitar riqueza negativa
            wealth = np.maximum(wealth, 0.1)

            wealth_history.append(wealth.copy())
            gini_history.append(self._gini_coefficient(wealth))

        return {
            'wealth_history': wealth_history,
            'gini_history': gini_history,
            'final_wealth_distribution': wealth,
            'final_gini': gini_history[-1],
            'wealth_concentration_ratio': np.max(wealth) / np.min(wealth)
        }

    def _gini_coefficient(self, values):
        """Calcular coeficiente de Gini"""
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def social_norm_evolution(self, initial_norms, adaptation_rate=0.1):
        """
        Evolução de normas sociais
        """
        norms = np.array(initial_norms)
        norm_history = [norms.copy()]

        # Simulação de evolução
        for generation in range(50):
            # Pressão social
            social_pressure = np.mean(norms) - norms

            # Adaptação
            norms += adaptation_rate * social_pressure

            # Mutação pequena
            norms += np.random.normal(0, 0.01, len(norms))

            # Manter entre 0 e 1
            norms = np.clip(norms, 0, 1)

            norm_history.append(norms.copy())

        # Análise de convergência
        final_norms = norm_history[-1]
        norm_convergence = 1 - np.std(final_norms)

        return {
            'norm_history': norm_history,
            'final_norms': final_norms,
            'norm_convergence': norm_convergence,
            'norm_homogeneity': 1 - np.std(final_norms)
        }

    def collective_decision_making(self, agent_preferences, voting_system='majority'):
        """
        Tomada de decisão coletiva
        """
        if voting_system == 'majority':
            return self._majority_voting(agent_preferences)
        elif voting_system == 'ranked_choice':
            return self._ranked_choice_voting(agent_preferences)
        elif voting_system == 'approval':
            return self._approval_voting(agent_preferences)
        else:
            raise ValueError("Sistema de votação não suportado")

    def _majority_voting(self, agent_preferences):
        """Votação por maioria simples"""
        n_options = len(agent_preferences[0])
        votes = np.zeros(n_options)

        for preference in agent_preferences:
            winner = np.argmax(preference)
            votes[winner] += 1

        winning_option = np.argmax(votes)

        return {
            'votes': votes,
            'winning_option': winning_option,
            'winning_percentage': votes[winning_option] / len(agent_preferences),
            'voter_satisfaction': np.mean([pref[winning_option] for pref in agent_preferences])
        }

    def _ranked_choice_voting(self, agent_preferences):
        """Votação por escolha classificada (IRV)"""
        n_options = len(agent_preferences[0])
        remaining_options = list(range(n_options))

        while len(remaining_options) > 1:
            # Contar primeiros lugares
            first_choice_votes = np.zeros(n_options)

            for preference in agent_preferences:
                # Encontrar primeira escolha entre opções restantes
                for choice in preference:
                    if choice in remaining_options:
                        first_choice_votes[choice] += 1
                        break

            # Eliminar opção com menos votos
            min_votes = np.min([first_choice_votes[opt] for opt in remaining_options])
            eliminated = [opt for opt in remaining_options if first_choice_votes[opt] == min_votes]

            if eliminated:
                remaining_options.remove(eliminated[0])

        winning_option = remaining_options[0]

        return {
            'winning_option': winning_option,
            'elimination_rounds': n_options - 1,
            'final_round_votes': first_choice_votes
        }

    def _approval_voting(self, agent_preferences):
        """Votação por aprovação"""
        n_options = len(agent_preferences[0])
        approval_votes = np.zeros(n_options)

        # Threshold de aprovação
        approval_threshold = 0.6

        for preference in agent_preferences:
            approved_options = [i for i, pref in enumerate(preference) if pref >= approval_threshold]

            for option in approved_options:
                approval_votes[option] += 1

        winning_option = np.argmax(approval_votes)

        return {
            'approval_votes': approval_votes,
            'winning_option': winning_option,
            'approval_rate': approval_votes[winning_option] / len(agent_preferences)
        }

    def social_tipping_points(self, initial_adoption, innovation_attractiveness=0.1):
        """
        Pontos de virada social (tipping points)
        """
        adoption = initial_adoption
        adoption_history = [adoption]

        # Modelo de difusão com efeito de rede
        network_effect = 0.1
        social_influence = 0.05

        # Simulação
        for t in range(100):
            # Adoção baseada em atratividade + efeitos sociais
            new_adopters = (1 - adoption) * (
                innovation_attractiveness +
                network_effect * adoption +
                social_influence * adoption**2  # Efeito quadrático
            )

            adoption += new_adopters
            adoption = min(adoption, 1.0)  # Limite superior

            adoption_history.append(adoption)

            # Verificar ponto de virada
            if adoption > 0.5 and adoption_history[-2] <= 0.5:
                tipping_point = t
                break
        else:
            tipping_point = None

        return {
            'adoption_history': adoption_history,
            'final_adoption': adoption,
            'tipping_point': tipping_point,
            'adoption_rate_at_tipping': adoption_history[tipping_point] if tipping_point else None
        }

    def cultural_transmission_model(self, n_generations=20, transmission_bias=0.1):
        """
        Modelo de transmissão cultural
        """
        # Traços culturais
        cultural_traits = ['cooperative', 'competitive', 'altruistic', 'selfish']
        population_size = 100

        # Inicializar população
        population = np.random.choice(cultural_traits, population_size)

        cultural_history = [np.bincount([cultural_traits.index(t) for t in population],
                                      minlength=len(cultural_traits)) / population_size]

        for generation in range(n_generations):
            new_population = []

            for _ in range(population_size):
                # Selecionar pai (com viés de transmissão)
                if np.random.random() < transmission_bias:
                    # Transmissão cultural enviesada
                    trait_counts = np.bincount([cultural_traits.index(t) for t in population])
                    parent_trait = cultural_traits[np.argmax(trait_counts)]
                else:
                    # Transmissão aleatória
                    parent_trait = np.random.choice(population)

                new_population.append(parent_trait)

            population = new_population

            # Registrar distribuição
            trait_distribution = np.bincount([cultural_traits.index(t) for t in population],
                                           minlength=len(cultural_traits)) / population_size
            cultural_history.append(trait_distribution)

        return {
            'cultural_history': cultural_history,
            'final_distribution': cultural_history[-1],
            'dominant_trait': cultural_traits[np.argmax(cultural_history[-1])],
            'cultural_diversity': np.sum(cultural_history[-1] > 0.01)  # Número de traços presentes
        }
```

**Sistemas Sociais Complexos:**
- Modelos epidemiológicos sociais
- Dinâmica de desigualdade econômica
- Evolução de normas sociais
- Tomada de decisão coletiva
- Pontos de virada social
- Transmissão cultural

---

## 3. APLICACOES COMPUTACIONAIS AVANÇADAS

### 3.1 Aprendizado de Máquina para Análise Social
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class SocialMachineLearning:
    """
    Aprendizado de máquina para análise sociológica
    """

    def __init__(self):
        self.social_models = {}

    def social_behavior_prediction(self, behavioral_data, target_behavior):
        """
        Predição de comportamento social usando aprendizado de máquina
        """
        # Preparar dados
        X = behavioral_data.drop(target_behavior, axis=1)
        y = behavioral_data[target_behavior]

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo de floresta aleatória
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predições
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Avaliação
        report = classification_report(y_test, predictions, output_dict=True)

        # Importância de características
        feature_importance = dict(zip(X.columns, model.feature_importances_))

        return {
            'model': model,
            'predictions': predictions,
            'probabilities': probabilities,
            'classification_report': report,
            'feature_importance': feature_importance,
            'accuracy': report['accuracy']
        }

    def social_network_embedding(self, network_data, embedding_dimension=128):
        """
        Incorporação de redes sociais (node embedding)
        """
        # Implementação simplificada de DeepWalk/Skip-gram

        class SimpleNodeEmbedding:
            def __init__(self, network, embedding_dim):
                self.network = network
                self.embedding_dim = embedding_dim
                self.node_embeddings = {}

            def random_walk_embedding(self, walk_length=10, n_walks=5):
                """Incorporação baseada em caminhadas aleatórias"""
                nodes = list(self.network.nodes())

                # Inicializar incorporações
                for node in nodes:
                    self.node_embeddings[node] = np.random.randn(self.embedding_dim) * 0.1

                # Gerar caminhadas aleatórias
                walks = []
                for _ in range(n_walks):
                    for node in nodes:
                        walk = self._random_walk(node, walk_length)
                        walks.append(walk)

                # Treinar incorporações (simplificado)
                learning_rate = 0.01

                for walk in walks:
                    for i in range(len(walk) - 1):
                        current_node = walk[i]
                        context_node = walk[i + 1]

                        # Atualizar incorporações
                        current_emb = self.node_embeddings[current_node]
                        context_emb = self.node_embeddings[context_node]

                        # Gradiente simplificado
                        gradient = current_emb - context_emb
                        self.node_embeddings[current_node] -= learning_rate * gradient
                        self.node_embeddings[context_node] += learning_rate * gradient

                return self.node_embeddings

            def _random_walk(self, start_node, walk_length):
                """Caminhada aleatória na rede"""
                walk = [start_node]

                for _ in range(walk_length - 1):
                    current_node = walk[-1]
                    neighbors = list(self.network.neighbors(current_node))

                    if neighbors:
                        next_node = np.random.choice(neighbors)
                        walk.append(next_node)
                    else:
                        break

                return walk

        embedder = SimpleNodeEmbedding(network_data, embedding_dimension)
        embeddings = embedder.random_walk_embedding()

        # Análise das incorporações
        embedding_matrix = np.array(list(embeddings.values()))
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embedding_matrix)

        return {
            'embeddings': embeddings,
            'embeddings_2d': embeddings_2d,
            'explained_variance': pca.explained_variance_ratio_,
            'embedding_dimension': embedding_dimension
        }

    def opinion_mining_social_media(self, social_media_data, sentiment_lexicon):
        """
        Mineração de opinião em mídia social
        """
        # Análise de sentimento simplificada
        sentiments = []

        for post in social_media_data['text']:
            sentiment_score = self._calculate_sentiment_score(post, sentiment_lexicon)
            sentiments.append(sentiment_score)

        # Estatísticas de sentimento
        sentiment_stats = {
            'mean_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'positive_posts': np.sum(np.array(sentiments) > 0),
            'negative_posts': np.sum(np.array(sentiments) < 0),
            'neutral_posts': np.sum(np.array(sentiments) == 0)
        }

        # Análise temporal
        if 'timestamp' in social_media_data:
            time_sentiments = pd.DataFrame({
                'timestamp': pd.to_datetime(social_media_data['timestamp']),
                'sentiment': sentiments
            })

            # Média móvel de sentimento
            time_sentiments['sentiment_ma'] = time_sentiments['sentiment'].rolling(window=7).mean()

            sentiment_stats['temporal_analysis'] = time_sentiments

        return {
            'sentiments': sentiments,
            'sentiment_statistics': sentiment_stats,
            'sentiment_lexicon': sentiment_lexicon
        }

    def _calculate_sentiment_score(self, text, lexicon):
        """Calcular pontuação de sentimento"""
        words = text.lower().split()
        sentiment_score = 0

        for word in words:
            if word in lexicon:
                sentiment_score += lexicon[word]

        return sentiment_score

    def social_impact_prediction(self, intervention_data, outcome_measures):
        """
        Predição de impacto social de intervenções
        """
        # Modelo de regressão para impacto
        X = intervention_data
        y = outcome_measures

        # Modelo Gradient Boosting
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predições
        predictions = model.predict(X)

        # Avaliação
        mse = mean_squared_error(y, predictions)
        r_squared = 1 - mse / np.var(y)

        # Importância das características
        feature_importance = dict(zip(X.columns, model.feature_importances_))

        return {
            'model': model,
            'predictions': predictions,
            'mse': mse,
            'r_squared': r_squared,
            'feature_importance': feature_importance,
            'predicted_impact': np.mean(predictions)
        }

    def behavioral_segmentation(self, customer_data, n_segments=5):
        """
        Segmentação comportamental de consumidores
        """
        # Clustering baseado em comportamento
        features = customer_data.drop('customer_id', axis=1)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_segments, random_state=42)
        segments = kmeans.fit_predict(features_scaled)

        # Características dos segmentos
        segment_profiles = []

        for i in range(n_segments):
            segment_data = customer_data[segments == i]
            profile = {
                'segment_id': i,
                'size': len(segment_data),
                'percentage': len(segment_data) / len(customer_data),
                'centroid': kmeans.cluster_centers_[i],
                'feature_means': segment_data.mean()
            }
            segment_profiles.append(profile)

        return {
            'segments': segments,
            'segment_profiles': segment_profiles,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }

    def causal_inference_social_science(self, treatment_variable, outcome_variable, confounders):
        """
        Inferência causal em ciências sociais
        """
        # Implementação simplificada de propensity score matching

        # Calcular propensity scores
        from sklearn.linear_model import LogisticRegression

        X = confounders
        y = treatment_variable

        ps_model = LogisticRegression()
        ps_model.fit(X, y)

        propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Matching
        treated_indices = np.where(treatment_variable == 1)[0]
        control_indices = np.where(treatment_variable == 0)[0]

        # Matching simples (mais próximo)
        matched_outcomes = []

        for treated_idx in treated_indices:
            treated_ps = propensity_scores[treated_idx]

            # Encontrar controle com PS mais próximo
            ps_differences = np.abs(propensity_scores[control_indices] - treated_ps)
            closest_control_idx = control_indices[np.argmin(ps_differences)]

            # Efeito causal
            causal_effect = outcome_variable.iloc[treated_idx] - outcome_variable.iloc[closest_control_idx]
            matched_outcomes.append(causal_effect)

        average_treatment_effect = np.mean(matched_outcomes)

        return {
            'propensity_scores': propensity_scores,
            'average_treatment_effect': average_treatment_effect,
            'matched_outcomes': matched_outcomes,
            'n_matched_pairs': len(matched_outcomes)
        }

    def social_recommendation_system(self, user_item_interactions, user_features):
        """
        Sistema de recomendação social
        """
        # Implementação simplificada de filtragem colaborativa

        # Matriz de interações usuário-item
        n_users = len(user_features)
        n_items = len(user_item_interactions.columns)

        # Fatorização de matriz
        from sklearn.decomposition import NMF

        interaction_matrix = user_item_interactions.values

        # Preencher valores ausentes
        interaction_matrix = np.nan_to_num(interaction_matrix, nan=0)

        # Fatorização não-negativa
        nmf = NMF(n_components=10, random_state=42)
        user_factors = nmf.fit_transform(interaction_matrix)
        item_factors = nmf.components_

        # Recomendações
        recommendations = np.dot(user_factors, item_factors)

        # Avaliação
        mse = mean_squared_error(interaction_matrix, recommendations)

        return {
            'user_factors': user_factors,
            'item_factors': item_factors,
            'recommendations': recommendations,
            'reconstruction_error': mse
        }
```

**Aprendizado de Máquina Social:**
- Predição de comportamento social
- Incorporação de redes sociais
- Mineração de opinião
- Predição de impacto social
- Inferência causal
- Sistemas de recomendação social

---

## 4. CONSIDERAÇÕES FINAIS

A sociologia computacional representa a convergência entre teoria sociológica e métodos computacionais, permitindo explorar padrões sociais complexos que transcendem a observação tradicional. Os modelos apresentados fornecem ferramentas para:

1. **Análise de Redes**: Compreensão de estruturas sociais e dinâmicas
2. **Modelagem Baseada em Agentes**: Simulação de sistemas sociais complexos
3. **Aprendizado de Máquina**: Predição e análise de comportamento social
4. **Inferência Causal**: Compreensão de relações causais em contextos sociais
5. **Aplicações Práticas**: Intervenções sociais baseadas em evidências

**Próximos Passos Recomendados**:
1. Dominar análise de redes sociais e teoria dos grafos
2. Desenvolver proficiência em modelagem baseada em agentes
3. Explorar aplicações de aprendizado de máquina em dados sociais
4. Integrar métodos computacionais com teoria sociológica
5. Contribuir para o avanço da sociologia computacional aplicada

---

*Documento preparado para fine-tuning de IA em Sociologia Computacional*
*Versão 1.0 - Preparado para implementação prática*
