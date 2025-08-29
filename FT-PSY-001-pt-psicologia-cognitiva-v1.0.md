# FT-PSY-001: Fine-Tuning para IA em Psicologia Cognitiva

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados em psicologia cognitiva, integrando modelagem computacional de processos mentais, memória, aprendizagem, percepção e tomada de decisão com princípios da ciência cognitiva.

### Contexto Filosófico
A psicologia cognitiva representa a ponte entre a mente e a máquina, buscando compreender como processos mentais podem ser modelados computacionalmente. Esta abordagem reconhece que a cognição é um processo computacional que pode ser simulado e compreendido através de algoritmos formais.

### Metodologia de Aprendizado Recomendada
1. **Fundamentos Cognitivos**: Compreensão de processos mentais básicos
2. **Modelagem Computacional**: Desenvolvimento de modelos formais de cognição
3. **Integração Multinível**: Conexão entre neurônio, sistema e comportamento
4. **Validação Experimental**: Comparação com dados empíricos humanos
5. **Aplicações Práticas**: Implementação em interfaces cérebro-computador e educação

---

## 1. MODELOS COMPUTACIONAIS DA MENTE

### 1.1 Arquitetura Cognitiva
```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class CognitiveArchitecture:
    """
    Modelos de arquitetura cognitiva (ACT-R, SOAR, etc.)
    """

    def __init__(self, working_memory_capacity=7, long_term_memory_size=1000):
        self.working_memory = []
        self.long_term_memory = {}
        self.working_memory_capacity = working_memory_capacity
        self.attention_focus = None

        # Parâmetros de arquitetura
        self.decay_rate = 0.1
        self.activation_threshold = 0.5
        self.learning_rate = 0.1

    def act_r_model(self, task_stimuli, time_steps=100):
        """
        Modelo ACT-R (Adaptive Control of Thought-Rational)
        """
        # Estados do sistema
        goal_stack = []
        declarative_memory = {}
        procedural_memory = {}

        # Buffer de produção
        production_buffer = None

        # Simulação
        cognitive_states = []

        for t in range(time_steps):
            # Percepção e codificação
            current_stimuli = task_stimuli[t] if t < len(task_stimuli) else None

            # Recuperação de memória declarativa
            if current_stimuli:
                retrieved_info = self._retrieve_from_memory(current_stimuli, declarative_memory)

                # Adicionar à memória de trabalho
                if retrieved_info:
                    self._add_to_working_memory(retrieved_info)

            # Processo de produção
            applicable_productions = self._match_productions(procedural_memory, goal_stack)

            if applicable_productions:
                selected_production = self._select_production(applicable_productions)
                production_buffer = selected_production

                # Aplicar produção
                self._apply_production(selected_production, goal_stack, declarative_memory)

            # Aprendizado
            if production_buffer:
                self._reinforce_production(production_buffer, procedural_memory)

            # Estado cognitivo atual
            cognitive_state = {
                'time': t,
                'stimuli': current_stimuli,
                'goal_stack': goal_stack.copy(),
                'working_memory': self.working_memory.copy(),
                'production_buffer': production_buffer,
                'memory_activation': self._calculate_memory_activation(declarative_memory)
            }

            cognitive_states.append(cognitive_state)

        return cognitive_states

    def _retrieve_from_memory(self, cue, memory):
        """Recuperação de memória declarativa"""
        if cue in memory:
            activation = memory[cue]['activation']
            if activation > self.activation_threshold:
                return memory[cue]['content']
        return None

    def _add_to_working_memory(self, item):
        """Adicionar item à memória de trabalho"""
        if len(self.working_memory) >= self.working_memory_capacity:
            # Remover item menos recente
            self.working_memory.pop(0)

        self.working_memory.append(item)

    def _match_productions(self, procedural_memory, goal_stack):
        """Encontrar produções aplicáveis"""
        applicable = []

        for production in procedural_memory.values():
            if self._matches_conditions(production['conditions'], goal_stack):
                applicable.append(production)

        return applicable

    def _select_production(self, productions):
        """Selecionar produção baseada em utilidade"""
        utilities = [prod['utility'] for prod in productions]
        selected_idx = np.argmax(utilities)
        return productions[selected_idx]

    def _apply_production(self, production, goal_stack, declarative_memory):
        """Aplicar produção selecionada"""
        # Ações da produção
        for action in production['actions']:
            if action['type'] == 'add_goal':
                goal_stack.append(action['goal'])
            elif action['type'] == 'remove_goal':
                if action['goal'] in goal_stack:
                    goal_stack.remove(action['goal'])
            elif action['type'] == 'store_memory':
                declarative_memory[action['key']] = action['value']

    def _reinforce_production(self, production, procedural_memory):
        """Reforçar produção usada"""
        key = production['name']
        procedural_memory[key]['utility'] += self.learning_rate

    def _calculate_memory_activation(self, memory):
        """Calcular ativação geral da memória"""
        if not memory:
            return 0

        activations = [item['activation'] for item in memory.values()]
        return np.mean(activations)

    def soar_architecture(self, problem_space, initial_state):
        """
        Arquitetura SOAR (State, Operator, And Result)
        """
        class SOARSystem:
            def __init__(self, problem_space, initial_state):
                self.problem_space = problem_space
                self.current_state = initial_state
                self.goal_stack = [initial_state]
                self.working_memory = {}

            def problem_solving_cycle(self, max_cycles=50):
                """Ciclo de resolução de problemas"""
                solution_path = [self.current_state]

                for cycle in range(max_cycles):
                    # Avaliação do estado atual
                    if self._is_goal_state(self.current_state):
                        break

                    # Geração de operadores
                    applicable_operators = self._generate_operators(self.current_state)

                    # Seleção de operador
                    if applicable_operators:
                        selected_operator = self._select_operator(applicable_operators)

                        # Aplicação do operador
                        new_state = self._apply_operator(selected_operator, self.current_state)

                        # Aprendizado (chunking)
                        self._learn_chunk(self.current_state, selected_operator, new_state)

                        # Atualização de estado
                        self.current_state = new_state
                        solution_path.append(new_state)
                    else:
                        # Impasse - subir na pilha de metas
                        self._handle_impasse()

                return solution_path

            def _is_goal_state(self, state):
                """Verificar se estado é meta"""
                return state in self.problem_space.get('goal_states', [])

            def _generate_operators(self, state):
                """Gerar operadores aplicáveis"""
                all_operators = self.problem_space.get('operators', [])
                applicable = []

                for operator in all_operators:
                    if self._operator_applicable(operator, state):
                        applicable.append(operator)

                return applicable

            def _operator_applicable(self, operator, state):
                """Verificar se operador é aplicável"""
                preconditions = operator.get('preconditions', [])
                return all(precond in state for precond in preconditions)

            def _select_operator(self, operators):
                """Selecionar operador baseado em preferências"""
                # Seleção simples: primeiro operador
                return operators[0]

            def _apply_operator(self, operator, state):
                """Aplicar operador ao estado"""
                new_state = state.copy()

                # Remover pré-condições
                for precond in operator.get('delete_list', []):
                    if precond in new_state:
                        new_state.remove(precond)

                # Adicionar pós-condições
                for postcond in operator.get('add_list', []):
                    if postcond not in new_state:
                        new_state.append(postcond)

                return new_state

            def _learn_chunk(self, state, operator, result):
                """Aprender novo chunk (regra de produção)"""
                chunk_name = f"chunk_{len(self.working_memory)}"

                self.working_memory[chunk_name] = {
                    'conditions': state + [operator['name']],
                    'actions': result,
                    'strength': 1.0
                }

            def _handle_impasse(self):
                """Lidar com impasse"""
                # Estratégia simples: adicionar meta de subproblema
                subproblem = f"subproblem_{len(self.goal_stack)}"
                self.goal_stack.append(subproblem)

        soar_system = SOARSystem(problem_space, initial_state)
        solution = soar_system.problem_solving_cycle()

        return {
            'solution_path': solution,
            'chunks_learned': len(soar_system.working_memory),
            'goal_stack_depth': len(soar_system.goal_stack)
        }

    def connectionist_networks(self, input_patterns, target_patterns, learning_rate=0.1):
        """
        Redes conexionistas para modelagem cognitiva
        """
        # Rede neural simples
        n_inputs = len(input_patterns[0])
        n_hidden = 10
        n_outputs = len(target_patterns[0])

        # Pesos
        W1 = np.random.randn(n_inputs, n_hidden) * 0.1
        W2 = np.random.randn(n_hidden, n_outputs) * 0.1

        # Treinamento
        n_epochs = 100
        errors = []

        for epoch in range(n_epochs):
            total_error = 0

            for input_pattern, target_pattern in zip(input_patterns, target_patterns):
                # Forward pass
                hidden_activation = self._sigmoid(np.dot(input_pattern, W1))
                output_activation = self._sigmoid(np.dot(hidden_activation, W2))

                # Erro
                error = target_pattern - output_activation
                total_error += np.sum(error**2)

                # Backward pass
                output_delta = error * self._sigmoid_derivative(output_activation)
                hidden_delta = np.dot(output_delta, W2.T) * self._sigmoid_derivative(hidden_activation)

                # Atualização de pesos
                W2 += learning_rate * np.outer(hidden_activation, output_delta)
                W1 += learning_rate * np.outer(input_pattern, hidden_delta)

            errors.append(total_error / len(input_patterns))

        return {
            'final_weights_input_hidden': W1,
            'final_weights_hidden_output': W2,
            'training_errors': errors,
            'final_error': errors[-1]
        }

    def _sigmoid(self, x):
        """Função sigmoide"""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivada da sigmoide"""
        return x * (1 - x)
```

**Arquiteturas Cognitivas:**
- Modelo ACT-R para cognição unificada
- Arquitetura SOAR para resolução de problemas
- Redes conexionistas para aprendizado associativo

### 1.2 Memória e Aprendizado
```python
import numpy as np
from collections import defaultdict
import heapq

class MemorySystems:
    """
    Modelos de sistemas de memória (declarativa, procedural, working memory)
    """

    def __init__(self, memory_capacity=100):
        self.declarative_memory = {}  # Memória declarativa (fatos)
        self.procedural_memory = {}   # Memória procedural (habilidades)
        self.working_memory = []      # Memória de trabalho
        self.capacity = memory_capacity

        # Parâmetros de esquecimento
        self.decay_rate = 0.1
        self.activation_threshold = 0.3

    def atkinson_shiffrin_model(self, stimuli_sequence, time_steps=100):
        """
        Modelo de memória de Atkinson-Shiffrin (multistore)
        """
        # Registros de memória
        sensory_store = []
        short_term_store = []
        long_term_store = []

        memory_trace = {
            'sensory': [],
            'short_term': [],
            'long_term': []
        }

        for t in range(time_steps):
            # Entrada sensorial
            if t < len(stimuli_sequence):
                current_stimulus = stimuli_sequence[t]
                sensory_store.append({
                    'stimulus': current_stimulus,
                    'activation': 1.0,
                    'timestamp': t
                })

            # Decaimento sensorial
            sensory_store = [item for item in sensory_store
                           if item['activation'] > self.activation_threshold]

            for item in sensory_store:
                item['activation'] *= (1 - self.decay_rate)

            # Transferência para memória curta
            if sensory_store and len(short_term_store) < 7:  # Capacidade limitada
                transferred_item = sensory_store.pop(0)
                short_term_store.append({
                    'stimulus': transferred_item['stimulus'],
                    'activation': 1.0,
                    'rehearsals': 1,
                    'timestamp': t
                })

            # Rehearsal e decrescimento na memória curta
            for item in short_term_store:
                if np.random.random() < 0.1:  # Probabilidade de rehearsal
                    item['rehearsals'] += 1
                    item['activation'] = min(1.0, item['activation'] + 0.1)

                # Decaimento baseado em tempo e rehearsals
                time_since_creation = t - item['timestamp']
                decay_factor = np.exp(-time_since_creation * self.decay_rate / item['rehearsals'])
                item['activation'] *= decay_factor

            # Remover itens com baixa ativação da memória curta
            short_term_store = [item for item in short_term_store
                              if item['activation'] > self.activation_threshold]

            # Transferência para memória longa
            for item in short_term_store:
                if item['rehearsals'] > 5 and np.random.random() < 0.05:
                    long_term_store.append({
                        'stimulus': item['stimulus'],
                        'strength': item['activation'],
                        'consolidation_time': t
                    })
                    short_term_store.remove(item)

            # Registrar estado
            memory_trace['sensory'].append(len(sensory_store))
            memory_trace['short_term'].append(len(short_term_store))
            memory_trace['long_term'].append(len(long_term_store))

        return {
            'memory_trace': memory_trace,
            'final_sensory_items': len(sensory_store),
            'final_short_term_items': len(short_term_store),
            'final_long_term_items': len(long_term_store)
        }

    def connectionist_memory_model(self, input_patterns, n_epochs=50):
        """
        Modelo conexionista de memória (Hopfield network)
        """
        class HopfieldNetwork:
            def __init__(self, n_neurons):
                self.n_neurons = n_neurons
                self.weights = np.zeros((n_neurons, n_neurons))

            def train(self, patterns):
                """Treinar rede com padrões"""
                for pattern in patterns:
                    pattern = np.array(pattern)
                    self.weights += np.outer(pattern, pattern)

                # Diagonal zero
                np.fill_diagonal(self.weights, 0)

                # Normalizar
                self.weights /= len(patterns)

            def recall(self, probe_pattern, max_iterations=10):
                """Recuperar padrão"""
                state = np.array(probe_pattern.copy())

                for iteration in range(max_iterations):
                    for i in range(self.n_neurons):
                        # Atualização assíncrona
                        net_input = np.dot(self.weights[i], state)
                        state[i] = 1 if net_input > 0 else -1

                    # Verificar convergência
                    if iteration > 0 and np.array_equal(state, previous_state):
                        break

                    previous_state = state.copy()

                return state

            def energy_function(self, state):
                """Função de energia da rede"""
                return -0.5 * np.sum(self.weights * np.outer(state, state))

        # Treinar rede Hopfield
        n_neurons = len(input_patterns[0])
        hopfield_net = HopfieldNetwork(n_neurons)
        hopfield_net.train(input_patterns)

        # Testar recuperação
        recall_results = []

        for i, pattern in enumerate(input_patterns):
            # Adicionar ruído
            noisy_pattern = pattern.copy()
            noise_indices = np.random.choice(n_neurons, size=int(0.1 * n_neurons), replace=False)
            noisy_pattern[noise_indices] *= -1

            # Recuperar
            recalled_pattern = hopfield_net.recall(noisy_pattern)

            # Calcular acurácia
            accuracy = np.mean(recalled_pattern == pattern)

            recall_results.append({
                'original_pattern': i,
                'recalled_pattern': recalled_pattern,
                'accuracy': accuracy
            })

        return {
            'hopfield_network': hopfield_net,
            'recall_results': recall_results,
            'average_recall_accuracy': np.mean([r['accuracy'] for r in recall_results])
        }

    def spaced_repetition_model(self, items_to_learn, review_schedule):
        """
        Modelo de repetição espaçada para aprendizado
        """
        class SpacedRepetition:
            def __init__(self, items):
                self.items = {item: {'strength': 0, 'last_review': 0, 'reviews': []}
                            for item in items}

            def review_item(self, item, current_time, performance):
                """Revisar item e atualizar força"""
                if item in self.items:
                    item_data = self.items[item]

                    # Atualizar força baseada na performance
                    if performance > 0.8:  # Boa performance
                        strength_increase = 1.0
                    elif performance > 0.5:  # Performance média
                        strength_increase = 0.5
                    else:  # Má performance
                        strength_increase = 0.1

                    item_data['strength'] += strength_increase
                    item_data['last_review'] = current_time
                    item_data['reviews'].append({
                        'time': current_time,
                        'performance': performance,
                        'strength_after': item_data['strength']
                    })

            def calculate_optimal_interval(self, item):
                """Calcular intervalo ótimo de revisão"""
                strength = self.items[item]['strength']
                # Fórmula simplificada de Leitner
                optimal_interval = 2 ** strength  # Dias

                return optimal_interval

            def get_items_due_review(self, current_time):
                """Obter itens que precisam de revisão"""
                due_items = []

                for item, data in self.items.items():
                    time_since_last = current_time - data['last_review']
                    optimal_interval = self.calculate_optimal_interval(item)

                    if time_since_last >= optimal_interval:
                        due_items.append({
                            'item': item,
                            'days_overdue': time_since_last - optimal_interval,
                            'strength': data['strength']
                        })

                return due_items

        spaced_rep = SpacedRepetition(items_to_learn)

        # Simulação de aprendizado
        learning_progress = []

        for time, schedule in enumerate(review_schedule):
            for item in schedule['items_to_review']:
                performance = np.random.beta(2, 2)  # Performance simulada
                spaced_rep.review_item(item, time, performance)

            # Progresso atual
            avg_strength = np.mean([data['strength'] for data in spaced_rep.items.values()])
            learning_progress.append({
                'time': time,
                'avg_strength': avg_strength,
                'items_reviewed': len(schedule['items_to_review'])
            })

        return {
            'spaced_repetition_system': spaced_rep,
            'learning_progress': learning_progress,
            'final_avg_strength': learning_progress[-1]['avg_strength']
        }

    def chunking_mechanism(self, information_stream, chunk_size=4):
        """
        Mecanismo de agrupamento (chunking) para memória
        """
        class ChunkingSystem:
            def __init__(self, chunk_size):
                self.chunk_size = chunk_size
                self.chunks = []
                self.chunk_hierarchy = {}

            def process_stream(self, stream):
                """Processar fluxo de informação"""
                chunks_created = []

                for i in range(0, len(stream), self.chunk_size):
                    chunk = stream[i:i + self.chunk_size]

                    # Verificar se chunk já existe
                    existing_chunk = self._find_existing_chunk(chunk)

                    if existing_chunk:
                        # Reforçar chunk existente
                        existing_chunk['frequency'] += 1
                        chunks_created.append(f"existing_{existing_chunk['id']}")
                    else:
                        # Criar novo chunk
                        new_chunk = {
                            'id': len(self.chunks),
                            'elements': chunk,
                            'frequency': 1,
                            'created_at': len(chunks_created)
                        }
                        self.chunks.append(new_chunk)
                        chunks_created.append(f"new_{new_chunk['id']}")

                return chunks_created

            def _find_existing_chunk(self, elements):
                """Encontrar chunk existente com elementos similares"""
                for chunk in self.chunks:
                    if self._similarity(chunk['elements'], elements) > 0.8:
                        return chunk
                return None

            def _similarity(self, chunk1, chunk2):
                """Calcular similaridade entre chunks"""
                if len(chunk1) != len(chunk2):
                    return 0

                matches = sum(1 for a, b in zip(chunk1, chunk2) if a == b)
                return matches / len(chunk1)

            def build_hierarchy(self):
                """Construir hierarquia de chunks"""
                # Agrupar chunks similares em super-chunks
                super_chunks = {}

                for i, chunk1 in enumerate(self.chunks):
                    similar_chunks = []

                    for j, chunk2 in enumerate(self.chunks):
                        if i != j and self._similarity(chunk1['elements'], chunk2['elements']) > 0.6:
                            similar_chunks.append(j)

                    if len(similar_chunks) > 1:
                        super_chunks[f"super_{i}"] = {
                            'sub_chunks': [i] + similar_chunks,
                            'elements': chunk1['elements']  # Representante
                        }

                self.chunk_hierarchy = super_chunks

                return super_chunks

        chunking_system = ChunkingSystem(chunk_size)
        chunks_created = chunking_system.process_stream(information_stream)

        # Construir hierarquia após processamento
        hierarchy = chunking_system.build_hierarchy()

        return {
            'chunks_created': chunks_created,
            'total_chunks': len(chunking_system.chunks),
            'chunk_hierarchy': hierarchy,
            'compression_ratio': len(information_stream) / len(chunks_created)
        }
```

**Sistemas de Memória:**
- Modelo Atkinson-Shiffrin (multistore)
- Redes Hopfield para memória conexionista
- Repetição espaçada otimizada
- Mecanismos de agrupamento (chunking)

### 1.3 Percepção e Atenção
```python
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

class PerceptionAttention:
    """
    Modelos de percepção e atenção cognitiva
    """

    def __init__(self):
        self.attention_filters = {}
        self.perceptual_buffers = {}

    def feature_integration_theory(self, visual_features, binding_time=100):
        """
        Teoria da Integração de Características (Treisman)
        """
        class FeatureIntegration:
            def __init__(self, features):
                self.features = features
                self.master_map = {}
                self.feature_maps = {}

            def parallel_processing_stage(self):
                """Processamento paralelo de características"""
                feature_maps = {}

                for feature_type, feature_data in self.features.items():
                    # Mapa de características (simplificado)
                    feature_map = np.zeros_like(feature_data)

                    # Detectar características específicas
                    if feature_type == 'color':
                        # Detectar mudanças de cor
                        feature_map = np.abs(np.gradient(feature_data, axis=0))
                    elif feature_type == 'orientation':
                        # Detectar orientações usando filtros Gabor
                        feature_map = self._apply_orientation_filters(feature_data)
                    elif feature_type == 'motion':
                        # Detectar movimento
                        feature_map = np.abs(np.gradient(feature_data, axis=2))  # Tempo

                    feature_maps[feature_type] = feature_map

                self.feature_maps = feature_maps
                return feature_maps

            def focused_attention_stage(self, attended_location):
                """Estágio de atenção focada"""
                # Integrar características no local atendido
                integrated_object = {}

                for feature_type, feature_map in self.feature_maps.items():
                    # Extrair valor no local de atenção
                    if len(feature_map.shape) >= 2:
                        x, y = attended_location
                        if x < feature_map.shape[0] and y < feature_map.shape[1]:
                            integrated_object[feature_type] = feature_map[x, y]
                        else:
                            integrated_object[feature_type] = 0
                    else:
                        integrated_object[feature_type] = np.mean(feature_map)

                # Criar representação integrada
                object_representation = {
                    'features': integrated_object,
                    'location': attended_location,
                    'binding_strength': np.mean(list(integrated_object.values())),
                    'processing_time': binding_time
                }

                return object_representation

            def _apply_orientation_filters(self, image):
                """Aplicar filtros de orientação"""
                orientations = [0, 45, 90, 135]
                orientation_responses = []

                for theta in orientations:
                    # Filtro simples de orientação
                    kernel = np.array([
                        [-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]
                    ]) * np.cos(theta)

                    response = correlate2d(image, kernel, mode='same')
                    orientation_responses.append(response)

                # Máxima resposta de orientação
                return np.max(orientation_responses, axis=0)

        feature_integration = FeatureIntegration(visual_features)

        # Processamento paralelo
        feature_maps = feature_integration.parallel_processing_stage()

        # Atenção focada em múltiplos locais
        attended_locations = [(10, 10), (20, 20), (30, 30)]  # Locais simulados
        integrated_objects = []

        for location in attended_locations:
            obj = feature_integration.focused_attention_stage(location)
            integrated_objects.append(obj)

        return {
            'feature_maps': feature_maps,
            'integrated_objects': integrated_objects,
            'attention_locations': attended_locations,
            'binding_time': binding_time
        }

    def biased_competition_model(self, stimuli, attention_bias=0.7):
        """
        Modelo de Competição Enviesada para atenção
        """
        class BiasedCompetition:
            def __init__(self, stimuli, bias):
                self.stimuli = stimuli
                self.bias = bias
                self.neural_activities = {}

            def competition_dynamics(self, time_steps=50):
                """Dinâmica de competição neural"""
                n_stimuli = len(self.stimuli)
                activities = np.ones(n_stimuli) * 0.5  # Atividade inicial

                activity_history = [activities.copy()]

                for t in range(time_steps):
                    new_activities = activities.copy()

                    for i in range(n_stimuli):
                        # Entrada do estímulo
                        stimulus_input = self.stimuli[i]['strength']

                        # Competição inibitória
                        inhibition = np.sum([activities[j] for j in range(n_stimuli) if j != i])

                        # Viés de atenção
                        attention_modulation = 1 + self.bias if self.stimuli[i]['attended'] else 1

                        # Equação de atividade
                        tau = 10  # Constante de tempo
                        excitatory_input = stimulus_input * attention_modulation
                        inhibitory_input = 0.1 * inhibition

                        d_activity_dt = (-activities[i] + excitatory_input - inhibitory_input) / tau
                        new_activities[i] += d_activity_dt

                        # Limites
                        new_activities[i] = np.clip(new_activities[i], 0, 1)

                    activities = new_activities
                    activity_history.append(activities.copy())

                # Estímulo vencedor
                winner_idx = np.argmax(activities)

                return {
                    'activity_history': activity_history,
                    'final_activities': activities,
                    'winner_stimulus': winner_idx,
                    'competition_strength': np.std(activities),
                    'attention_bias_effect': self.bias
                }

        biased_competition = BiasedCompetition(stimuli, attention_bias)
        competition_results = biased_competition.competition_dynamics()

        return competition_results

    def change_detection_theory(self, stimulus_sequence, detection_threshold=0.5):
        """
        Teoria da Detecção de Mudanças
        """
        class ChangeDetection:
            def __init__(self, sequence, threshold):
                self.sequence = sequence
                self.threshold = threshold
                self.memory_buffer = []

            def detect_changes(self):
                """Detectar mudanças na sequência"""
                changes_detected = []
                false_alarms = []
                misses = []

                current_memory = None

                for i, stimulus in enumerate(self.sequence):
                    # Atualizar memória
                    self.memory_buffer.append(stimulus)

                    if len(self.memory_buffer) > 5:  # Capacidade limitada
                        self.memory_buffer.pop(0)

                    if current_memory is None:
                        current_memory = stimulus
                        continue

                    # Calcular diferença
                    difference = self._calculate_difference(current_memory, stimulus)

                    # Decidir se houve mudança
                    if difference > self.threshold:
                        if self._is_actual_change(i):  # Verificar se mudança real
                            changes_detected.append(i)
                        else:
                            false_alarms.append(i)

                        current_memory = stimulus
                    else:
                        if self._is_actual_change(i):
                            misses.append(i)

                return {
                    'changes_detected': changes_detected,
                    'false_alarms': false_alarms,
                    'misses': misses,
                    'detection_accuracy': len(changes_detected) / (len(changes_detected) + len(misses)) if (len(changes_detected) + len(misses)) > 0 else 0,
                    'false_alarm_rate': len(false_alarms) / len(self.sequence)
                }

            def _calculate_difference(self, memory, current):
                """Calcular diferença entre estímulos"""
                # Diferença simples (pode ser mais sofisticada)
                if isinstance(memory, dict) and isinstance(current, dict):
                    # Diferença baseada em características
                    diff = 0
                    for key in memory.keys():
                        if key in current:
                            diff += abs(memory[key] - current[key])
                    return diff
                else:
                    return abs(memory - current)

            def _is_actual_change(self, index):
                """Verificar se houve mudança real (simulado)"""
                # Simulação: mudança a cada 10 passos
                return index % 10 == 0

        change_detector = ChangeDetection(stimulus_sequence, detection_threshold)
        detection_results = change_detector.detect_changes()

        return detection_results

    def visual_search_model(self, search_array, target_features, search_type='feature'):
        """
        Modelo de Busca Visual (Feature Integration vs Guided Search)
        """
        class VisualSearch:
            def __init__(self, array, target, search_type):
                self.search_array = array
                self.target_features = target
                self.search_type = search_type

            def perform_search(self):
                """Executar busca visual"""
                if self.search_type == 'feature':
                    # Busca por características (paralela)
                    return self._feature_search()
                elif self.search_type == 'conjunction':
                    # Busca por conjunção (serial)
                    return self._conjunction_search()
                else:
                    raise ValueError("Tipo de busca não suportado")

            def _feature_search(self):
                """Busca por característica simples"""
                search_times = []
                accuracies = []

                for item in self.search_array:
                    # Tempo de busca baseado na saliência
                    saliency = self._calculate_saliency(item, self.target_features)
                    search_time = 50 + (1 - saliency) * 200  # ms

                    # Acurácia baseada na saliência
                    accuracy = min(0.95, saliency + 0.5)

                    search_times.append(search_time)
                    accuracies.append(accuracy)

                return {
                    'search_times': search_times,
                    'accuracies': accuracies,
                    'average_search_time': np.mean(search_times),
                    'overall_accuracy': np.mean(accuracies)
                }

            def _conjunction_search(self):
                """Busca por conjunção de características"""
                search_times = []
                accuracies = []

                for i, item in enumerate(self.search_array):
                    # Busca serial: verificar item por item
                    time_per_item = 50  # ms por item

                    # Probabilidade de encontrar alvo
                    target_match = self._matches_target(item, self.target_features)
                    accuracy = 0.9 if target_match else 0.1

                    # Tempo até encontrar (ou não)
                    if target_match:
                        search_time = (i + 1) * time_per_item
                    else:
                        search_time = len(self.search_array) * time_per_item

                    search_times.append(search_time)
                    accuracies.append(accuracy)

                return {
                    'search_times': search_times,
                    'accuracies': accuracies,
                    'average_search_time': np.mean(search_times),
                    'overall_accuracy': np.mean(accuracies)
                }

            def _calculate_saliency(self, item, target_features):
                """Calcular saliência do item"""
                saliency = 0

                for feature, target_value in target_features.items():
                    if feature in item:
                        diff = abs(item[feature] - target_value)
                        saliency += 1 / (1 + diff)  # Saliência inversamente proporcional à diferença

                return saliency / len(target_features)

            def _matches_target(self, item, target_features):
                """Verificar se item combina com alvo"""
                matches = 0

                for feature, target_value in target_features.items():
                    if feature in item and abs(item[feature] - target_value) < 0.1:
                        matches += 1

                return matches == len(target_features)

        visual_search = VisualSearch(search_array, target_features, search_type)
        search_results = visual_search.perform_search()

        return search_results

    def predictive_coding_attention(self, sensory_input, prediction_error_threshold=0.3):
        """
        Modelo de Atenção por Codificação Preditiva
        """
        class PredictiveCoding:
            def __init__(self, input_data, threshold):
                self.input = input_data
                self.threshold = threshold
                self.prediction_errors = []
                self.attention_allocation = []

            def predictive_processing(self, prediction_model):
                """Processamento preditivo"""
                predictions = []
                errors = []

                for i, sensory_data in enumerate(self.input):
                    # Fazer previsão
                    if i == 0:
                        prediction = np.mean(self.input[:5]) if len(self.input) > 5 else 0.5
                    else:
                        prediction = prediction_model.predict_next(sensory_data)

                    # Calcular erro de previsão
                    prediction_error = abs(sensory_data - prediction)
                    errors.append(prediction_error)

                    # Alocar atenção baseada no erro
                    if prediction_error > self.threshold:
                        attention = 1.0  # Atenção máxima
                    else:
                        attention = prediction_error / self.threshold  # Atenção proporcional

                    self.attention_allocation.append(attention)

                    # Atualizar modelo com erro
                    prediction_model.update(sensory_data, prediction_error)

                return {
                    'prediction_errors': errors,
                    'attention_allocation': self.attention_allocation,
                    'total_attention': np.sum(self.attention_allocation),
                    'attention_efficiency': np.mean(self.attention_allocation)
                }

        # Modelo de previsão simples
        class SimplePredictionModel:
            def __init__(self):
                self.memory = []

            def predict_next(self, current_input):
                """Previsão simples baseada na média"""
                if self.memory:
                    return np.mean(self.memory[-5:])  # Média dos últimos 5
                else:
                    return current_input

            def update(self, new_input, error):
                """Atualizar memória"""
                self.memory.append(new_input)
                if len(self.memory) > 10:
                    self.memory.pop(0)

        prediction_model = SimplePredictionModel()
        predictive_system = PredictiveCoding(sensory_input, prediction_error_threshold)
        processing_results = predictive_system.predictive_processing(prediction_model)

        return processing_results
```

**Percepção e Atenção:**
- Teoria da Integração de Características
- Modelo de Competição Enviesada
- Detecção de mudanças visuais
- Busca visual (feature vs conjunction)
- Atenção por codificação preditiva

---

## 2. APRENDIZADO E TOMADA DE DECISÃO

### 2.1 Teorias da Aprendizagem
```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class LearningDecisionTheories:
    """
    Teorias de aprendizagem e tomada de decisão cognitiva
    """

    def __init__(self):
        self.learning_models = {}

    def reinforcement_learning_cognitive(self, task_environment, learning_rate=0.1):
        """
        Aprendizado por reforço em contextos cognitivos
        """
        class CognitiveRLAgent:
            def __init__(self, n_states, n_actions, lr):
                self.n_states = n_states
                self.n_actions = n_actions
                self.lr = lr
                self.q_table = np.zeros((n_states, n_actions))
                self.state_visits = np.zeros(n_states)

            def cognitive_state_representation(self, raw_state):
                """Representação cognitiva do estado"""
                # Abstração do estado bruto
                if isinstance(raw_state, dict):
                    # Codificar características relevantes
                    cognitive_features = []
                    for key, value in raw_state.items():
                        if key in ['attention', 'memory_load', 'emotional_state']:
                            cognitive_features.append(value)
                        else:
                            cognitive_features.append(hash(str(value)) % 10 / 10)

                    return np.mean(cognitive_features)
                else:
                    return raw_state

            def metacognitive_monitoring(self, state, action):
                """Monitoramento metacognitivo"""
                # Avaliar confiança na decisão
                state_visits = self.state_visits[state]

                if state_visits < 5:
                    confidence = 0.5  # Pouca experiência
                else:
                    q_values = self.q_table[state]
                    best_q = np.max(q_values)
                    second_best_q = np.partition(q_values, -2)[-2]
                    confidence = (best_q - second_best_q) / (np.std(q_values) + 1e-6)

                return confidence

            def learn_from_experience(self, state, action, reward, next_state, confidence):
                """Aprendizagem com consideração metacognitiva"""
                # Atualização Q-learning com confiança
                current_q = self.q_table[state, action]

                # Ajustar taxa de aprendizado baseada na confiança
                adjusted_lr = self.lr * (1 + confidence)

                # Atualização
                best_next_q = np.max(self.q_table[next_state])
                td_target = reward + 0.9 * best_next_q
                td_error = td_target - current_q

                self.q_table[state, action] += adjusted_lr * td_error
                self.state_visits[state] += 1

                return td_error

        # Simulação de aprendizado cognitivo
        n_states, n_actions = 10, 4
        agent = CognitiveRLAgent(n_states, n_actions, learning_rate)

        learning_history = []

        for episode in range(50):
            state = np.random.randint(n_states)

            for step in range(20):
                # Representação cognitiva
                cognitive_state = agent.cognitive_state_representation({'attention': np.random.random(), 'memory_load': step/20})

                action = np.argmax(agent.q_table[state])

                # Monitoramento metacognitivo
                confidence = agent.metacognitive_monitoring(state, action)

                # Recompensa baseada na performance cognitiva
                reward = np.random.normal(0, 1) + confidence

                next_state = np.random.randint(n_states)

                # Aprendizagem
                td_error = agent.learn_from_experience(state, action, reward, next_state, confidence)

                state = next_state

            learning_history.append({
                'episode': episode,
                'average_q': np.mean(agent.q_table),
                'total_visits': np.sum(agent.state_visits)
            })

        return {
            'cognitive_agent': agent,
            'learning_history': learning_history,
            'final_q_table': agent.q_table,
            'metacognitive_development': np.mean([h['average_q'] for h in learning_history[-10:]])
        }

    def dual_process_theory(self, problems, cognitive_capacity):
        """
        Teoria dos Processos Duplos (System 1 vs System 2)
        """
        class DualProcessSystem:
            def __init__(self, capacity):
                self.capacity = capacity
                self.system1_responses = {}
                self.system2_responses = {}

            def process_decision(self, problem):
                """Processar decisão usando sistemas duplos"""
                # Sistema 1: Rápido, automático, heurístico
                system1_response = self._system1_processing(problem)

                # Avaliar se resposta do Sistema 1 é adequada
                if self._needs_system2(problem, system1_response):
                    # Sistema 2: Lento, deliberativo, analítico
                    system2_response = self._system2_processing(problem, system1_response)
                    final_response = system2_response
                    system_used = 'system2'
                else:
                    final_response = system1_response
                    system_used = 'system1'

                return {
                    'final_decision': final_response,
                    'system_used': system_used,
                    'system1_response': system1_response,
                    'cognitive_effort': 'high' if system_used == 'system2' else 'low'
                }

            def _system1_processing(self, problem):
                """Processamento Sistema 1"""
                # Heurísticas simples
                if 'familiar' in problem.get('context', ''):
                    return 'trust_familiar'
                elif problem.get('time_pressure', False):
                    return 'quick_choice'
                else:
                    return 'default_heuristic'

            def _system2_processing(self, problem, system1_hint):
                """Processamento Sistema 2"""
                # Análise deliberativa
                options = problem.get('options', ['A', 'B'])
                criteria = problem.get('criteria', ['cost', 'benefit'])

                # Avaliação sistemática
                scores = {}

                for option in options:
                    score = 0
                    for criterion in criteria:
                        # Simulação de avaliação ponderada
                        weight = np.random.uniform(0.3, 0.7)
                        value = np.random.uniform(0, 1)
                        score += weight * value
                    scores[option] = score

                best_option = max(scores, key=scores.get)
                return best_option

            def _needs_system2(self, problem, system1_response):
                """Determinar se precisa do Sistema 2"""
                # Critérios para engajar Sistema 2
                needs_system2 = (
                    problem.get('complexity', 0) > 0.7 or
                    problem.get('importance', 0) > 0.8 or
                    problem.get('conflict', False) or
                    self.capacity < 0.5  # Capacidade cognitiva baixa
                )

                return needs_system2

        dual_system = DualProcessSystem(cognitive_capacity)

        decision_results = []
        for problem in problems:
            result = dual_system.process_decision(problem)
            decision_results.append(result)

        # Análise de uso dos sistemas
        system1_usage = np.mean([1 if r['system_used'] == 'system1' else 0 for r in decision_results])
        system2_usage = 1 - system1_usage

        return {
            'decision_results': decision_results,
            'system1_usage': system1_usage,
            'system2_usage': system2_usage,
            'cognitive_load': np.mean([1 if r['cognitive_effort'] == 'high' else 0 for r in decision_results])
        }

    def instance_based_learning(self, training_instances, query_instance):
        """
        Aprendizagem Baseada em Instâncias (Instance-Based Learning)
        """
        class InstanceBasedLearner:
            def __init__(self, instances):
                self.instances = instances
                self.memory = []

            def store_instance(self, instance, outcome):
                """Armazenar instância na memória"""
                self.memory.append({
                    'features': instance,
                    'outcome': outcome,
                    'retrieval_strength': 1.0,
                    'timestamp': len(self.memory)
                })

            def retrieve_similar_instances(self, query, k=5):
                """Recuperar instâncias similares"""
                similarities = []

                for instance in self.memory:
                    similarity = self._calculate_similarity(query, instance['features'])
                    similarities.append({
                        'instance': instance,
                        'similarity': similarity
                    })

                # Ordenar por similaridade
                similarities.sort(key=lambda x: x['similarity'], reverse=True)

                return similarities[:k]

            def make_prediction(self, query):
                """Fazer predição baseada em instâncias similares"""
                similar_instances = self.retrieve_similar_instances(query)

                if not similar_instances:
                    return None

                # Predição ponderada por similaridade
                weighted_outcomes = 0
                total_weight = 0

                for sim_instance in similar_instances:
                    weight = sim_instance['similarity']
                    outcome = sim_instance['instance']['outcome']

                    weighted_outcomes += weight * outcome
                    total_weight += weight

                if total_weight > 0:
                    prediction = weighted_outcomes / total_weight
                else:
                    prediction = np.mean([inst['instance']['outcome'] for inst in similar_instances])

                return prediction

            def _calculate_similarity(self, features1, features2):
                """Calcular similaridade entre características"""
                if len(features1) != len(features2):
                    return 0

                # Similaridade euclidiana normalizada
                distances = [abs(f1 - f2) for f1, f2 in zip(features1, features2)]
                max_distance = max(distances) if distances else 1

                similarity = 1 - (np.mean(distances) / max_distance)

                return similarity

        # Treinar aprendiz baseado em instâncias
        learner = InstanceBasedLearner(training_instances)

        # Armazenar instâncias de treinamento
        for instance in training_instances:
            learner.store_instance(instance['features'], instance['outcome'])

        # Fazer predição
        prediction = learner.make_prediction(query_instance)

        # Recuperar casos similares
        similar_cases = learner.retrieve_similar_instances(query_instance)

        return {
            'prediction': prediction,
            'similar_cases': similar_cases,
            'memory_size': len(learner.memory),
            'prediction_confidence': np.mean([case['similarity'] for case in similar_cases])
        }

    def category_learning_models(self, categories, exemplars, learning_rate=0.1):
        """
        Modelos de aprendizagem de categorias
        """
        class CategoryLearning:
            def __init__(self, categories, exemplars, lr):
                self.categories = categories
                self.exemplars = exemplars
                self.lr = lr
                self.category_prototypes = {}

                # Inicializar protótipos
                for category in self.categories:
                    category_exemplars = [ex for ex in self.exemplars if ex['category'] == category]
                    if category_exemplars:
                        prototype = np.mean([ex['features'] for ex in category_exemplars], axis=0)
                        self.category_prototypes[category] = prototype

            def prototype_model(self, test_exemplar):
                """Modelo de protótipo"""
                best_category = None
                min_distance = float('inf')

                for category, prototype in self.category_prototypes.items():
                    distance = np.linalg.norm(test_exemplar - prototype)
                    if distance < min_distance:
                        min_distance = distance
                        best_category = category

                return {
                    'predicted_category': best_category,
                    'distance_to_prototype': min_distance,
                    'prototype': self.category_prototypes[best_category]
                }

            def exemplar_model(self, test_exemplar):
                """Modelo de exemplar"""
                best_category = None
                max_similarity = 0

                category_similarities = {}

                for category in self.categories:
                    category_exemplars = [ex for ex in self.exemplars if ex['category'] == category]
                    similarities = []

                    for exemplar in category_exemplars:
                        similarity = self._exemplar_similarity(test_exemplar, exemplar['features'])
                        similarities.append(similarity)

                    avg_similarity = np.mean(similarities)
                    category_similarities[category] = avg_similarity

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        best_category = category

                return {
                    'predicted_category': best_category,
                    'category_similarities': category_similarities,
                    'max_similarity': max_similarity
                }

            def rule_based_model(self, test_exemplar, rules):
                """Modelo baseado em regras"""
                for rule in rules:
                    if self._satisfies_rule(test_exemplar, rule['conditions']):
                        return {
                            'predicted_category': rule['category'],
                            'rule_used': rule,
                            'satisfaction_score': self._rule_satisfaction_score(test_exemplar, rule['conditions'])
                        }

                return {
                    'predicted_category': None,
                    'rule_used': None,
                    'satisfaction_score': 0
                }

            def _exemplar_similarity(self, features1, features2):
                """Similaridade entre exemplares"""
                distance = np.linalg.norm(features1 - features2)
                return 1 / (1 + distance)  # Similaridade gaussiana

            def _satisfies_rule(self, exemplar, conditions):
                """Verificar se exemplar satisfaz condições da regra"""
                for condition in conditions:
                    feature_idx = condition['feature']
                    operator = condition['operator']
                    value = condition['value']

                    if operator == '>':
                        if not exemplar[feature_idx] > value:
                            return False
                    elif operator == '<':
                        if not exemplar[feature_idx] < value:
                            return False
                    elif operator == '==':
                        if not abs(exemplar[feature_idx] - value) < 0.1:
                            return False

                return True

            def _rule_satisfaction_score(self, exemplar, conditions):
                """Pontuação de satisfação da regra"""
                score = 0
                for condition in conditions:
                    feature_idx = condition['feature']
                    value = condition['value']
                    score += 1 / (1 + abs(exemplar[feature_idx] - value))

                return score / len(conditions)

        # Exemplos de regras
        rules = [
            {
                'category': 'A',
                'conditions': [
                    {'feature': 0, 'operator': '>', 'value': 0.5},
                    {'feature': 1, 'operator': '<', 'value': 0.3}
                ]
            },
            {
                'category': 'B',
                'conditions': [
                    {'feature': 0, 'operator': '<', 'value': 0.5},
                    {'feature': 1, 'operator': '>', 'value': 0.7}
                ]
            }
        ]

        category_learner = CategoryLearning(categories, exemplars, learning_rate)

        # Testar exemplar
        test_exemplar = np.random.rand(2)

        prototype_result = category_learner.prototype_model(test_exemplar)
        exemplar_result = category_learner.exemplar_model(test_exemplar)
        rule_result = category_learner.rule_based_model(test_exemplar, rules)

        return {
            'test_exemplar': test_exemplar,
            'prototype_model': prototype_result,
            'exemplar_model': exemplar_result,
            'rule_model': rule_result
        }
```

**Teorias da Aprendizagem:**
- Aprendizado por reforço metacognitivo
- Teoria dos processos duplos
- Aprendizagem baseada em instâncias
- Modelos de aprendizagem de categorias

### 2.2 Tomada de Decisão e Raciocínio
```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class DecisionReasoning:
    """
    Modelos de tomada de decisão e raciocínio cognitivo
    """

    def __init__(self):
        self.decision_models = {}

    def prospect_theory_decisions(self, options, reference_point=0):
        """
        Tomada de decisão usando Teoria do Prospecto
        """
        class ProspectTheoryDecision:
            def __init__(self, reference_point):
                self.reference_point = reference_point
                self.alpha = 0.88  # Curvatura para ganhos
                self.beta = 0.88   # Curvatura para perdas
                self.lambda_param = 2.25  # Aversão a perdas

            def evaluate_option(self, option):
                """Avaliar opção usando função de valor do prospecto"""
                outcome, probability = option['outcome'], option['probability']

                # Ganho/perda relativo ao ponto de referência
                gain_loss = outcome - self.reference_point

                # Função de valor
                if gain_loss >= 0:
                    value = gain_loss ** self.alpha
                else:
                    value = -self.lambda_param * (-gain_loss) ** self.beta

                # Utilidade esperada (com peso probabilístico)
                weighted_value = probability * value

                return weighted_value

            def choose_best_option(self, options):
                """Escolher melhor opção segundo Teoria do Prospecto"""
                option_values = []

                for i, option in enumerate(options):
                    value = self.evaluate_option(option)
                    option_values.append({
                        'option_index': i,
                        'value': value,
                        'outcome': option['outcome'],
                        'probability': option['probability']
                    })

                # Escolher opção com maior valor
                best_option = max(option_values, key=lambda x: x['value'])

                return {
                    'chosen_option': best_option,
                    'all_option_values': option_values,
                    'decision_rule': 'prospect_theory'
                }

        prospect_decision = ProspectTheoryDecision(reference_point)

        decision_result = prospect_decision.choose_best_option(options)

        return decision_result

    def bayesian_reasoning_model(self, prior_beliefs, evidence, likelihoods):
        """
        Raciocínio bayesiano para tomada de decisão
        """
        class BayesianReasoner:
            def __init__(self, priors, evidence_data, likelihood_data):
                self.priors = np.array(priors)
                self.evidence = evidence_data
                self.likelihoods = np.array(likelihoods)

            def update_beliefs(self):
                """Atualizar crenças usando Teorema de Bayes"""
                # P(H|E) = P(E|H) * P(H) / P(E)

                # Likelihood: P(E|H)
                likelihood = self.likelihoods

                # Prior: P(H)
                prior = self.priors

                # Evidence: P(E) = Σ P(E|H) * P(H)
                evidence = np.sum(likelihood * prior)

                # Posterior: P(H|E)
                posterior = (likelihood * prior) / evidence

                return {
                    'prior': prior,
                    'likelihood': likelihood,
                    'evidence': evidence,
                    'posterior': posterior,
                    'belief_change': posterior - prior
                }

            def decision_under_uncertainty(self, utilities):
                """Decisão sob incerteza usando valor esperado"""
                posterior = self.update_beliefs()['posterior']

                # Valor esperado para cada ação
                expected_utilities = []

                for action_utility in utilities:
                    expected_utility = np.sum(posterior * action_utility)
                    expected_utilities.append(expected_utility)

                best_action = np.argmax(expected_utilities)

                return {
                    'expected_utilities': expected_utilities,
                    'best_action': best_action,
                    'expected_value': expected_utilities[best_action],
                    'decision_criterion': 'expected_utility'
                }

        bayesian_reasoner = BayesianReasoner(prior_beliefs, evidence, likelihoods)

        belief_update = bayesian_reasoner.update_beliefs()

        # Simulação de decisão
        n_actions = len(prior_beliefs)
        utilities = np.random.rand(n_actions, len(prior_beliefs))  # Matriz de utilidades

        decision_result = bayesian_reasoner.decision_under_uncertainty(utilities)

        return {
            'belief_update': belief_update,
            'decision_result': decision_result
        }

    def heuristic_systematic_processing(self, message, recipient_characteristics):
        """
        Modelo de Processamento Heurístico vs Sistemático
        """
        class HeuristicSystematicModel:
            def __init__(self, message_content, recipient):
                self.message = message_content
                self.recipient = recipient

            def process_message(self):
                """Processar mensagem usando heurísticas ou análise sistemática"""
                # Capacidade cognitiva do destinatário
                cognitive_capacity = self.recipient.get('cognitive_capacity', 0.5)

                # Motivação para processamento
                motivation = self.recipient.get('motivation', 0.5)

                # Suficiência da heurística
                heuristic_cues = self._extract_heuristic_cues()
                heuristic_sufficiency = np.mean(heuristic_cues)

                # Decidir modo de processamento
                if cognitive_capacity > 0.7 and motivation > 0.7:
                    # Processamento sistemático
                    processing_mode = 'systematic'
                    persuasion = self._systematic_processing()
                elif heuristic_sufficiency > 0.6:
                    # Processamento heurístico
                    processing_mode = 'heuristic'
                    persuasion = self._heuristic_processing(heuristic_cues)
                else:
                    # Processamento sistemático forçado
                    processing_mode = 'biased_systematic'
                    persuasion = self._biased_systematic_processing(heuristic_cues)

                return {
                    'processing_mode': processing_mode,
                    'persuasion_level': persuasion,
                    'cognitive_capacity': cognitive_capacity,
                    'motivation': motivation,
                    'heuristic_sufficiency': heuristic_sufficiency
                }

            def _extract_heuristic_cues(self):
                """Extrair pistas heurísticas da mensagem"""
                cues = []

                # Comprimento da mensagem (heurística de autoridade)
                cues.append(min(len(self.message) / 1000, 1))

                # Presença de estatísticas (heurística de prova)
                if 'statistic' in self.message.lower() or '%' in self.message:
                    cues.append(0.8)
                else:
                    cues.append(0.3)

                # Linguagem emocional (heurística de afeto)
                emotional_words = ['amazing', 'terrible', 'wonderful', 'awful']
                emotional_score = sum(1 for word in emotional_words if word in self.message.lower())
                cues.append(min(emotional_score / 3, 1))

                return cues

            def _heuristic_processing(self, cues):
                """Processamento heurístico"""
                return np.mean(cues) * 0.7  # Influência moderada

            def _systematic_processing(self):
                """Processamento sistemático"""
                # Análise detalhada do conteúdo
                content_quality = len(self.message) / 500  # Qualidade baseada no comprimento
                argument_strength = np.random.uniform(0.4, 0.9)  # Força do argumento

                return (content_quality + argument_strength) / 2

            def _biased_systematic_processing(self, cues):
                """Processamento sistemático enviesado"""
                systematic_result = self._systematic_processing()
                heuristic_bias = np.mean(cues) * 0.3

                return systematic_result + heuristic_bias

        hsm = HeuristicSystematicModel(message, recipient_characteristics)
        processing_result = hsm.process_message()

        return processing_result

    def recognition_primed_decision(self, situation, experience_base):
        """
        Modelo de Decisão Primed por Reconhecimento
        """
        class RecognitionPrimedDecision:
            def __init__(self, situation, experiences):
                self.situation = situation
                self.experiences = experiences

            def make_decision(self):
                """Tomar decisão baseada em reconhecimento"""
                # Encontrar situações similares na memória
                similar_situations = self._find_similar_situations()

                if similar_situations:
                    # Usar experiência passada
                    best_experience = max(similar_situations, key=lambda x: x['outcome'])

                    decision = best_experience['action']
                    confidence = best_experience['outcome']

                    decision_type = 'recognition_primed'
                else:
                    # Decisão analítica
                    decision = self._analytical_decision()
                    confidence = 0.6  # Menos confiança

                    decision_type = 'analytical'

                return {
                    'decision': decision,
                    'confidence': confidence,
                    'decision_type': decision_type,
                    'similar_situations_found': len(similar_situations),
                    'situation_features': self.situation
                }

            def _find_similar_situations(self):
                """Encontrar situações similares"""
                similar = []

                for experience in self.experiences:
                    similarity = self._calculate_situation_similarity(
                        self.situation, experience['situation']
                    )

                    if similarity > 0.7:  # Threshold de similaridade
                        similar.append({
                            'action': experience['action'],
                            'outcome': experience['outcome'],
                            'similarity': similarity
                        })

                return similar

            def _calculate_situation_similarity(self, sit1, sit2):
                """Calcular similaridade entre situações"""
                common_features = 0
                total_features = len(sit1)

                for feature in sit1:
                    if feature in sit2 and abs(sit1[feature] - sit2[feature]) < 0.2:
                        common_features += 1

                return common_features / total_features if total_features > 0 else 0

            def _analytical_decision(self):
                """Decisão analítica quando sem reconhecimento"""
                # Lógica simples baseada em características da situação
                if self.situation.get('urgency', 0) > 0.7:
                    return 'immediate_action'
                elif self.situation.get('complexity', 0) > 0.8:
                    return 'consult_experts'
                else:
                    return 'standard_procedure'

        rpd = RecognitionPrimedDecision(situation, experience_base)
        decision_result = rpd.make_decision()

        return decision_result

    def naturalistic_decision_making(self, scenario, expertise_level):
        """
        Tomada de decisão naturalística
        """
        class NaturalisticDecisionMaker:
            def __init__(self, scenario_desc, expertise):
                self.scenario = scenario_desc
                self.expertise = expertise

            def situation_awareness(self):
                """Consciência situacional"""
                # Percepção dos elementos da situação
                perception_accuracy = self.expertise * 0.8 + np.random.normal(0, 0.1)

                # Compreensão das relações
                comprehension_accuracy = self.expertise * 0.7 + np.random.normal(0, 0.15)

                # Projeção do futuro
                projection_accuracy = self.expertise * 0.6 + np.random.normal(0, 0.2)

                situation_awareness = {
                    'perception': np.clip(perception_accuracy, 0, 1),
                    'comprehension': np.clip(comprehension_accuracy, 0, 1),
                    'projection': np.clip(projection_accuracy, 0, 1),
                    'overall_awareness': np.mean([
                        perception_accuracy, comprehension_accuracy, projection_accuracy
                    ])
                }

                return situation_awareness

            def recognition_primed_choice(self, options):
                """Escolha primed por reconhecimento"""
                # Avaliar opções baseada em expertise
                option_scores = []

                for option in options:
                    # Similaridade com experiências passadas
                    experience_match = self._match_past_experience(option)

                    # Adequação situacional
                    situational_fit = self._evaluate_situational_fit(option)

                    # Score combinado
                    score = self.expertise * experience_match + (1 - self.expertise) * situational_fit
                    option_scores.append(score)

                best_option_idx = np.argmax(option_scores)

                return {
                    'chosen_option': options[best_option_idx],
                    'option_scores': option_scores,
                    'decision_confidence': option_scores[best_option_idx],
                    'expertise_influence': self.expertise
                }

            def _match_past_experience(self, option):
                """Corresponder com experiência passada"""
                # Simulação baseada em expertise
                return self.expertise * np.random.uniform(0.5, 1.0)

            def _evaluate_situational_fit(self, option):
                """Avaliar adequação situacional"""
                # Análise baseada em características da opção
                if 'conservative' in option.lower():
                    return 0.6
                elif 'innovative' in option.lower():
                    return 0.4
                else:
                    return 0.5

        ndm = NaturalisticDecisionMaker(scenario, expertise_level)

        situation_awareness = ndm.situation_awareness()

        # Opções de decisão
        options = ['Conservative Approach', 'Innovative Solution', 'Standard Procedure']

        recognition_choice = ndm.recognition_primed_choice(options)

        return {
            'situation_awareness': situation_awareness,
            'decision_result': recognition_choice,
            'scenario_complexity': len(scenario.split()) / 100  # Medida simples
        }
```

**Tomada de Decisão e Raciocínio:**
- Teoria do Prospecto para decisões
- Raciocínio bayesiano
- Processamento heurístico vs sistemático
- Decisão primed por reconhecimento
- Tomada de decisão naturalística

---

## 3. APLICACOES PRÁTICAS E INTERFACES

### 3.1 Interfaces Cérebro-Computador
```python
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

class BrainComputerInterfaces:
    """
    Interfaces cérebro-computador baseadas em psicologia cognitiva
    """

    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.eeg_channels = ['Fz', 'Cz', 'Pz', 'Oz', 'C3', 'C4']

    def eeg_attention_monitoring(self, eeg_data, attention_task):
        """
        Monitoramento de atenção usando EEG
        """
        class AttentionMonitor:
            def __init__(self, eeg_data, task):
                self.eeg = eeg_data
                self.task = task

            def extract_attention_features(self):
                """Extrair características de atenção do EEG"""
                features = {}

                # Análise de potência por banda
                bands = {
                    'theta': (4, 8),
                    'alpha': (8, 12),
                    'beta': (12, 30),
                    'gamma': (30, 50)
                }

                for channel in self.eeg.keys():
                    channel_data = self.eeg[channel]

                    # FFT para análise espectral
                    freqs, psd = welch(channel_data, fs=250, nperseg=1024)

                    band_powers = {}
                    for band_name, (low_freq, high_freq) in bands.items():
                        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                        band_powers[band_name] = np.mean(psd[band_mask])

                    features[channel] = band_powers

                # Razão theta/beta (marcador de atenção)
                if 'Fz' in features:
                    theta_beta_ratio = features['Fz']['theta'] / features['Fz']['beta']
                    features['attention_ratio'] = theta_beta_ratio

                return features

            def classify_attention_state(self, features):
                """Classificar estado de atenção"""
                attention_ratio = features.get('attention_ratio', 1.0)

                # Thresholds baseados em literatura
                if attention_ratio > 1.5:
                    state = 'low_attention'
                    confidence = min(attention_ratio / 2, 1.0)
                elif attention_ratio < 0.8:
                    state = 'high_attention'
                    confidence = min(1 / attention_ratio, 1.0)
                else:
                    state = 'moderate_attention'
                    confidence = 0.8

                return {
                    'attention_state': state,
                    'confidence': confidence,
                    'attention_ratio': attention_ratio,
                    'feature_importance': {'theta_beta_ratio': 0.7, 'alpha_power': 0.3}
                }

            def adaptive_task_difficulty(self, attention_state):
                """Adaptar dificuldade da tarefa baseada na atenção"""
                if attention_state == 'low_attention':
                    new_difficulty = 'easier'
                    adaptation_reason = 'Increase engagement'
                elif attention_state == 'high_attention':
                    new_difficulty = 'harder'
                    adaptation_reason = 'Challenge user'
                else:
                    new_difficulty = 'maintain'
                    adaptation_reason = 'Optimal difficulty'

                return {
                    'adapted_difficulty': new_difficulty,
                    'adaptation_reason': adaptation_reason,
                    'attention_state': attention_state
                }

        attention_monitor = AttentionMonitor(eeg_data, attention_task)

        features = attention_monitor.extract_attention_features()
        attention_state = attention_monitor.classify_attention_state(features)
        adaptation = attention_monitor.adaptive_task_difficulty(attention_state['attention_state'])

        return {
            'attention_features': features,
            'attention_classification': attention_state,
            'task_adaptation': adaptation
        }

    def cognitive_workload_assessment(self, physiological_signals, task_complexity):
        """
        Avaliação de carga cognitiva
        """
        class WorkloadAssessor:
            def __init__(self, signals, complexity):
                self.signals = signals
                self.complexity = complexity

            def multimodal_workload_index(self):
                """Índice multimodal de carga cognitiva"""
                indices = {}

                # EEG: Beta/Alpha ratio
                if 'eeg' in self.signals:
                    eeg_data = self.signals['eeg']
                    beta_power = np.mean([self._band_power(ch, 12, 30) for ch in eeg_data.values()])
                    alpha_power = np.mean([self._band_power(ch, 8, 12) for ch in eeg_data.values()])

                    indices['eeg_workload'] = beta_power / (alpha_power + 1e-6)

                # Frequência cardíaca
                if 'heart_rate' in self.signals:
                    hr_data = self.signals['heart_rate']
                    mean_hr = np.mean(hr_data)
                    hr_variability = np.std(hr_data)

                    indices['cardiac_workload'] = mean_hr / (hr_variability + 1e-6)

                # Pupila
                if 'pupil_diameter' in self.signals:
                    pupil_data = self.signals['pupil_diameter']
                    mean_pupil = np.mean(pupil_data)
                    pupil_variation = np.std(pupil_data)

                    indices['pupil_workload'] = mean_pupil + pupil_variation

                # Índice combinado
                weights = {'eeg': 0.5, 'cardiac': 0.3, 'pupil': 0.2}
                combined_index = sum(
                    indices.get(signal_type, 0) * weight
                    for signal_type, weight in weights.items()
                )

                return {
                    'individual_indices': indices,
                    'combined_workload': combined_index,
                    'workload_level': self._classify_workload(combined_index),
                    'task_complexity': self.complexity
                }

            def _band_power(self, signal, low_freq, high_freq):
                """Calcular potência em banda específica"""
                freqs, psd = welch(signal, fs=250, nperseg=512)
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                return np.mean(psd[band_mask])

            def _classify_workload(self, index):
                """Classificar nível de carga cognitiva"""
                if index < 1.0:
                    return 'low'
                elif index < 2.0:
                    return 'moderate'
                else:
                    return 'high'

            def workload_adaptation_strategy(self, workload_level):
                """Estratégia de adaptação baseada na carga"""
                strategies = {
                    'low': {
                        'task_adjustment': 'increase_difficulty',
                        'interface_change': 'add_complexity',
                        'reason': 'Underutilized cognitive capacity'
                    },
                    'moderate': {
                        'task_adjustment': 'maintain',
                        'interface_change': 'optimize',
                        'reason': 'Optimal cognitive engagement'
                    },
                    'high': {
                        'task_adjustment': 'decrease_difficulty',
                        'interface_change': 'simplify',
                        'reason': 'Risk of cognitive overload'
                    }
                }

                return strategies.get(workload_level, strategies['moderate'])

        workload_assessor = WorkloadAssessor(physiological_signals, task_complexity)

        workload_analysis = workload_assessor.multimodal_workload_index()
        adaptation_strategy = workload_assessor.workload_adaptation_strategy(
            workload_analysis['workload_level']
        )

        return {
            'workload_analysis': workload_analysis,
            'adaptation_strategy': adaptation_strategy
        }

    def mental_state_decoding(self, neural_signals, mental_states):
        """
        Decodificação de estados mentais
        """
        class MentalStateDecoder:
            def __init__(self, signals, states):
                self.signals = signals
                self.states = states

            def decode_emotional_state(self):
                """Decodificar estado emocional"""
                # Análise de padrões emocionais no EEG
                emotional_patterns = {
                    'happy': {'alpha_frontal': 'high', 'beta_asymmetry': 'positive'},
                    'sad': {'alpha_frontal': 'low', 'beta_asymmetry': 'negative'},
                    'angry': {'beta_frontal': 'high', 'gamma': 'high'},
                    'calm': {'alpha': 'high', 'theta': 'high'}
                }

                # Decodificação baseada em características
                decoded_emotions = {}

                for emotion, pattern in emotional_patterns.items():
                    match_score = 0

                    for feature, expected in pattern.items():
                        if feature in self.signals:
                            # Calcular correspondência
                            signal_value = self._extract_feature(self.signals[feature])
                            match_score += self._pattern_match(signal_value, expected)

                    decoded_emotions[emotion] = match_score / len(pattern)

                best_emotion = max(decoded_emotions, key=decoded_emotions.get)

                return {
                    'decoded_emotions': decoded_emotions,
                    'primary_emotion': best_emotion,
                    'confidence': decoded_emotions[best_emotion],
                    'emotional_stability': np.std(list(decoded_emotions.values()))
                }

            def decode_cognitive_state(self):
                """Decodificar estado cognitivo"""
                cognitive_patterns = {
                    'focused': {'beta_frontal': 'high', 'theta_parietal': 'low'},
                    'distracted': {'alpha_occipital': 'high', 'beta_frontal': 'low'},
                    'creative': {'alpha': 'high', 'gamma': 'high'},
                    'analytical': {'beta': 'high', 'theta': 'low'}
                }

                decoded_states = {}

                for state, pattern in cognitive_patterns.items():
                    match_score = 0

                    for feature, expected in pattern.items():
                        if feature in self.signals:
                            signal_value = self._extract_feature(self.signals[feature])
                            match_score += self._pattern_match(signal_value, expected)

                    decoded_states[state] = match_score / len(pattern)

                best_state = max(decoded_states, key=decoded_states.get)

                return {
                    'decoded_states': decoded_states,
                    'primary_state': best_state,
                    'confidence': decoded_states[best_state],
                    'cognitive_engagement': np.mean(list(decoded_states.values()))
                }

            def _extract_feature(self, signal):
                """Extrair característica do sinal"""
                # Análise espectral simples
                freqs, psd = welch(signal, fs=250, nperseg=512)
                mean_power = np.mean(psd)

                return mean_power

            def _pattern_match(self, signal_value, expected):
                """Calcular correspondência com padrão"""
                if expected == 'high':
                    return min(signal_value / 1.0, 1.0)  # Normalizado
                elif expected == 'low':
                    return min(1.0 / (signal_value + 0.1), 1.0)
                elif expected == 'positive':
                    return max(0, signal_value) / 1.0
                elif expected == 'negative':
                    return max(0, -signal_value) / 1.0
                else:
                    return 0.5  # Correspondência neutra

        mental_decoder = MentalStateDecoder(neural_signals, mental_states)

        emotional_state = mental_decoder.decode_emotional_state()
        cognitive_state = mental_decoder.decode_cognitive_state()

        return {
            'emotional_decoding': emotional_state,
            'cognitive_decoding': cognitive_state,
            'overall_mental_state': {
                'primary_emotion': emotional_state['primary_emotion'],
                'primary_cognition': cognitive_state['primary_state'],
                'mental_coherence': (emotional_state['confidence'] + cognitive_state['confidence']) / 2
            }
        }

    def adaptive_learning_system(self, student_signals, learning_content):
        """
        Sistema de aprendizado adaptativo baseado em sinais cognitivos
        """
        class AdaptiveLearning:
            def __init__(self, signals, content):
                self.signals = signals
                self.content = content

            def assess_learning_state(self):
                """Avaliar estado de aprendizagem"""
                # Medir engajamento cognitivo
                engagement_indicators = {}

                if 'eeg' in self.signals:
                    # Razão beta/alpha como indicador de engajamento
                    beta_power = np.mean([self._band_power(ch, 12, 30)
                                        for ch in self.signals['eeg'].values()])
                    alpha_power = np.mean([self._band_power(ch, 8, 12)
                                         for ch in self.signals['eeg'].values()])

                    engagement_indicators['cognitive_engagement'] = beta_power / (alpha_power + 1e-6)

                if 'heart_rate' in self.signals:
                    # Variabilidade da frequência cardíaca
                    hr_data = self.signals['heart_rate']
                    hr_mean = np.mean(hr_data)
                    hr_std = np.std(hr_data)

                    engagement_indicators['physiological_arousal'] = hr_std / (hr_mean + 1e-6)

                # Estado de aprendizagem combinado
                combined_engagement = np.mean(list(engagement_indicators.values()))

                if combined_engagement > 1.2:
                    learning_state = 'highly_engaged'
                elif combined_engagement > 0.8:
                    learning_state = 'moderately_engaged'
                else:
                    learning_state = 'disengaged'

                return {
                    'learning_state': learning_state,
                    'engagement_indicators': engagement_indicators,
                    'combined_engagement': combined_engagement
                }

            def adapt_content_difficulty(self, learning_state):
                """Adaptar dificuldade do conteúdo"""
                adaptation_rules = {
                    'highly_engaged': {
                        'difficulty_adjustment': 'increase',
                        'pace_adjustment': 'faster',
                        'content_type': 'advanced'
                    },
                    'moderately_engaged': {
                        'difficulty_adjustment': 'maintain',
                        'pace_adjustment': 'steady',
                        'content_type': 'standard'
                    },
                    'disengaged': {
                        'difficulty_adjustment': 'decrease',
                        'pace_adjustment': 'slower',
                        'content_type': 'remedial'
                    }
                }

                return adaptation_rules.get(learning_state['learning_state'],
                                          adaptation_rules['moderately_engaged'])

            def personalize_learning_path(self, student_profile):
                """Personalizar caminho de aprendizagem"""
                # Análise do perfil do estudante
                learning_style = student_profile.get('learning_style', 'visual')
                prior_knowledge = student_profile.get('prior_knowledge', 0.5)
                motivation_level = student_profile.get('motivation', 0.5)

                # Recomendação de conteúdo
                if learning_style == 'visual':
                    content_recommendation = 'video_lectures'
                elif learning_style == 'auditory':
                    content_recommendation = 'audio_explanations'
                else:
                    content_recommendation = 'interactive_simulations'

                # Ajuste baseado em conhecimento prévio
                if prior_knowledge < 0.3:
                    difficulty_start = 'basic'
                elif prior_knowledge < 0.7:
                    difficulty_start = 'intermediate'
                else:
                    difficulty_start = 'advanced'

                return {
                    'content_recommendation': content_recommendation,
                    'starting_difficulty': difficulty_start,
                    'motivation_boost_needed': motivation_level < 0.6,
                    'personalized_path': f"{difficulty_start}_{content_recommendation}"
                }

            def _band_power(self, signal, low_freq, high_freq):
                """Calcular potência em banda específica"""
                freqs, psd = welch(signal, fs=250, nperseg=512)
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                return np.mean(psd[band_mask])

        adaptive_learner = AdaptiveLearning(student_signals, learning_content)

        learning_state = adaptive_learner.assess_learning_state()
        content_adaptation = adaptive_learner.adapt_content_difficulty(learning_state)

        # Perfil do estudante simulado
        student_profile = {
            'learning_style': 'visual',
            'prior_knowledge': 0.6,
            'motivation': 0.7
        }

        personalized_path = adaptive_learner.personalize_learning_path(student_profile)

        return {
            'learning_assessment': learning_state,
            'content_adaptation': content_adaptation,
            'personalized_learning': personalized_path
        }
```

**Interfaces Cérebro-Computador:**
- Monitoramento de atenção via EEG
- Avaliação de carga cognitiva multimodal
- Decodificação de estados mentais
- Sistema de aprendizado adaptativo

---

## 4. CONSIDERAÇÕES FINAIS

A psicologia cognitiva oferece um arcabouço fundamental para compreender como a mente humana processa informação, toma decisões e interage com o mundo. Os modelos apresentados fornecem ferramentas para:

1. **Arquiteturas Cognitivas**: Modelagem de sistemas de processamento de informação
2. **Memória e Aprendizado**: Compreensão de mecanismos de armazenamento e recuperação
3. **Percepção e Atenção**: Processos de detecção e seleção de informação
4. **Tomada de Decisão**: Modelos de escolha racional e irracional
5. **Aplicações Práticas**: Interfaces adaptativas e sistemas inteligentes

**Próximos Passos Recomendados**:
1. Dominar fundamentos da arquitetura cognitiva (ACT-R, SOAR)
2. Explorar modelos conexionistas de memória e aprendizado
3. Compreender vieses cognitivos e heurísticas de decisão
4. Desenvolver aplicações práticas em BCI e aprendizado adaptativo
5. Integrar insights cognitivos com desenvolvimento de IA

---

*Documento preparado para fine-tuning de IA em Psicologia Cognitiva*
*Versão 1.0 - Preparado para implementação prática*
