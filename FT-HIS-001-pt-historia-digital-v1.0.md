# FT-HIS-001: Fine-Tuning para IA em História Digital

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados em história digital, integrando métodos computacionais para análise histórica, preservação digital de documentos, modelagem de eventos históricos, análise de redes sociais do passado e visualização de dados históricos com princípios da ciência histórica e computação.

### Contexto Filosófico
A história digital representa a evolução da historiografia tradicional para uma abordagem data-driven, onde métodos computacionais revelam padrões, conexões e narrativas ocultas nos registros históricos. Esta abordagem reconhece que a história não é apenas narrativa, mas também pode ser quantificada, visualizada e analisada através de algoritmos formais.

### Metodologia de Aprendizado Recomendada
1. **Fundamentos Históricos**: Compreensão de métodos historiográficos tradicionais
2. **Análise Computacional**: Desenvolvimento de algoritmos para análise de texto histórico
3. **Preservação Digital**: Técnicas para digitalização e preservação de documentos
4. **Modelagem Temporal**: Análise de séries temporais e eventos históricos
5. **Visualização Interativa**: Criação de interfaces para exploração histórica

---

## 1. ANÁLISE COMPUTACIONAL DE TEXTO HISTÓRICO

### 1.1 Processamento de Linguagem Natural para Textos Históricos
```python
import numpy as np
from collections import Counter, defaultdict
import re
from datetime import datetime
import matplotlib.pyplot as plt

class HistoricalTextAnalysis:
    """
    Análise computacional de textos históricos
    """

    def __init__(self):
        self.stop_words = set(['a', 'o', 'as', 'os', 'de', 'do', 'da', 'das', 'dos',
                              'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com',
                              'que', 'um', 'uma', 'uns', 'umas', 'seu', 'sua'])

    def temporal_text_analysis(self, documents, dates):
        """
        Análise temporal de textos históricos
        """
        class TemporalAnalyzer:
            def __init__(self, docs, temporal_info):
                self.documents = docs
                self.dates = temporal_info
                self.vocabulary = self._build_vocabulary()

            def _build_vocabulary(self):
                """Construir vocabulário do corpus"""
                all_words = []

                for doc in self.documents:
                    words = self._tokenize_and_clean(doc)
                    all_words.extend(words)

                # Frequência de palavras
                word_freq = Counter(all_words)

                # Remover palavras muito frequentes e muito raras
                vocabulary = {word: freq for word, freq in word_freq.items()
                            if 3 <= freq <= len(self.documents) * 0.8}

                return vocabulary

            def _tokenize_and_clean(self, text):
                """Tokenizar e limpar texto"""
                # Converter para minúsculas
                text = text.lower()

                # Remover pontuação e números
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\d+', '', text)

                # Tokenizar
                words = text.split()

                # Remover stop words
                words = [word for word in words if word not in self.stop_words and len(word) > 2]

                return words

            def calculate_term_frequencies(self):
                """Calcular frequências de termos ao longo do tempo"""
                # Organizar documentos por período
                time_periods = self._create_time_periods()

                term_frequencies = defaultdict(lambda: defaultdict(int))

                for period, period_docs in time_periods.items():
                    period_text = ' '.join(period_docs)
                    words = self._tokenize_and_clean(period_text)

                    for word in words:
                        if word in self.vocabulary:
                            term_frequencies[word][period] += 1

                return dict(term_frequencies)

            def _create_time_periods(self):
                """Criar períodos temporais"""
                # Agrupar por décadas
                periods = defaultdict(list)

                for doc, date in zip(self.documents, self.dates):
                    if isinstance(date, str):
                        try:
                            year = datetime.strptime(date, '%Y-%m-%d').year
                        except:
                            year = 1900  # Ano padrão
                    else:
                        year = date

                    decade = (year // 10) * 10
                    periods[decade].append(doc)

                return dict(periods)

            def identify_temporal_trends(self):
                """Identificar tendências temporais"""
                term_freqs = self.calculate_term_frequencies()

                trends = {}

                for term, freqs in term_freqs.items():
                    # Calcular tendência linear
                    periods = sorted(freqs.keys())
                    frequencies = [freqs[period] for period in periods]

                    if len(periods) > 2:
                        # Ajuste linear
                        x = np.array(periods)
                        y = np.array(frequencies)

                        slope = np.polyfit(x, y, 1)[0]
                        trends[term] = {
                            'slope': slope,
                            'trend': 'increasing' if slope > 0 else 'decreasing',
                            'periods': periods,
                            'frequencies': frequencies
                        }

                return trends

            def semantic_network_analysis(self):
                """Análise de rede semântica"""
                # Construir matriz de co-ocorrência
                window_size = 5
                co_occurrence = defaultdict(lambda: defaultdict(int))

                for doc in self.documents:
                    words = self._tokenize_and_clean(doc)

                    for i, word1 in enumerate(words):
                        if word1 in self.vocabulary:
                            # Janela de co-ocorrência
                            start = max(0, i - window_size)
                            end = min(len(words), i + window_size + 1)

                            for j in range(start, end):
                                if i != j:
                                    word2 = words[j]
                                    if word2 in self.vocabulary:
                                        co_occurrence[word1][word2] += 1

                return dict(co_occurrence)

        temporal_analyzer = TemporalAnalyzer(documents, dates)
        term_frequencies = temporal_analyzer.calculate_term_frequencies()
        trends = temporal_analyzer.identify_temporal_trends()
        semantic_network = temporal_analyzer.semantic_network_analysis()

        return {
            'temporal_analyzer': temporal_analyzer,
            'term_frequencies': term_frequencies,
            'temporal_trends': trends,
            'semantic_network': semantic_network
        }

    def topic_modeling_historical(self, documents, n_topics=10):
        """
        Modelagem de tópicos para textos históricos
        """
        class HistoricalTopicModeler:
            def __init__(self, docs, n_topics):
                self.documents = docs
                self.n_topics = n_topics
                self.vocabulary = self._build_vocabulary()

            def _build_vocabulary(self):
                """Construir vocabulário"""
                all_words = []

                for doc in self.documents:
                    words = self._preprocess_text(doc)
                    all_words.extend(words)

                word_freq = Counter(all_words)
                vocabulary = [word for word, freq in word_freq.items() if freq >= 3]

                return vocabulary

            def _preprocess_text(self, text):
                """Pré-processar texto"""
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\d+', '', text)

                words = text.split()
                words = [word for word in words if word not in self.stop_words and len(word) > 2]

                return words

            def fit_lda_model(self):
                """Ajustar modelo LDA"""
                # Implementação simplificada do LDA
                n_docs = len(self.documents)
                n_vocab = len(self.vocabulary)

                # Matriz documento-termo
                doc_term_matrix = np.zeros((n_docs, n_vocab))

                for i, doc in enumerate(self.documents):
                    words = self._preprocess_text(doc)
                    word_counts = Counter(words)

                    for word, count in word_counts.items():
                        if word in self.vocabulary:
                            j = self.vocabulary.index(word)
                            doc_term_matrix[i, j] = count

                # Distribuições de tópicos (simplificado)
                topic_word_dist = np.random.dirichlet(np.ones(n_vocab), self.n_topics)
                doc_topic_dist = np.random.dirichlet(np.ones(self.n_topics), n_docs)

                # Otimização simplificada
                for iteration in range(50):
                    # E-step
                    for d in range(n_docs):
                        for k in range(self.n_topics):
                            likelihood = np.sum(doc_term_matrix[d] * np.log(topic_word_dist[k] + 1e-10))
                            doc_topic_dist[d, k] = np.exp(likelihood)

                        doc_topic_dist[d] /= np.sum(doc_topic_dist[d])

                    # M-step
                    for k in range(self.n_topics):
                        for w in range(n_vocab):
                            topic_word_dist[k, w] = np.sum(doc_term_matrix[:, w] * doc_topic_dist[:, k])
                        topic_word_dist[k] /= np.sum(topic_word_dist[k])

                return {
                    'topic_word_distributions': topic_word_dist,
                    'doc_topic_distributions': doc_topic_dist,
                    'vocabulary': self.vocabulary,
                    'n_topics': self.n_topics
                }

            def extract_topic_keywords(self, topic_word_dist, n_keywords=10):
                """Extrair palavras-chave para cada tópico"""
                topic_keywords = {}

                for k in range(self.n_topics):
                    word_probs = topic_word_dist[k]
                    top_indices = np.argsort(word_probs)[-n_keywords:][::-1]
                    keywords = [self.vocabulary[i] for i in top_indices]
                    probabilities = [word_probs[i] for i in top_indices]

                    topic_keywords[f'topic_{k}'] = {
                        'keywords': keywords,
                        'probabilities': probabilities
                    }

                return topic_keywords

            def temporal_topic_evolution(self, dates):
                """Evolução temporal dos tópicos"""
                lda_results = self.fit_lda_model()
                doc_topic_dist = lda_results['doc_topic_distributions']

                # Organizar por tempo
                time_periods = defaultdict(list)

                for i, date in enumerate(dates):
                    if isinstance(date, str):
                        try:
                            year = datetime.strptime(date, '%Y-%m-%d').year
                        except:
                            year = 1900
                    else:
                        year = date

                    decade = (year // 10) * 10
                    time_periods[decade].append(doc_topic_dist[i])

                # Média por período
                temporal_evolution = {}

                for period, topic_distributions in time_periods.items():
                    avg_topic_dist = np.mean(topic_distributions, axis=0)
                    temporal_evolution[period] = avg_topic_dist

                return {
                    'temporal_evolution': temporal_evolution,
                    'lda_model': lda_results
                }

        topic_modeler = HistoricalTopicModeler(documents, n_topics)
        lda_results = topic_modeler.fit_lda_model()
        topic_keywords = topic_modeler.extract_topic_keywords(lda_results['topic_word_distributions'])

        return {
            'topic_modeler': topic_modeler,
            'lda_results': lda_results,
            'topic_keywords': topic_keywords
        }

    def sentiment_analysis_historical(self, texts, time_periods):
        """
        Análise de sentimento em textos históricos
        """
        class HistoricalSentimentAnalyzer:
            def __init__(self, texts, periods):
                self.texts = texts
                self.periods = periods
                self.sentiment_lexicon = self._build_sentiment_lexicon()

            def _build_sentiment_lexicon(self):
                """Construir léxico de sentimento para português histórico"""
                # Léxico simplificado
                positive_words = ['bom', 'excelente', 'ótimo', 'maravilhoso', 'feliz',
                                'alegre', 'bem', 'melhor', 'superior', 'vantajoso']

                negative_words = ['mau', 'ruim', 'terrível', 'péssimo', 'triste',
                                'infeliz', 'mal', 'pior', 'inferior', 'desvantajoso']

                neutral_words = ['neutro', 'normal', 'regular', 'comum', 'habitual']

                return {
                    'positive': set(positive_words),
                    'negative': set(negative_words),
                    'neutral': set(neutral_words)
                }

            def analyze_sentiment(self, text):
                """Analisar sentimento de um texto"""
                words = self._preprocess_text(text)

                positive_score = 0
                negative_score = 0

                for word in words:
                    if word in self.sentiment_lexicon['positive']:
                        positive_score += 1
                    elif word in self.sentiment_lexicon['negative']:
                        negative_score += 1

                total_words = len(words)

                if total_words == 0:
                    return {'sentiment': 'neutral', 'score': 0}

                sentiment_score = (positive_score - negative_score) / total_words

                if sentiment_score > 0.1:
                    sentiment = 'positive'
                elif sentiment_score < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'

                return {
                    'sentiment': sentiment,
                    'score': sentiment_score,
                    'positive_words': positive_score,
                    'negative_words': negative_score
                }

            def _preprocess_text(self, text):
                """Pré-processar texto"""
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                words = text.split()
                words = [word for word in words if word not in self.stop_words]

                return words

            def temporal_sentiment_trends(self):
                """Tendências de sentimento ao longo do tempo"""
                sentiment_trends = defaultdict(list)

                for text, period in zip(self.texts, self.periods):
                    sentiment_result = self.analyze_sentiment(text)
                    sentiment_trends[period].append(sentiment_result['score'])

                # Calcular médias por período
                temporal_sentiment = {}

                for period, scores in sentiment_trends.items():
                    avg_sentiment = np.mean(scores)
                    std_sentiment = np.std(scores)

                    temporal_sentiment[period] = {
                        'average_sentiment': avg_sentiment,
                        'sentiment_std': std_sentiment,
                        'n_documents': len(scores)
                    }

                return temporal_sentiment

            def sentiment_network_analysis(self):
                """Análise de rede de sentimento"""
                # Rede de co-ocorrência de palavras de sentimento
                sentiment_network = defaultdict(lambda: defaultdict(int))

                for text in self.texts:
                    words = self._preprocess_text(text)
                    sentiment_words = []

                    # Identificar palavras de sentimento
                    for word in words:
                        for sentiment_type, word_set in self.sentiment_lexicon.items():
                            if word in word_set:
                                sentiment_words.append((word, sentiment_type))
                                break

                    # Construir rede
                    for i in range(len(sentiment_words)):
                        for j in range(i + 1, len(sentiment_words)):
                            word1, type1 = sentiment_words[i]
                            word2, type2 = sentiment_words[j]

                            if type1 == type2:  # Mesma polaridade
                                sentiment_network[word1][word2] += 1
                                sentiment_network[word2][word1] += 1

                return dict(sentiment_network)

        sentiment_analyzer = HistoricalSentimentAnalyzer(texts, time_periods)
        temporal_trends = sentiment_analyzer.temporal_sentiment_trends()
        sentiment_network = sentiment_analyzer.sentiment_network_analysis()

        return {
            'sentiment_analyzer': sentiment_analyzer,
            'temporal_sentiment_trends': temporal_trends,
            'sentiment_network': sentiment_network
        }

    def authorship_attribution(self, documents, candidate_authors):
        """
        Atribuição de autoria para textos históricos
        """
        class AuthorshipAttribution:
            def __init__(self, docs, authors):
                self.documents = docs
                self.authors = authors
                self.author_profiles = self._build_author_profiles()

            def _build_author_profiles(self):
                """Construir perfis de autores"""
                author_docs = defaultdict(list)

                # Assumir que cada documento tem um autor conhecido
                for doc, author in zip(self.documents, self.authors):
                    author_docs[author].append(doc)

                author_profiles = {}

                for author, docs in author_docs.items():
                    # Características estilométricas
                    all_text = ' '.join(docs)
                    words = self._preprocess_text(all_text)

                    # Frequência de palavras
                    word_freq = Counter(words)

                    # Comprimento médio de sentenças
                    sentences = re.split(r'[.!?]+', all_text)
                    avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])

                    # Diversidade lexical
                    lexical_diversity = len(set(words)) / len(words) if words else 0

                    author_profiles[author] = {
                        'word_frequencies': dict(word_freq.most_common(100)),
                        'avg_sentence_length': avg_sentence_length,
                        'lexical_diversity': lexical_diversity,
                        'total_words': len(words)
                    }

                return author_profiles

            def _preprocess_text(self, text):
                """Pré-processar texto"""
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                words = text.split()
                words = [word for word in words if word not in self.stop_words]

                return words

            def attribute_authorship(self, unknown_document):
                """Atribuir autoria a um documento desconhecido"""
                unknown_words = self._preprocess_text(unknown_document)
                unknown_freq = Counter(unknown_words)

                # Calcular similaridade com cada autor
                similarities = {}

                for author, profile in self.author_profiles.items():
                    # Similaridade baseada em frequência de palavras
                    common_words = set(unknown_freq.keys()) & set(profile['word_frequencies'].keys())

                    similarity_score = 0

                    for word in common_words:
                        unknown_count = unknown_freq[word]
                        author_freq = profile['word_frequencies'].get(word, 0) / profile['total_words']

                        similarity_score += unknown_count * author_freq

                    # Normalizar
                    similarity_score /= len(unknown_words)

                    similarities[author] = similarity_score

                # Autor mais provável
                best_author = max(similarities, key=similarities.get)

                return {
                    'attributed_author': best_author,
                    'similarity_scores': similarities,
                    'confidence': similarities[best_author] / sum(similarities.values())
                }

            def cross_validation_accuracy(self):
                """Avaliar acurácia usando validação cruzada"""
                correct_predictions = 0
                total_predictions = 0

                # Leave-one-out cross-validation
                for i in range(len(self.documents)):
                    # Remover um documento para teste
                    test_doc = self.documents[i]
                    test_author = self.authors[i]

                    train_docs = self.documents[:i] + self.documents[i+1:]
                    train_authors = self.authors[:i] + self.authors[i+1:]

                    # Treinar modelo com dados restantes
                    temp_model = AuthorshipAttribution(train_docs, train_authors)

                    # Prever autor do documento de teste
                    prediction = temp_model.attribute_authorship(test_doc)

                    if prediction['attributed_author'] == test_author:
                        correct_predictions += 1

                    total_predictions += 1

                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

                return {
                    'accuracy': accuracy,
                    'correct_predictions': correct_predictions,
                    'total_predictions': total_predictions
                }

        authorship_model = AuthorshipAttribution(documents, candidate_authors)
        attribution_accuracy = authorship_model.cross_validation_accuracy()

        return {
            'authorship_model': authorship_model,
            'cross_validation_accuracy': attribution_accuracy
        }
```

**Análise Computacional de Texto Histórico:**
- Análise temporal de textos históricos
- Modelagem de tópicos para documentos históricos
- Análise de sentimento em contexto histórico
- Atribuição de autoria usando estilometria

### 1.2 Preservação Digital e OCR Avançado
```python
import numpy as np
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt

class DigitalPreservation:
    """
    Técnicas de preservação digital para documentos históricos
    """

    def __init__(self):
        self.ocr_engines = {}

    def document_image_enhancement(self, document_image):
        """
        Aprimoramento de imagens de documentos históricos
        """
        class ImageEnhancer:
            def __init__(self, image):
                self.original_image = image
                self.processed_image = image.copy()

            def noise_reduction(self):
                """Redução de ruído"""
                # Filtro de mediana para reduzir ruído de sal e pimenta
                self.processed_image = cv2.medianBlur(self.processed_image, 3)

                # Filtro bilateral para preservar bordas
                self.processed_image = cv2.bilateralFilter(self.processed_image, 9, 75, 75)

                return self.processed_image

            def contrast_enhancement(self):
                """Melhoria de contraste"""
                # Equalização de histograma adaptativa
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                if len(self.processed_image.shape) == 3:
                    # Imagem colorida
                    lab = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    self.processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    # Imagem em tons de cinza
                    self.processed_image = clahe.apply(self.processed_image)

                return self.processed_image

            def binarization(self, method='adaptive'):
                """Binarização de imagem"""
                if method == 'adaptive':
                    # Binarização adaptativa
                    self.processed_image = cv2.adaptiveThreshold(
                        self.processed_image, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                elif method == 'otsu':
                    # Método de Otsu
                    _, self.processed_image = cv2.threshold(
                        self.processed_image, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                return self.processed_image

            def text_line_extraction(self):
                """Extração de linhas de texto"""
                # Detecção de linhas usando projeção horizontal
                if len(self.processed_image.shape) == 3:
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image

                # Projeção horizontal
                horizontal_projection = np.sum(gray, axis=1)

                # Suavização
                horizontal_projection = cv2.GaussianBlur(horizontal_projection.reshape(-1, 1), (1, 15), 0).flatten()

                # Detecção de picos (linhas de texto)
                threshold = np.mean(horizontal_projection) * 0.8
                line_positions = []

                for i in range(1, len(horizontal_projection) - 1):
                    if (horizontal_projection[i] > threshold and
                        horizontal_projection[i] > horizontal_projection[i-1] and
                        horizontal_projection[i] > horizontal_projection[i+1]):
                        line_positions.append(i)

                return {
                    'processed_image': self.processed_image,
                    'line_positions': line_positions,
                    'horizontal_projection': horizontal_projection
                }

            def complete_enhancement_pipeline(self):
                """Pipeline completo de aprimoramento"""
                # Converter para array numpy se necessário
                if isinstance(self.original_image, Image.Image):
                    self.processed_image = np.array(self.original_image)

                # Pipeline de processamento
                self.noise_reduction()
                self.contrast_enhancement()
                self.binarization()

                # Extração de linhas
                line_extraction = self.text_line_extraction()

                return {
                    'enhanced_image': self.processed_image,
                    'line_extraction': line_extraction,
                    'enhancement_steps': ['noise_reduction', 'contrast_enhancement', 'binarization']
                }

        enhancer = ImageEnhancer(document_image)
        enhancement_result = enhancer.complete_enhancement_pipeline()

        return enhancement_result

    def historical_ocr_system(self, enhanced_image, language_model):
        """
        Sistema OCR para documentos históricos
        """
        class HistoricalOCR:
            def __init__(self, image, lang_model):
                self.image = image
                self.language_model = lang_model

            def character_segmentation(self):
                """Segmentação de caracteres"""
                # Usar connected components para segmentação
                if len(self.image.shape) == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image

                # Binarização se necessário
                if np.max(gray) > 1:
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                else:
                    binary = (gray * 255).astype(np.uint8)

                # Componentes conectados
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

                # Filtrar componentes (remover muito pequenos ou muito grandes)
                character_candidates = []

                for i in range(1, num_labels):  # Pular fundo
                    area = stats[i, cv2.CC_STAT_AREA]
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    height = stats[i, cv2.CC_STAT_HEIGHT]

                    # Critérios para caracteres
                    if 20 < area < 1000 and 0.2 < width/height < 5:
                        character_candidates.append({
                            'label': i,
                            'bbox': stats[i],
                            'centroid': centroids[i],
                            'area': area
                        })

                return character_candidates

            def character_recognition(self, character_candidates):
                """Reconhecimento de caracteres usando modelo de linguagem"""
                recognized_characters = []

                for candidate in character_candidates:
                    # Extrair região do caractere
                    x, y, w, h = candidate['bbox'][:4]
                    char_image = self.image[y:y+h, x:x+w]

                    # Reconhecimento simplificado (placeholder)
                    # Em implementação real, usaria modelo CNN treinado
                    predicted_char = self._simple_character_recognition(char_image)

                    recognized_characters.append({
                        'character': predicted_char,
                        'bbox': candidate['bbox'],
                        'confidence': np.random.uniform(0.7, 0.95)  # Placeholder
                    })

                return recognized_characters

            def _simple_character_recognition(self, char_image):
                """Reconhecimento simples de caracteres"""
                # Placeholder: retornar caractere aleatório
                # Em implementação real: usar modelo treinado
                chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                return np.random.choice(list(chars))

            def word_formation(self, recognized_characters):
                """Formação de palavras usando modelo de linguagem"""
                # Agrupar caracteres em linhas
                lines = self._group_characters_into_lines(recognized_characters)

                words = []

                for line in lines:
                    # Ordenar caracteres por posição x
                    line_sorted = sorted(line, key=lambda x: x['bbox'][0])

                    # Formar palavras baseado em proximidade
                    current_word = []
                    word_start = line_sorted[0]['bbox'][0] if line_sorted else 0

                    for char in line_sorted:
                        char_center = char['bbox'][0] + char['bbox'][2] / 2

                        # Se caractere está próximo da palavra atual
                        if char_center - word_start < 50:  # Distância máxima
                            current_word.append(char)
                        else:
                            # Finalizar palavra atual e começar nova
                            if current_word:
                                word_text = ''.join([c['character'] for c in current_word])
                                words.append({
                                    'text': word_text,
                                    'characters': current_word,
                                    'bbox': self._get_word_bbox(current_word)
                                })

                            current_word = [char]
                            word_start = char_center

                    # Última palavra
                    if current_word:
                        word_text = ''.join([c['character'] for c in current_word])
                        words.append({
                            'text': word_text,
                            'characters': current_word,
                            'bbox': self._get_word_bbox(current_word)
                        })

                return words

            def _group_characters_into_lines(self, characters):
                """Agrupar caracteres em linhas"""
                if not characters:
                    return []

                # Ordenar por coordenada y
                characters_sorted = sorted(characters, key=lambda x: x['bbox'][1])

                lines = []
                current_line = [characters_sorted[0]]

                for char in characters_sorted[1:]:
                    # Verificar se está na mesma linha
                    if abs(char['bbox'][1] - current_line[0]['bbox'][1]) < 10:  # Tolerância
                        current_line.append(char)
                    else:
                        lines.append(current_line)
                        current_line = [char]

                if current_line:
                    lines.append(current_line)

                return lines

            def _get_word_bbox(self, characters):
                """Calcular bounding box de uma palavra"""
                min_x = min([c['bbox'][0] for c in characters])
                min_y = min([c['bbox'][1] for c in characters])
                max_x = max([c['bbox'][0] + c['bbox'][2] for c in characters])
                max_y = max([c['bbox'][1] + c['bbox'][3] for c in characters])

                return [min_x, min_y, max_x - min_x, max_y - min_y]

            def post_processing_correction(self, words):
                """Correção pós-processamento usando modelo de linguagem"""
                corrected_words = []

                for word in words:
                    corrected_text = self._language_model_correction(word['text'])
                    corrected_words.append({
                        'original': word['text'],
                        'corrected': corrected_text,
                        'bbox': word['bbox'],
                        'confidence': word.get('confidence', 0.8)
                    })

                return corrected_words

            def _language_model_correction(self, text):
                """Correção usando modelo de linguagem"""
                # Placeholder: correção simples baseada em dicionário
                corrections = {
                    'teh': 'the',
                    'adn': 'and',
                    'si': 'is',
                    'taht': 'that'
                }

                return corrections.get(text.lower(), text)

        ocr_system = HistoricalOCR(enhanced_image, language_model)
        character_candidates = ocr_system.character_segmentation()
        recognized_characters = ocr_system.character_recognition(character_candidates)
        words = ocr_system.word_formation(recognized_characters)
        corrected_text = ocr_system.post_processing_correction(words)

        return {
            'ocr_system': ocr_system,
            'character_candidates': character_candidates,
            'recognized_characters': recognized_characters,
            'extracted_words': words,
            'corrected_text': corrected_text
        }

    def document_layout_analysis(self, document_image):
        """
        Análise de layout de documentos históricos
        """
        class LayoutAnalyzer:
            def __init__(self, image):
                self.image = image

            def detect_document_regions(self):
                """Detectar regiões do documento"""
                # Converter para escala de cinza
                if len(self.image.shape) == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image

                # Binarização
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Detecção de contornos
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                regions = []

                for contour in contours:
                    area = cv2.contourArea(contour)

                    if area > 1000:  # Filtrar regiões muito pequenas
                        x, y, w, h = cv2.boundingRect(contour)

                        # Classificar região
                        aspect_ratio = w / h

                        if aspect_ratio > 3:  # Largura muito maior que altura
                            region_type = 'text_line'
                        elif aspect_ratio < 0.3:  # Altura muito maior que largura
                            region_type = 'text_column'
                        else:
                            region_type = 'text_block'

                        regions.append({
                            'bbox': [x, y, w, h],
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'type': region_type,
                            'contour': contour
                        })

                return regions

            def classify_document_zones(self, regions):
                """Classificar zonas do documento"""
                # Agrupar regiões similares
                text_regions = []
                image_regions = []
                table_regions = []

                for region in regions:
                    if region['type'] in ['text_line', 'text_block', 'text_column']:
                        text_regions.append(region)
                    elif region['aspect_ratio'] > 0.8 and region['aspect_ratio'] < 1.2:
                        # Quadrado - possivelmente imagem
                        image_regions.append(region)
                    else:
                        table_regions.append(region)

                return {
                    'text_regions': text_regions,
                    'image_regions': image_regions,
                    'table_regions': table_regions,
                    'all_regions': regions
                }

            def reading_order_analysis(self):
                """Análise de ordem de leitura"""
                regions = self.detect_document_regions()
                classified_zones = self.classify_document_zones(regions)

                # Ordenar regiões de texto por posição
                text_regions = classified_zones['text_regions']

                # Ordenar por coordenada y (topo para baixo), depois por x (esquerda para direita)
                reading_order = sorted(text_regions,
                                     key=lambda r: (r['bbox'][1], r['bbox'][0]))

                return {
                    'reading_order': reading_order,
                    'classified_zones': classified_zones
                }

        layout_analyzer = LayoutAnalyzer(document_image)
        regions = layout_analyzer.detect_document_regions()
        zones = layout_analyzer.classify_document_zones(regions)
        reading_order = layout_analyzer.reading_order_analysis()

        return {
            'layout_analyzer': layout_analyzer,
            'detected_regions': regions,
            'classified_zones': zones,
            'reading_order': reading_order
        }

    def metadata_extraction_system(self, document_content, document_metadata):
        """
        Sistema de extração de metadados para preservação
        """
        class MetadataExtractor:
            def __init__(self, content, metadata):
                self.content = content
                self.metadata = metadata

            def extract_structural_metadata(self):
                """Extrair metadados estruturais"""
                # Análise de layout
                n_pages = self._count_pages()
                n_paragraphs = self._count_paragraphs()
                n_words = self._count_words()
                n_characters = self._count_characters()

                # Linguagem e codificação
                language = self._detect_language()
                encoding = self._detect_encoding()

                return {
                    'n_pages': n_pages,
                    'n_paragraphs': n_paragraphs,
                    'n_words': n_words,
                    'n_characters': n_characters,
                    'language': language,
                    'encoding': encoding
                }

            def extract_content_metadata(self):
                """Extrair metadados de conteúdo"""
                # Tópicos principais
                main_topics = self._extract_topics()

                # Entidades nomeadas
                named_entities = self._extract_named_entities()

                # Palavras-chave
                keywords = self._extract_keywords()

                # Resumo automático
                summary = self._generate_summary()

                return {
                    'main_topics': main_topics,
                    'named_entities': named_entities,
                    'keywords': keywords,
                    'automatic_summary': summary
                }

            def extract_preservation_metadata(self):
                """Extrair metadados de preservação"""
                # Formato do arquivo
                file_format = self._detect_file_format()

                # Qualidade da digitalização
                digitization_quality = self._assess_digitization_quality()

                # Condição física (se aplicável)
                physical_condition = self.metadata.get('physical_condition', 'unknown')

                # Direitos e acesso
                access_rights = self.metadata.get('access_rights', 'unknown')

                return {
                    'file_format': file_format,
                    'digitization_quality': digitization_quality,
                    'physical_condition': physical_condition,
                    'access_rights': access_rights,
                    'preservation_date': datetime.now().isoformat()
                }

            def _count_pages(self):
                """Contar páginas"""
                # Placeholder
                return 1

            def _count_paragraphs(self):
                """Contar parágrafos"""
                paragraphs = self.content.split('\n\n')
                return len([p for p in paragraphs if p.strip()])

            def _count_words(self):
                """Contar palavras"""
                words = self.content.split()
                return len(words)

            def _count_characters(self):
                """Contar caracteres"""
                return len(self.content)

            def _detect_language(self):
                """Detectar linguagem"""
                # Placeholder
                return 'portuguese'

            def _detect_encoding(self):
                """Detectar codificação"""
                # Placeholder
                return 'utf-8'

            def _extract_topics(self):
                """Extrair tópicos principais"""
                # Placeholder
                return ['história', 'documento']

            def _extract_named_entities(self):
                """Extrair entidades nomeadas"""
                # Placeholder
                return ['Entidade1', 'Entidade2']

            def _extract_keywords(self):
                """Extrair palavras-chave"""
                words = self.content.lower().split()
                word_freq = Counter(words)

                return [word for word, freq in word_freq.most_common(10)]

            def _generate_summary(self):
                """Gerar resumo automático"""
                # Placeholder
                return self.content[:200] + '...'

            def _detect_file_format(self):
                """Detectar formato do arquivo"""
                # Placeholder
                return 'text/plain'

            def _assess_digitization_quality(self):
                """Avaliar qualidade da digitalização"""
                # Placeholder
                return 'good'

        metadata_extractor = MetadataExtractor(document_content, document_metadata)
        structural_metadata = metadata_extractor.extract_structural_metadata()
        content_metadata = metadata_extractor.extract_content_metadata()
        preservation_metadata = metadata_extractor.extract_preservation_metadata()

        return {
            'metadata_extractor': metadata_extractor,
            'structural_metadata': structural_metadata,
            'content_metadata': content_metadata,
            'preservation_metadata': preservation_metadata
        }
```

**Preservação Digital e OCR Avançado:**
- Aprimoramento de imagens de documentos históricos
- Sistema OCR para reconhecimento de texto histórico
- Análise de layout de documentos
- Sistema de extração de metadados

---

## 2. MODELAGEM DE EVENTOS HISTÓRICOS

### 2.1 Análise de Redes Sociais Históricas
```python
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

class HistoricalSocialNetworks:
    """
    Análise de redes sociais em contextos históricos
    """

    def __init__(self):
        self.network_models = {}

    def correspondence_network_analysis(self, letters_data):
        """
        Análise de rede de correspondência histórica
        """
        class CorrespondenceNetwork:
            def __init__(self, letters):
                self.letters = letters
                self.network = self._build_correspondence_network()

            def _build_correspondence_network(self):
                """Construir rede de correspondência"""
                G = nx.Graph()

                # Adicionar nós (pessoas)
                people = set()

                for letter in self.letters:
                    people.add(letter['sender'])
                    people.add(letter['recipient'])

                for person in people:
                    G.add_node(person, type='person')

                # Adicionar arestas (correspondências)
                for letter in self.letters:
                    sender = letter['sender']
                    recipient = letter['recipient']
                    date = letter.get('date', 'unknown')
                    topic = letter.get('topic', 'general')

                    if G.has_edge(sender, recipient):
                        # Incrementar peso da aresta existente
                        G[sender][recipient]['weight'] += 1
                        G[sender][recipient]['dates'].append(date)
                        G[sender][recipient]['topics'].append(topic)
                    else:
                        # Criar nova aresta
                        G.add_edge(sender, recipient,
                                 weight=1,
                                 dates=[date],
                                 topics=[topic])

                return G

            def calculate_network_metrics(self):
                """Calcular métricas da rede"""
                # Métricas básicas
                n_nodes = self.network.number_of_nodes()
                n_edges = self.network.number_of_edges()
                density = nx.density(self.network)

                # Centralidade
                degree_centrality = nx.degree_centrality(self.network)
                betweenness_centrality = nx.betweenness_centrality(self.network)
                closeness_centrality = nx.closeness_centrality(self.network)

                # Componentes conectados
                connected_components = list(nx.connected_components(self.network))
                largest_component = max(connected_components, key=len) if connected_components else set()

                return {
                    'n_nodes': n_nodes,
                    'n_edges': n_edges,
                    'density': density,
                    'degree_centrality': degree_centrality,
                    'betweenness_centrality': betweenness_centrality,
                    'closeness_centrality': closeness_centrality,
                    'connected_components': len(connected_components),
                    'largest_component_size': len(largest_component)
                }

            def identify_communities(self):
                """Identificar comunidades na rede"""
                # Usar algoritmo de Louvain para detecção de comunidades
                try:
                    import community as community_louvain
                    partition = community_louvain.best_partition(self.network)

                    communities = defaultdict(list)
                    for person, community_id in partition.items():
                        communities[community_id].append(person)

                    return dict(communities)
                except ImportError:
                    # Fallback: componentes conectados
                    components = list(nx.connected_components(self.network))
                    communities = {i: list(comp) for i, comp in enumerate(components)}
                    return communities

            def temporal_network_evolution(self):
                """Evolução temporal da rede"""
                # Ordenar cartas por data
                sorted_letters = sorted(self.letters, key=lambda x: x.get('date', ''))

                # Construir snapshots temporais
                snapshots = []
                current_network = nx.Graph()

                for letter in sorted_letters:
                    sender = letter['sender']
                    recipient = letter['recipient']

                    # Adicionar nós se não existirem
                    for person in [sender, recipient]:
                        if not current_network.has_node(person):
                            current_network.add_node(person)

                    # Adicionar aresta
                    if current_network.has_edge(sender, recipient):
                        current_network[sender][recipient]['weight'] += 1
                    else:
                        current_network.add_edge(sender, recipient, weight=1)

                    # Salvar snapshot
                    snapshots.append({
                        'date': letter.get('date', 'unknown'),
                        'network': current_network.copy(),
                        'n_nodes': current_network.number_of_nodes(),
                        'n_edges': current_network.number_of_edges()
                    })

                return snapshots

            def topic_based_subnetworks(self):
                """Sub-redes baseadas em tópicos"""
                subnetworks = defaultdict(lambda: nx.Graph())

                for letter in self.letters:
                    sender = letter['sender']
                    recipient = letter['recipient']
                    topic = letter.get('topic', 'general')

                    # Adicionar nós à sub-rede do tópico
                    for person in [sender, recipient]:
                        if not subnetworks[topic].has_node(person):
                            subnetworks[topic].add_node(person)

                    # Adicionar aresta
                    if subnetworks[topic].has_edge(sender, recipient):
                        subnetworks[topic][sender][recipient]['weight'] += 1
                    else:
                        subnetworks[topic].add_edge(sender, recipient, weight=1)

                return dict(subnetworks)

        correspondence_network = CorrespondenceNetwork(letters_data)
        network_metrics = correspondence_network.calculate_network_metrics()
        communities = correspondence_network.identify_communities()
        temporal_evolution = correspondence_network.temporal_network_evolution()
        topic_subnetworks = correspondence_network.topic_based_subnetworks()

        return {
            'correspondence_network': correspondence_network,
            'network_metrics': network_metrics,
            'communities': communities,
            'temporal_evolution': temporal_evolution,
            'topic_subnetworks': topic_subnetworks
        }

    def kinship_network_modeling(self, genealogical_data):
        """
        Modelagem de redes de parentesco histórico
        """
        class KinshipNetwork:
            def __init__(self, genealogical_data):
                self.genealogical_data = genealogical_data
                self.kinship_network = self._build_kinship_network()

            def _build_kinship_network(self):
                """Construir rede de parentesco"""
                G = nx.DiGraph()  # Grafo direcionado (pai->filho)

                # Adicionar nós (pessoas)
                for person in self.genealogical_data['people']:
                    G.add_node(person['id'],
                             name=person['name'],
                             birth_date=person.get('birth_date'),
                             death_date=person.get('death_date'),
                             gender=person.get('gender'))

                # Adicionar arestas (relações de parentesco)
                for relation in self.genealogical_data['relations']:
                    parent_id = relation['parent']
                    child_id = relation['child']
                    relation_type = relation.get('type', 'biological')

                    G.add_edge(parent_id, child_id,
                             relation_type=relation_type,
                             weight=1)

                return G

            def calculate_kinship_metrics(self):
                """Calcular métricas de parentesco"""
                # Graus de parentesco
                in_degrees = dict(self.kinship_network.in_degree())
                out_degrees = dict(self.kinship_network.out_degree())

                # Análise de gerações
                generations = self._identify_generations()

                # Coeficiente de consanguinidade
                consanguinity = self._calculate_consanguinity()

                return {
                    'in_degrees': in_degrees,
                    'out_degrees': out_degrees,
                    'generations': generations,
                    'consanguinity_coefficient': consanguinity
                }

            def _identify_generations(self):
                """Identificar gerações"""
                # Encontrar raízes (pessoas sem pais)
                roots = [node for node in self.kinship_network.nodes()
                        if self.kinship_network.in_degree(node) == 0]

                generations = defaultdict(list)

                for root in roots:
                    generations[0].append(root)

                    # BFS para identificar gerações
                    visited = set([root])
                    queue = [(root, 0)]

                    while queue:
                        current_person, current_gen = queue.pop(0)

                        for child in self.kinship_network.successors(current_person):
                            if child not in visited:
                                visited.add(child)
                                generations[current_gen + 1].append(child)
                                queue.append((child, current_gen + 1))

                return dict(generations)

            def _calculate_consanguinity(self):
                """Calcular coeficiente de consanguinidade médio"""
                # Placeholder: cálculo simplificado
                total_coefficient = 0
                n_pairs = 0

                nodes_list = list(self.kinship_network.nodes())

                for i in range(len(nodes_list)):
                    for j in range(i + 1, len(nodes_list)):
                        person1 = nodes_list[i]
                        person2 = nodes_list[j]

                        # Encontrar ancestral comum mais recente
                        common_ancestors = self._find_common_ancestors(person1, person2)

                        if common_ancestors:
                            # Cálculo simplificado do coeficiente
                            coefficient = 0.5 ** len(common_ancestors)
                            total_coefficient += coefficient
                            n_pairs += 1

                avg_consanguinity = total_coefficient / n_pairs if n_pairs > 0 else 0

                return avg_consanguinity

            def _find_common_ancestors(self, person1, person2):
                """Encontrar ancestrais comuns"""
                ancestors1 = self._get_ancestors(person1)
                ancestors2 = self._get_ancestors(person2)

                common = ancestors1 & ancestors2

                return common

            def _get_ancestors(self, person):
                """Obter ancestrais de uma pessoa"""
                ancestors = set()

                # BFS para cima na árvore genealógica
                queue = [person]
                visited = set()

                while queue:
                    current = queue.pop(0)

                    if current in visited:
                        continue

                    visited.add(current)

                    for parent in self.kinship_network.predecessors(current):
                        if parent not in visited:
                            ancestors.add(parent)
                            queue.append(parent)

                return ancestors

            def inheritance_pattern_analysis(self):
                """Análise de padrões de herança"""
                # Analisar transmissão de propriedades/tradições
                inheritance_patterns = defaultdict(int)

                for node in self.kinship_network.nodes():
                    successors = list(self.kinship_network.successors(node))
                    n_children = len(successors)

                    inheritance_patterns[n_children] += 1

                return dict(inheritance_patterns)

        kinship_network = KinshipNetwork(genealogical_data)
        kinship_metrics = kinship_network.calculate_kinship_metrics()
        inheritance_patterns = kinship_network.inheritance_pattern_analysis()

        return {
            'kinship_network': kinship_network,
            'kinship_metrics': kinship_metrics,
            'inheritance_patterns': inheritance_patterns
        }

    def trade_network_modeling(self, trade_records):
        """
        Modelagem de redes de comércio histórico
        """
        class TradeNetwork:
            def __init__(self, trade_records):
                self.trade_records = trade_records
                self.trade_network = self._build_trade_network()

            def _build_trade_network(self):
                """Construir rede de comércio"""
                G = nx.DiGraph()  # Direcionado (exportador -> importador)

                # Adicionar nós (cidades/regiões)
                locations = set()

                for record in self.trade_records:
                    locations.add(record['exporter'])
                    locations.add(record['importer'])

                for location in locations:
                    G.add_node(location, type='location')

                # Adicionar arestas (fluxos comerciais)
                for record in self.trade_records:
                    exporter = record['exporter']
                    importer = record['importer']
                    goods = record.get('goods', 'general')
                    value = record.get('value', 1)
                    date = record.get('date', 'unknown')

                    if G.has_edge(exporter, importer):
                        # Atualizar aresta existente
                        G[exporter][importer]['weight'] += value
                        G[exporter][importer]['goods'].append(goods)
                        G[exporter][importer]['dates'].append(date)
                    else:
                        # Criar nova aresta
                        G.add_edge(exporter, importer,
                                 weight=value,
                                 goods=[goods],
                                 dates=[date])

                return G

            def calculate_trade_flows(self):
                """Calcular fluxos comerciais"""
                # Importações e exportações por local
                imports = defaultdict(float)
                exports = defaultdict(float)

                for exporter, importer, data in self.trade_network.edges(data=True):
                    value = data['weight']
                    exports[exporter] += value
                    imports[importer] += value

                # Balança comercial
                trade_balance = {}

                for location in self.trade_network.nodes():
                    balance = exports[location] - imports[location]
                    trade_balance[location] = balance

                return {
                    'exports': dict(exports),
                    'imports': dict(imports),
                    'trade_balance': trade_balance
                }

            def identify_trade_routes(self):
                """Identificar rotas comerciais"""
                # Usar algoritmo de caminhos mais curtos
                routes = {}

                for source in self.trade_network.nodes():
                    for target in self.trade_network.nodes():
                        if source != target:
                            try:
                                # Caminho mais curto
                                path = nx.shortest_path(self.trade_network, source, target, weight='weight')
                                path_length = nx.shortest_path_length(self.trade_network, source, target, weight='weight')

                                routes[f"{source}_{target}"] = {
                                    'path': path,
                                    'length': path_length
                                }
                            except nx.NetworkXNoPath:
                                continue

                return routes

            def commodity_flow_analysis(self):
                """Análise de fluxo de commodities"""
                commodity_flows = defaultdict(lambda: defaultdict(float))

                for exporter, importer, data in self.trade_network.edges(data=True):
                    goods_list = data['goods']

                    for good in goods_list:
                        commodity_flows[good][f"{exporter}_{importer}"] += data['weight'] / len(goods_list)

                return {
                    'commodity_flows': dict(commodity_flows),
                    'total_commodities': len(commodity_flows)
                }

        trade_network = TradeNetwork(trade_records)
        trade_flows = trade_network.calculate_trade_flows()
        trade_routes = trade_network.identify_trade_routes()
        commodity_flows = trade_network.commodity_flow_analysis()

        return {
            'trade_network': trade_network,
            'trade_flows': trade_flows,
            'trade_routes': trade_routes,
            'commodity_flows': commodity_flows
        }

    def political_alliance_networks(self, diplomatic_records):
        """
        Redes de alianças políticas históricas
        """
        class PoliticalAllianceNetwork:
            def __init__(self, diplomatic_records):
                self.diplomatic_records = diplomatic_records
                self.alliance_network = self._build_alliance_network()

            def _build_alliance_network(self):
                """Construir rede de alianças políticas"""
                G = nx.Graph()  # Não direcionado (alianças mútuas)

                # Adicionar nós (países/estados)
                entities = set()

                for record in self.diplomatic_records:
                    entities.add(record['entity1'])
                    entities.add(record['entity2'])

                for entity in entities:
                    G.add_node(entity, type='political_entity')

                # Adicionar arestas (alianças)
                for record in self.diplomatic_records:
                    entity1 = record['entity1']
                    entity2 = record['entity2']
                    alliance_type = record.get('alliance_type', 'general')
                    date = record.get('date', 'unknown')
                    strength = record.get('strength', 1)

                    if G.has_edge(entity1, entity2):
                        # Fortalecer aliança existente
                        G[entity1][entity2]['weight'] += strength
                        G[entity1][entity2]['dates'].append(date)
                        G[entity1][entity2]['types'].append(alliance_type)
                    else:
                        # Criar nova aliança
                        G.add_edge(entity1, entity2,
                                 weight=strength,
                                 dates=[date],
                                 types=[alliance_type])

                return G

            def identify_coalitions(self):
                """Identificar coalizões políticas"""
                # Usar algoritmo de comunidades
                try:
                    import community as community_louvain
                    partition = community_louvain.best_partition(self.alliance_network)

                    coalitions = defaultdict(list)
                    for entity, coalition_id in partition.items():
                        coalitions[coalition_id].append(entity)

                    return dict(coalitions)
                except ImportError:
                    # Fallback
                    components = list(nx.connected_components(self.alliance_network))
                    coalitions = {i: list(comp) for i, comp in enumerate(components)}
                    return coalitions

            def power_structure_analysis(self):
                """Análise da estrutura de poder"""
                # Centralidade de poder
                degree_centrality = nx.degree_centrality(self.alliance_network)

                # Centralidade de intermediação (controle de fluxos)
                betweenness_centrality = nx.betweenness_centrality(self.alliance_network)

                # Centralidade de eigenvector (conexões importantes)
                eigenvector_centrality = nx.eigenvector_centrality(self.alliance_network, max_iter=1000)

                # Ranking de poder
                power_ranking = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

                return {
                    'degree_centrality': degree_centrality,
                    'betweenness_centrality': betweenness_centrality,
                    'eigenvector_centrality': eigenvector_centrality,
                    'power_ranking': power_ranking
                }

            def conflict_prediction_model(self):
                """Modelo de predição de conflitos"""
                # Analisar tensões baseadas em alianças
                conflict_risks = {}

                for entity1 in self.alliance_network.nodes():
                    for entity2 in self.alliance_network.nodes():
                        if entity1 != entity2 and not self.alliance_network.has_edge(entity1, entity2):
                            # Calcular risco de conflito baseado em vizinhos comuns
                            common_neighbors = len(list(nx.common_neighbors(self.alliance_network, entity1, entity2)))

                            # Distância na rede
                            try:
                                distance = nx.shortest_path_length(self.alliance_network, entity1, entity2)
                                conflict_risk = 1 / (distance + common_neighbors + 1)
                            except nx.NetworkXNoPath:
                                conflict_risk = 0.5  # Alto risco se não conectados

                            conflict_risks[f"{entity1}_{entity2}"] = conflict_risk

                return conflict_risks

        political_network = PoliticalAllianceNetwork(diplomatic_records)
        coalitions = political_network.identify_coalitions()
        power_structure = political_network.power_structure_analysis()
        conflict_risks = political_network.conflict_prediction_model()

        return {
            'political_network': political_network,
            'coalitions': coalitions,
            'power_structure': power_structure,
            'conflict_risks': conflict_risks
        }
```

**Análise de Redes Sociais Históricas:**
- Análise de rede de correspondência histórica
- Modelagem de redes de parentesco
- Modelagem de redes de comércio
- Redes de alianças políticas

### 2.2 Modelagem de Eventos e Processos Históricos
```python
import numpy as np
from scipy.integrate import odeint
from collections import defaultdict
import matplotlib.pyplot as plt

class HistoricalEventModeling:
    """
    Modelagem de eventos e processos históricos
    """

    def __init__(self):
        self.event_models = {}

    def epidemic_spread_modeling(self, population_data, transmission_parameters):
        """
        Modelagem de propagação de epidemias históricas
        """
        class EpidemicModel:
            def __init__(self, population, parameters):
                self.population = population
                self.parameters = parameters

            def sir_model_simulation(self, initial_conditions, time_span):
                """Simulação do modelo SIR"""
                def sir_equations(y, t):
                    S, I, R = y

                    # Parâmetros
                    beta = self.parameters.get('transmission_rate', 0.3)  # Taxa de transmissão
                    gamma = self.parameters.get('recovery_rate', 0.1)     # Taxa de recuperação
                    mu = self.parameters.get('mortality_rate', 0.02)      # Taxa de mortalidade

                    N = S + I + R  # População total

                    # Equações SIR
                    dS_dt = -beta * S * I / N
                    dI_dt = beta * S * I / N - gamma * I - mu * I
                    dR_dt = gamma * I

                    return [dS_dt, dI_dt, dR_dt]

                # Condições iniciais
                S0 = initial_conditions.get('susceptible', self.population * 0.99)
                I0 = initial_conditions.get('infected', self.population * 0.01)
                R0 = initial_conditions.get('recovered', 0)

                y0 = [S0, I0, R0]

                # Simulação
                t = np.linspace(0, time_span, 1000)
                solution = odeint(sir_equations, y0, t)

                S, I, R = solution.T

                return {
                    'time': t,
                    'susceptible': S,
                    'infected': I,
                    'recovered': R,
                    'total_cases': I + R,
                    'mortality': self.parameters.get('mortality_rate', 0.02) * np.trapz(I, t)
                }

            def seir_model_with_demographics(self, demographic_data):
                """Modelo SEIR com demografia"""
                def seir_demographic_equations(y, t):
                    S, E, I, R = y

                    # Parâmetros demográficos
                    birth_rate = demographic_data.get('birth_rate', 0.02)
                    death_rate = demographic_data.get('death_rate', 0.01)

                    # Parâmetros epidemiológicos
                    beta = self.parameters.get('transmission_rate', 0.3)
                    sigma = self.parameters.get('incubation_rate', 0.2)  # 1/incubação
                    gamma = self.parameters.get('recovery_rate', 0.1)

                    N = S + E + I + R

                    # Equações SEIR com demografia
                    dS_dt = birth_rate * N - beta * S * I / N - death_rate * S
                    dE_dt = beta * S * I / N - sigma * E - death_rate * E
                    dI_dt = sigma * E - gamma * I - death_rate * I
                    dR_dt = gamma * I - death_rate * R

                    return [dS_dt, dE_dt, dI_dt, dR_dt]

                # Condições iniciais
                S0 = demographic_data.get('initial_susceptible', self.population * 0.9)
                E0 = demographic_data.get('initial_exposed', self.population * 0.05)
                I0 = demographic_data.get('initial_infected', self.population * 0.05)
                R0 = demographic_data.get('initial_recovered', 0)

                y0 = [S0, E0, I0, R0]

                # Simulação
                t = np.linspace(0, 365, 1000)  # 1 ano
                solution = odeint(seir_demographic_equations, y0, t)

                S, E, I, R = solution.T

                return {
                    'time': t,
                    'susceptible': S,
                    'exposed': E,
                    'infected': I,
                    'recovered': R,
                    'total_population': S + E + I + R,
                    'reproduction_number': self._calculate_reproduction_number()
                }

            def _calculate_reproduction_number(self):
                """Calcular número básico de reprodução R0"""
                beta = self.parameters.get('transmission_rate', 0.3)
                gamma = self.parameters.get('recovery_rate', 0.1)
                mu = self.parameters.get('mortality_rate', 0.02)

                R0 = beta / (gamma + mu)

                return R0

            def spatial_epidemic_modeling(self, geographical_data):
                """Modelagem espacial da epidemia"""
                # Modelo simplificado com difusão espacial
                def reaction_diffusion_epidemic(y, t):
                    # y contém concentrações S, I, R em diferentes localizações
                    # Implementação simplificada
                    return -0.1 * y  # Decaimento simples

                # Placeholder para modelagem espacial
                return {
                    'spatial_spread': 'implemented',
                    'diffusion_coefficient': geographical_data.get('diffusion_coeff', 0.1),
                    'spatial_heterogeneity': geographical_data.get('heterogeneity', 0.5)
                }

        epidemic_model = EpidemicModel(population_data, transmission_parameters)
        sir_simulation = epidemic_model.sir_model_simulation({}, 365)

        return {
            'epidemic_model': epidemic_model,
            'sir_simulation': sir_simulation
        }

    def economic_cycle_modeling(self, economic_indicators, historical_period):
        """
        Modelagem de ciclos econômicos históricos
        """
        class EconomicCycleModel:
            def __init__(self, indicators, period):
                self.indicators = indicators
                self.period = period

            def business_cycle_analysis(self):
                """Análise de ciclos econômicos"""
                # Usar filtro HP para decompor tendência e ciclo
                gdp_data = self.indicators.get('gdp', [])

                if len(gdp_data) > 0:
                    # Filtro Hodrick-Prescott simplificado
                    lam = 1600  # Parâmetro de suavização

                    n = len(gdp_data)
                    gdp_array = np.array(gdp_data)

                    # Matriz de diferenças segundas
                    D = np.zeros((n-2, n))
                    for i in range(n-2):
                        D[i, i] = 1
                        D[i, i+1] = -2
                        D[i, i+2] = 1

                    # Resolver minimização
                    I = np.eye(n)
                    trend = np.linalg.solve(I + lam * D.T @ D, gdp_array)

                    cycle = gdp_array - trend

                    return {
                        'original_gdp': gdp_data,
                        'trend': trend.tolist(),
                        'cycle': cycle.tolist(),
                        'cycle_amplitude': np.std(cycle),
                        'cycle_period': self._estimate_cycle_period(cycle)
                    }
                else:
                    return {'error': 'Insufficient GDP data'}

            def _estimate_cycle_period(self, cycle_data):
                """Estimar período do ciclo"""
                # Análise de Fourier simplificada
                if len(cycle_data) > 10:
                    fft = np.fft.fft(cycle_data)
                    freqs = np.fft.fftfreq(len(cycle_data))

                    # Encontrar frequência dominante
                    dominant_freq_idx = np.argmax(np.abs(fft[1:])) + 1
                    dominant_period = 1 / abs(freqs[dominant_freq_idx])

                    return dominant_period
                else:
                    return None

            def economic_indicator_correlations(self):
                """Correlações entre indicadores econômicos"""
                indicator_names = list(self.indicators.keys())
                n_indicators = len(indicator_names)

                correlation_matrix = np.zeros((n_indicators, n_indicators))

                for i in range(n_indicators):
                    for j in range(n_indicators):
                        if i != j:
                            data_i = self.indicators[indicator_names[i]]
                            data_j = self.indicators[indicator_names[j]]

                            # Calcular correlação
                            if len(data_i) == len(data_j) and len(data_i) > 1:
                                correlation_matrix[i, j] = np.corrcoef(data_i, data_j)[0, 1]
                            else:
                                correlation_matrix[i, j] = 0

                return {
                    'indicator_names': indicator_names,
                    'correlation_matrix': correlation_matrix,
                    'strongest_correlations': self._find_strongest_correlations(correlation_matrix, indicator_names)
                }

            def _find_strongest_correlations(self, corr_matrix, names):
                """Encontrar correlações mais fortes"""
                strongest = []

                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        corr = corr_matrix[i, j]
                        if abs(corr) > 0.7:  # Correlação forte
                            strongest.append({
                                'indicators': (names[i], names[j]),
                                'correlation': corr
                            })

                return strongest

            def economic_crisis_prediction(self):
                """Predição de crises econômicas"""
                # Modelo simples baseado em indicadores leading
                leading_indicators = ['stock_prices', 'interest_rates', 'unemployment']

                crisis_probability = 0

                for indicator in leading_indicators:
                    if indicator in self.indicators:
                        data = self.indicators[indicator]

                        if len(data) > 1:
                            # Verificar tendência descendente
                            recent_trend = np.polyfit(range(len(data)), data, 1)[0]

                            if recent_trend < -0.1:  # Tendência negativa significativa
                                crisis_probability += 0.3

                return {
                    'crisis_probability': min(crisis_probability, 1.0),
                    'risk_factors': leading_indicators,
                    'early_warning_signals': crisis_probability > 0.5
                }

        economic_model = EconomicCycleModel(economic_indicators, historical_period)
        cycle_analysis = economic_model.business_cycle_analysis()
        correlations = economic_model.economic_indicator_correlations()
        crisis_prediction = economic_model.economic_crisis_prediction()

        return {
            'economic_model': economic_model,
            'cycle_analysis': cycle_analysis,
            'indicator_correlations': correlations,
            'crisis_prediction': crisis_prediction
        }

    def technological_innovation_diffusion(self, innovation_data, social_network):
        """
        Difusão de inovações tecnológicas históricas
        """
        class InnovationDiffusionModel:
            def __init__(self, innovation_data, network):
                self.innovation = innovation_data
                self.network = network

            def bass_diffusion_model(self, market_size, time_span):
                """Modelo de difusão de Bass"""
                def bass_equations(y, t):
                    adopters = y[0]

                    # Parâmetros do modelo de Bass
                    p = self.innovation.get('innovation_coefficient', 0.03)  # Coeficiente de inovação
                    q = self.innovation.get('imitation_coefficient', 0.38)   # Coeficiente de imitação

                    # Equação de Bass
                    potential_adopters = market_size - adopters
                    d_adopters_dt = (p + q * adopters / market_size) * potential_adopters

                    return [d_adopters_dt]

                # Condições iniciais
                adopters_0 = self.innovation.get('initial_adopters', 1)

                # Simulação
                t = np.linspace(0, time_span, 100)
                solution = odeint(bass_equations, [adopters_0], t)

                adopters = solution[:, 0]

                return {
                    'time': t,
                    'cumulative_adopters': adopters,
                    'adoption_rate': np.gradient(adopters, t),
                    'peak_adoption_time': t[np.argmax(np.gradient(adopters, t))],
                    'total_adoption_time': t[np.where(adopters >= 0.95 * market_size)[0][0]] if np.any(adopters >= 0.95 * market_size) else None
                }

            def network_based_diffusion(self):
                """Difusão baseada em rede social"""
                # Modelo de difusão em rede
                adoption_status = {node: False for node in self.network.nodes()}
                adoption_time = {}

                # Adotantes iniciais
                initial_adopters = self.innovation.get('initial_adopters_list', [])

                for adopter in initial_adopters:
                    if adopter in adoption_status:
                        adoption_status[adopter] = True
                        adoption_time[adopter] = 0

                # Simulação de difusão
                t = 0
                max_time = 100

                while t < max_time:
                    new_adopters = []

                    for node in self.network.nodes():
                        if not adoption_status[node]:
                            # Calcular pressão social
                            neighbors = list(self.network.neighbors(node))
                            adopted_neighbors = [n for n in neighbors if adoption_status.get(n, False)]

                            social_pressure = len(adopted_neighbors) / len(neighbors) if neighbors else 0

                            # Probabilidade de adoção
                            adoption_prob = self.innovation.get('adoption_probability', 0.1) * social_pressure

                            if np.random.random() < adoption_prob:
                                new_adopters.append(node)

                    # Atualizar status
                    for adopter in new_adopters:
                        adoption_status[adopter] = True
                        adoption_time[adopter] = t

                    if not new_adopters:
                        break

                    t += 1

                return {
                    'final_adoption_status': adoption_status,
                    'adoption_times': adoption_time,
                    'total_adopters': sum(adoption_status.values()),
                    'diffusion_time': t
                }

            def s_curve_analysis(self):
                """Análise de curva S da adoção"""
                # Placeholder para análise de curva S
                return {
                    'innovation_phase': 'early_adopters',
                    'growth_phase': 'rapid_diffusion',
                    'maturity_phase': 'saturation',
                    's_curve_parameters': {
                        'inflection_point': 0.5,
                        'growth_rate': 0.2
                    }
                }

        innovation_model = InnovationDiffusionModel(innovation_data, social_network)
        bass_diffusion = innovation_model.bass_diffusion_model(1000, 50)

        return {
            'innovation_model': innovation_model,
            'bass_diffusion': bass_diffusion
        }

    def demographic_transition_modeling(self, population_statistics, time_period):
        """
        Modelagem de transições demográficas históricas
        """
        class DemographicTransitionModel:
            def __init__(self, population_stats, period):
                self.population = population_stats
                self.period = period

            def population_growth_model(self):
                """Modelo de crescimento populacional"""
                def logistic_growth(y, t):
                    population = y[0]

                    # Parâmetros
                    r = self.population.get('growth_rate', 0.02)  # Taxa de crescimento
                    K = self.population.get('carrying_capacity', 1000000)  # Capacidade de suporte

                    # Equação logística
                    dP_dt = r * population * (1 - population / K)

                    return [dP_dt]

                # Condições iniciais
                P0 = self.population.get('initial_population', 1000)

                # Simulação
                t = np.linspace(0, 200, 100)  # 200 anos
                solution = odeint(logistic_growth, [P0], t)

                population = solution[:, 0]

                return {
                    'time': t,
                    'population': population,
                    'growth_rate': np.gradient(population, t),
                    'carrying_capacity': self.population.get('carrying_capacity', 1000000),
                    'doubling_time': t[np.where(population >= 2 * P0)[0][0]] if np.any(population >= 2 * P0) else None
                }

            def age_structure_analysis(self):
                """Análise da estrutura etária"""
                age_groups = self.population.get('age_distribution', {})

                if age_groups:
                    # Calcular índices demográficos
                    total_population = sum(age_groups.values())

                    # Proporção de jovens (0-14)
                    young_proportion = sum(count for age, count in age_groups.items() if age <= 14) / total_population

                    # Proporção de idosos (65+)
                    elderly_proportion = sum(count for age, count in age_groups.items() if age >= 65) / total_population

                    # Razão de dependência
                    working_age = sum(count for age, count in age_groups.items() if 15 <= age <= 64)
                    dependency_ratio = (total_population - working_age) / working_age

                    return {
                        'age_distribution': age_groups,
                        'young_proportion': young_proportion,
                        'elderly_proportion': elderly_proportion,
                        'dependency_ratio': dependency_ratio,
                        'demographic_dividend': working_age / total_population
                    }
                else:
                    return {'error': 'No age distribution data available'}

            def mortality_transition_model(self):
                """Modelo de transição de mortalidade"""
                # Modelo simplificado de transição epidemiológica
                time_periods = np.linspace(self.period[0], self.period[1], 50)

                # Mortalidade infantil (declínio exponencial)
                infant_mortality = 200 * np.exp(-0.05 * (time_periods - time_periods[0]))

                # Expectativa de vida (aumento logístico)
                life_expectancy = 40 + 40 / (1 + np.exp(-0.1 * (time_periods - np.mean(time_periods))))

                return {
                    'time_periods': time_periods,
                    'infant_mortality': infant_mortality,
                    'life_expectancy': life_expectancy,
                    'transition_speed': 0.05  # Velocidade de transição
                }

            def urbanization_model(self):
                """Modelo de urbanização"""
                # Modelo de crescimento urbano
                initial_urban = self.population.get('initial_urban_population', 0.1)
                urban_growth_rate = self.population.get('urban_growth_rate', 0.03)

                t = np.linspace(0, 100, 50)  # 100 anos
                urban_fraction = initial_urban * np.exp(urban_growth_rate * t)

                # Limitar a 1.0
                urban_fraction = np.clip(urban_fraction, 0, 1)

                return {
                    'time': t,
                    'urban_fraction': urban_fraction,
                    'urban_population': urban_fraction * self.population.get('total_population', 100000),
                    'urbanization_rate': urban_growth_rate
                }

        demographic_model = DemographicTransitionModel(population_statistics, time_period)
        population_growth = demographic_model.population_growth_model()
        age_structure = demographic_model.age_structure_analysis()
        mortality_transition = demographic_model.mortality_transition_model()
        urbanization = demographic_model.urbanization_model()

        return {
            'demographic_model': demographic_model,
            'population_growth': population_growth,
            'age_structure': age_structure,
            'mortality_transition': mortality_transition,
            'urbanization': urbanization
        }

    def cultural_evolution_modeling(self, cultural_artifacts, transmission_network):
        """
        Modelagem da evolução cultural histórica
        """
        class CulturalEvolutionModel:
            def __init__(self, artifacts, network):
                self.artifacts = artifacts
                self.network = network

            def cultural_transmission_model(self):
                """Modelo de transmissão cultural"""
                # Modelo de transmissão vertical/obliqua/horizontal
                transmission_types = ['vertical', 'oblique', 'horizontal']

                transmission_rates = {}

                for t_type in transmission_types:
                    # Taxa baseada no tipo de transmissão
                    if t_type == 'vertical':
                        rate = 0.8  # Pais para filhos
                    elif t_type == 'oblique':
                        rate = 0.6  # Anciãos para jovens
                    else:  # horizontal
                        rate = 0.4  # Pares

                    transmission_rates[t_type] = rate

                return {
                    'transmission_types': transmission_types,
                    'transmission_rates': transmission_rates,
                    'dominant_transmission': max(transmission_rates, key=transmission_rates.get)
                }

            def meme_evolution_simulation(self):
                """Simulação de evolução de memes culturais"""
                # Representar artefatos culturais como "memes"
                memes = {artifact['id']: artifact for artifact in self.artifacts}

                # Simulação de competição entre memes
                fitness_scores = {}

                for meme_id, meme in memes.items():
                    # Fitness baseado em popularidade e adaptabilidade
                    popularity = meme.get('popularity', 1)
                    adaptability = meme.get('adaptability', 0.5)

                    fitness = popularity * adaptability
                    fitness_scores[meme_id] = fitness

                # Evolução temporal
                time_steps = 50
                meme_populations = {meme_id: [meme.get('initial_population', 10)] for meme_id in memes.keys()}

                for t in range(1, time_steps):
                    for meme_id in memes.keys():
                        # Crescimento baseado em fitness
                        growth_rate = 0.1 * fitness_scores[meme_id]

                        # Competição com outros memes
                        competition_factor = 1 - sum(fitness_scores.values()) / (len(memes) * 10)

                        new_population = meme_populations[meme_id][-1] * (1 + growth_rate * competition_factor)
                        meme_populations[meme_id].append(new_population)

                return {
                    'meme_fitness': fitness_scores,
                    'meme_evolution': meme_populations,
                    'surviving_memes': [meme_id for meme_id, pop in meme_populations.items() if pop[-1] > 1]
                }

            def cultural_diversity_index(self):
                """Índice de diversidade cultural"""
                # Medir diversidade baseada em artefatos
                artifact_types = [artifact.get('type', 'unknown') for artifact in self.artifacts]
                type_counts = Counter(artifact_types)

                # Índice de Shannon
                total_artifacts = len(self.artifacts)
                diversity_index = 0

                for count in type_counts.values():
                    p = count / total_artifacts
                    diversity_index -= p * np.log(p)

                return {
                    'artifact_types': dict(type_counts),
                    'diversity_index': diversity_index,
                    'dominant_type': max(type_counts, key=type_counts.get)
                }

        cultural_model = CulturalEvolutionModel(cultural_artifacts, transmission_network)
        transmission_model = cultural_model.cultural_transmission_model()
        meme_evolution = cultural_model.meme_evolution_simulation()
        diversity_index = cultural_model.cultural_diversity_index()

        return {
            'cultural_model': cultural_model,
            'transmission_model': transmission_model,
            'meme_evolution': meme_evolution,
            'cultural_diversity': diversity_index
        }
```

**Modelagem de Eventos e Processos Históricos:**
- Modelagem de propagação de epidemias históricas
- Modelagem de ciclos econômicos históricos
- Difusão de inovações tecnológicas
- Modelagem de transições demográficas
- Modelagem da evolução cultural

---

## 3. VISUALIZAÇÃO E INTERFACES INTERATIVAS

### 3.1 Visualização de Dados Históricos
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import networkx as nx

class HistoricalDataVisualization:
    """
    Visualização de dados históricos
    """

    def __init__(self):
        self.visualization_styles = {}

    def timeline_visualization(self, historical_events, time_range):
        """
        Visualização de linha do tempo interativa
        """
        class TimelineVisualizer:
            def __init__(self, events, time_range):
                self.events = events
                self.time_range = time_range

            def create_interactive_timeline(self):
                """Criar linha do tempo interativa"""
                # Organizar eventos por data
                sorted_events = sorted(self.events, key=lambda x: x.get('date', ''))

                # Criar figura Plotly
                fig = go.Figure()

                # Adicionar eventos como pontos na linha do tempo
                dates = []
                descriptions = []
                categories = []

                for event in sorted_events:
                    if 'date' in event:
                        dates.append(event['date'])
                        descriptions.append(event.get('description', ''))
                        categories.append(event.get('category', 'general'))

                # Scatter plot da linha do tempo
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=[0] * len(dates),  # Todos na mesma linha
                    mode='markers+text',
                    text=[f"{cat}: {desc[:50]}..." for cat, desc in zip(categories, descriptions)],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color=[self._category_color(cat) for cat in categories],
                        symbol='circle'
                    ),
                    name='Eventos Históricos'
                ))

                # Configurar layout
                fig.update_layout(
                    title='Linha do Tempo Interativa de Eventos Históricos',
                    xaxis_title='Data',
                    yaxis=dict(showticklabels=False, showgrid=False),
                    showlegend=False,
                    hovermode='x unified'
                )

                return fig

            def _category_color(self, category):
                """Atribuir cores por categoria"""
                color_map = {
                    'political': 'red',
                    'economic': 'green',
                    'social': 'blue',
                    'technological': 'orange',
                    'cultural': 'purple',
                    'general': 'gray'
                }

                return color_map.get(category, 'gray')

            def create_timeline_animation(self):
                """Criar animação da linha do tempo"""
                # Animação mostrando eventos ao longo do tempo
                frames = []

                sorted_events = sorted(self.events, key=lambda x: x.get('date', ''))

                for i in range(len(sorted_events)):
                    frame_events = sorted_events[:i+1]

                    frame = go.Frame(
                        data=[
                            go.Scatter(
                                x=[event['date'] for event in frame_events],
                                y=[0] * len(frame_events),
                                mode='markers',
                                marker=dict(size=10, color='blue')
                            )
                        ],
                        name=f"Frame {i}"
                    )
                    frames.append(frame)

                # Figura base
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=[sorted_events[0]['date']],
                            y=[0],
                            mode='markers',
                            marker=dict(size=10, color='blue')
                        )
                    ],
                    frames=frames
                )

                fig.update_layout(
                    title='Animação da Linha do Tempo Histórica',
                    xaxis_title='Data',
                    yaxis=dict(showticklabels=False),
                    updatemenus=[
                        dict(
                            type='buttons',
                            buttons=[
                                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=500, redraw=True), mode='immediate')]),
                                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                            ]
                        )
                    ]
                )

                return fig

        timeline_viz = TimelineVisualizer(historical_events, time_range)
        interactive_timeline = timeline_viz.create_interactive_timeline()

        return {
            'timeline_visualizer': timeline_viz,
            'interactive_timeline': interactive_timeline
        }

    def network_visualization(self, historical_network, layout_algorithm='force_directed'):
        """
        Visualização de redes históricas
        """
        class NetworkVisualizer:
            def __init__(self, network, layout):
                self.network = network
                self.layout = layout

            def create_network_visualization(self):
                """Criar visualização de rede"""
                # Calcular layout
                if self.layout == 'force_directed':
                    pos = nx.spring_layout(self.network, k=1, iterations=50)
                elif self.layout == 'circular':
                    pos = nx.circular_layout(self.network)
                else:
                    pos = nx.random_layout(self.network)

                # Preparar dados para Plotly
                edge_x = []
                edge_y = []

                for edge in self.network.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                # Nós
                node_x = [pos[node][0] for node in self.network.nodes()]
                node_y = [pos[node][1] for node in self.network.nodes()]
                node_text = [str(node) for node in self.network.nodes()]

                # Centralidade para tamanho dos nós
                degree_centrality = nx.degree_centrality(self.network)
                node_sizes = [degree_centrality[node] * 50 + 10 for node in self.network.nodes()]

                # Criar figura
                fig = go.Figure()

                # Adicionar arestas
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    name='Conexões'
                ))

                # Adicionar nós
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition='top center',
                    marker=dict(
                        size=node_sizes,
                        color=list(degree_centrality.values()),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Centralidade')
                    ),
                    name='Entidades'
                ))

                fig.update_layout(
                    title='Visualização de Rede Histórica',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )

                return fig

            def create_temporal_network_animation(self, temporal_networks):
                """Criar animação de rede temporal"""
                frames = []

                for i, network_snapshot in enumerate(temporal_networks):
                    # Layout para snapshot
                    pos = nx.spring_layout(network_snapshot, k=1, iterations=20)

                    # Dados dos nós e arestas
                    node_x = [pos[node][0] for node in network_snapshot.nodes()]
                    node_y = [pos[node][1] for node in network_snapshot.nodes()]

                    edge_x = []
                    edge_y = []

                    for edge in network_snapshot.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    frame = go.Frame(
                        data=[
                            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')),
                            go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='blue'))
                        ],
                        name=f"Tempo {i}"
                    )
                    frames.append(frame)

                # Figura base
                fig = go.Figure(
                    data=[
                        go.Scatter(x=[], y=[], mode='lines'),
                        go.Scatter(x=[], y=[], mode='markers')
                    ],
                    frames=frames
                )

                fig.update_layout(
                    title='Evolução Temporal da Rede Histórica',
                    updatemenus=[
                        dict(
                            type='buttons',
                            buttons=[
                                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=1000, redraw=True), mode='immediate')])
                            ]
                        )
                    ]
                )

                return fig

        network_viz = NetworkVisualizer(historical_network, layout_algorithm)
        network_visualization = network_viz.create_network_visualization()

        return {
            'network_visualizer': network_viz,
            'network_visualization': network_visualization
        }

    def geographical_visualization(self, historical_data, geographical_bounds):
        """
        Visualização geográfica de dados históricos
        """
        class GeographicalVisualizer:
            def __init__(self, data, bounds):
                self.data = data
                self.bounds = bounds

            def create_choropleth_map(self):
                """Criar mapa coroplético"""
                # Dados de exemplo para regiões
                regions = ['Europe', 'Asia', 'Africa', 'Americas', 'Oceania']
                values = np.random.rand(len(regions)) * 100

                fig = go.Figure(data=go.Choropleth(
                    locations=regions,
                    z=values,
                    locationmode='country names',
                    colorscale='Viridis',
                    colorbar_title='Valor Histórico'
                ))

                fig.update_layout(
                    title='Mapa Coroplético de Dados Históricos',
                    geo=dict(
                        showframe=False,
                        showcoastlines=True,
                        projection_type='equirectangular'
                    )
                )

                return fig

            def create_bubble_map(self):
                """Criar mapa de bolhas para eventos pontuais"""
                # Eventos históricos com coordenadas
                events = [
                    {'name': 'Evento 1', 'lat': 48.8566, 'lon': 2.3522, 'size': 10},
                    {'name': 'Evento 2', 'lat': 51.5074, 'lon': -0.1278, 'size': 15},
                    {'name': 'Evento 3', 'lat': 40.7128, 'lon': -74.0060, 'size': 20}
                ]

                latitudes = [event['lat'] for event in events]
                longitudes = [event['lon'] for event in events]
                sizes = [event['size'] for event in events]
                names = [event['name'] for event in events]

                fig = go.Figure(data=go.Scattergeo(
                    lat=latitudes,
                    lon=longitudes,
                    text=names,
                    marker=dict(
                        size=sizes,
                        color=sizes,
                        colorscale='Viridis',
                        showscale=True,
                        sizemode='diameter'
                    )
                ))

                fig.update_layout(
                    title='Mapa de Eventos Históricos',
                    geo=dict(
                        scope='world',
                        showland=True,
                        landcolor='rgb(243, 243, 243)',
                        countrycolor='rgb(204, 204, 204)'
                    )
                )

                return fig

        geo_viz = GeographicalVisualizer(historical_data, geographical_bounds)
        choropleth_map = geo_viz.create_choropleth_map()

        return {
            'geographical_visualizer': geo_viz,
            'choropleth_map': choropleth_map
        }

    def statistical_visualization(self, historical_statistics, visualization_type):
        """
        Visualização estatística de dados históricos
        """
        class StatisticalVisualizer:
            def __init__(self, statistics, viz_type):
                self.statistics = statistics
                self.viz_type = viz_type

            def create_time_series_plot(self):
                """Criar gráfico de séries temporais"""
                # Dados de exemplo
                years = np.arange(1800, 1900, 10)
                population = 1000000 * np.exp(0.01 * (years - 1800))
                gdp = 1000000 * (1.02 ** (years - 1800))

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=years,
                    y=population,
                    mode='lines+markers',
                    name='População',
                    line=dict(color='blue')
                ))

                fig.add_trace(go.Scatter(
                    x=years,
                    y=gdp,
                    mode='lines+markers',
                    name='PIB',
                    line=dict(color='red'),
                    yaxis='y2'
                ))

                fig.update_layout(
                    title='Séries Temporais Históricas',
                    xaxis_title='Ano',
                    yaxis_title='População',
                    yaxis2=dict(
                        title='PIB',
                        overlaying='y',
                        side='right'
                    )
                )

                return fig

            def create_correlation_matrix(self):
                """Criar matriz de correlação"""
                variables = ['População', 'PIB', 'Tecnologia', 'Educação', 'Saúde']
                n_vars = len(variables)

                # Matriz de correlação de exemplo
                correlation_matrix = np.random.rand(n_vars, n_vars)
                correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
                np.fill_diagonal(correlation_matrix, 1)

                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix,
                    x=variables,
                    y=variables,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))

                fig.update_layout(
                    title='Matriz de Correlação de Variáveis Históricas',
                    xaxis_title='Variáveis',
                    yaxis_title='Variáveis'
                )

                return fig

            def create_distribution_plots(self):
                """Criar gráficos de distribuição"""
                # Dados de exemplo
                data1 = np.random.normal(50, 10, 1000)
                data2 = np.random.normal(60, 15, 1000)

                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=data1,
                    name='Período 1',
                    opacity=0.7,
                    nbinsx=30
                ))

                fig.add_trace(go.Histogram(
                    x=data2,
                    name='Período 2',
                    opacity=0.7,
                    nbinsx=30
                ))

                fig.update_layout(
                    title='Distribuição de Variáveis Históricas',
                    xaxis_title='Valor',
                    yaxis_title='Frequência',
                    barmode='overlay'
                )

                return fig

        stat_viz = StatisticalVisualizer(historical_statistics, visualization_type)
        time_series = stat_viz.create_time_series_plot()

        return {
            'statistical_visualizer': stat_viz,
            'time_series_plot': time_series
        }

    def create_interactive_dashboard(self, historical_datasets):
        """
        Criar dashboard interativo para exploração histórica
        """
        class HistoricalDashboard:
            def __init__(self, datasets):
                self.datasets = datasets

            def create_comprehensive_dashboard(self):
                """Criar dashboard abrangente"""
                # Usar subplots para múltiplas visualizações
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Linha do Tempo', 'Rede Social', 'Séries Temporais', 'Distribuições'),
                    specs=[
                        [{"type": "scatter"}, {"type": "scatter"}],
                        [{"type": "scatter"}, {"type": "histogram"}]
                    ]
                )

                # Adicionar dados de exemplo a cada subplot
                # Linha do tempo
                fig.add_trace(
                    go.Scatter(x=[1800, 1850, 1900], y=[0, 0, 0], mode='markers',
                              marker=dict(size=20, color='red')),
                    row=1, col=1
                )

                # Rede (simplificado)
                fig.add_trace(
                    go.Scatter(x=[1, 2, 3], y=[1, 2, 1], mode='markers+lines',
                              marker=dict(size=10), line=dict(color='blue')),
                    row=1, col=2
                )

                # Séries temporais
                years = np.arange(1800, 1900, 10)
                values = np.random.rand(len(years))
                fig.add_trace(
                    go.Scatter(x=years, y=values, mode='lines'),
                    row=2, col=1
                )

                # Distribuição
                data = np.random.normal(50, 10, 1000)
                fig.add_trace(
                    go.Histogram(x=data, nbinsx=30),
                    row=2, col=2
                )

                fig.update_layout(
                    title='Dashboard Interativo de História Digital',
                    height=800
                )

                return fig

        dashboard = HistoricalDashboard(historical_datasets)
        comprehensive_dashboard = dashboard.create_comprehensive_dashboard()

        return {
            'historical_dashboard': dashboard,
            'comprehensive_dashboard': comprehensive_dashboard
        }
```

**Visualização de Dados Históricos:**
- Visualização de linha do tempo interativa
- Visualização de redes históricas
- Visualização geográfica de dados históricos
- Visualização estatística de dados históricos
- Dashboard interativo para exploração histórica

---

## 4. CONSIDERAÇÕES FINAIS

A história digital representa uma revolução metodológica na historiografia, combinando rigor acadêmico tradicional com ferramentas computacionais modernas. Os modelos apresentados fornecem uma base sólida para:

1. **Análise Computacional**: Processamento de texto, modelagem de tópicos e análise de sentimento
2. **Preservação Digital**: OCR avançado, análise de layout e extração de metadados
3. **Modelagem de Eventos**: Redes sociais, epidemias, ciclos econômicos e evolução cultural
4. **Visualização Interativa**: Linhas do tempo, redes, mapas geográficos e dashboards

**Próximos Passos Recomendados**:
1. Dominar técnicas de processamento de linguagem natural para textos históricos
2. Explorar métodos de preservação digital e OCR para documentos antigos
3. Aplicar análise de redes para compreender conexões históricas
4. Desenvolver visualizações interativas para comunicação de resultados
5. Integrar múltiplas fontes de dados para narrativas históricas abrangentes

---

*Documento preparado para fine-tuning de IA em História Digital*
*Versão 1.0 - Preparado para implementação prática*
