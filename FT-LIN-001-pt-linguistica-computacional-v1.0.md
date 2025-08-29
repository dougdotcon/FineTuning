# FT-LIN-001: Fine-Tuning para IA em Linguística Computacional

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados em linguística computacional, integrando processamento de linguagem natural, análise sintática e semântica, fonética computacional, tradução automática e linguística forense com princípios da teoria linguística e aplicações práticas em português.

### Contexto Filosófico
A linguística computacional representa a interface entre a teoria linguística e a implementação computacional, buscando modelar formalmente os mecanismos subjacentes à linguagem humana. Esta abordagem reconhece que a linguagem é um sistema complexo de regras, padrões e usos que pode ser compreendido através de algoritmos formais.

### Metodologia de Aprendizado Recomendada
1. **Fundamentos Linguísticos**: Compreensão de níveis de análise linguística
2. **Modelos Computacionais**: Desenvolvimento de algoritmos para processamento de linguagem
3. **Aplicações Práticas**: Implementação em tarefas reais de PLN
4. **Avaliação de Sistemas**: Métricas e protocolos de avaliação
5. **Integração Multinível**: Conexão entre teoria e aplicação prática

---

## 1. PROCESSAMENTO DE LINGUAGEM NATURAL

### 1.1 Tokenização e Normalização
```python
import re
import unicodedata
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

class TextPreprocessing:
    """
    Técnicas de pré-processamento de texto para português
    """

    def __init__(self):
        self.contractions = {
            "não": ["n", "não"],
            "estou": ["tô", "estou"],
            "porque": ["pq", "porque"],
            "para": ["pra", "para"],
            "qualquer": ["qlqr", "qualquer"],
            "agora": ["agr", "agora"]
        }

        self.abbreviations = {
            "dr.": "doutor",
            "sra.": "senhora",
            "sr.": "senhor",
            "prof.": "professor",
            "av.": "avenida",
            "r.": "rua"
        }

    def advanced_tokenization(self, text, language='portuguese'):
        """
        Tokenização avançada para português
        """
        class AdvancedTokenizer:
            def __init__(self, lang='portuguese'):
                self.language = lang
                self.special_tokens = {
                    'portuguese': {
                        'clitics': ['me', 'te', 'se', 'lhe', 'nos', 'vos', 'lhes', 'o', 'a', 'os', 'as', 'lo', 'la', 'los', 'las'],
                        'contractions': ['do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas', 'ao', 'à', 'aos', 'às', 'pelo', 'pela', 'pelos', 'pelas']
                    }
                }

            def tokenize_with_clitics(self, text):
                """Tokenização considerando clíticos em português"""
                # Padrões para clíticos
                clitic_patterns = [
                    r'(\w+)-(me|te|se|lhe|nos|vos|lhes|o|a|os|as|lo|la|los|las)\b',  # clíticos finais
                    r'(\w+)-(mo|to|so|lho|no|vo|lho|o|a|os|as|lo|la|los|las)\b',  # clíticos com assimilação
                ]

                tokens = []

                # Aplicar padrões de clíticos
                for pattern in clitic_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if len(match) == 2:
                            base_word, clitic = match
                            tokens.extend([base_word, clitic])

                # Tokenização padrão para o restante
                remaining_text = re.sub(r'(\w+)-(me|te|se|lhe|nos|vos|lhes|o|a|os|as|lo|la|los|las)\b',
                                      '', text, flags=re.IGNORECASE)
                remaining_tokens = word_tokenize(remaining_text, language=self.language)
                tokens.extend(remaining_tokens)

                return [token for token in tokens if token.strip()]

            def handle_compound_words(self, text):
                """Tratamento de palavras compostas em português"""
                compound_patterns = [
                    r'(\w+)-(se|o|a|os|as|nos|nas|lhe|lhes|vos)',  # contrações
                    r'(\w+)-(\w+)',  # compostos com hífen
                ]

                compounds = []

                for pattern in compound_patterns:
                    matches = re.findall(pattern, text)
                    compounds.extend([''.join(match) for match in matches])

                return compounds

            def multilingual_tokenization(self, text, languages=['portuguese', 'english']):
                """Tokenização multilíngue"""
                # Detectar idioma de cada segmento
                segments = []
                current_segment = ""
                current_lang = None

                words = word_tokenize(text)

                for word in words:
                    # Detecção simples de idioma baseada em sufixos
                    if self._detect_language(word) != current_lang:
                        if current_segment:
                            segments.append((current_segment.strip(), current_lang))
                        current_segment = word
                        current_lang = self._detect_language(word)
                    else:
                        current_segment += " " + word

                if current_segment:
                    segments.append((current_segment.strip(), current_lang))

                return segments

            def _detect_language(self, word):
                """Detecção simples de idioma"""
                pt_indicators = ['ão', 'ões', 'inha', 'inho', 'mente', 'ção', 'são']
                en_indicators = ['tion', 'ing', 'ment', 'ness', 'ship']

                word_lower = word.lower()

                pt_score = sum(1 for indicator in pt_indicators if indicator in word_lower)
                en_score = sum(1 for indicator in en_indicators if indicator in word_lower)

                return 'portuguese' if pt_score > en_score else 'english'

        tokenizer = AdvancedTokenizer(language)
        clitic_tokens = tokenizer.tokenize_with_clitics(text)
        compounds = tokenizer.handle_compound_words(text)

        return {
            'advanced_tokenizer': tokenizer,
            'clitic_tokens': clitic_tokens,
            'compound_words': compounds,
            'multilingual_segments': tokenizer.multilingual_tokenization(text)
        }

    def text_normalization(self, text, normalization_type='standard'):
        """
        Normalização de texto para português
        """
        class TextNormalizer:
            def __init__(self, norm_type='standard'):
                self.norm_type = norm_type

            def unicode_normalization(self, text):
                """Normalização Unicode"""
                # Normalização NFC (Canonical Composition)
                normalized = unicodedata.normalize('NFC', text)

                # Converter caracteres especiais
                char_map = {
                    'ç': 'c',
                    'ã': 'a',
                    'õ': 'o',
                    'á': 'a',
                    'é': 'e',
                    'í': 'i',
                    'ó': 'o',
                    'ú': 'u',
                    'â': 'a',
                    'ê': 'e',
                    'î': 'i',
                    'ô': 'o',
                    'û': 'u'
                }

                if self.norm_type == 'ascii':
                    for char, replacement in char_map.items():
                        normalized = normalized.replace(char, replacement)

                return normalized

            def case_normalization(self, text):
                """Normalização de maiúsculas/minúsculas"""
                if self.norm_type == 'lower':
                    return text.lower()
                elif self.norm_type == 'upper':
                    return text.upper()
                elif self.norm_type == 'title':
                    return text.title()
                else:
                    return text

            def number_normalization(self, text):
                """Normalização de números"""
                # Converter números por extenso para dígitos
                number_words = {
                    'zero': '0', 'um': '1', 'dois': '2', 'três': '3', 'quatro': '4',
                    'cinco': '5', 'seis': '6', 'sete': '7', 'oito': '8', 'nove': '9',
                    'dez': '10', 'vinte': '20', 'trinta': '30', 'quarenta': '40',
                    'cinquenta': '50', 'sessenta': '60', 'setenta': '70', 'oitenta': '80', 'noventa': '90'
                }

                for word, digit in number_words.items():
                    text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)

                return text

            def abbreviation_expansion(self, text):
                """Expansão de abreviações"""
                abbreviations = {
                    'dr.': 'doutor',
                    'dra.': 'doutora',
                    'sr.': 'senhor',
                    'sra.': 'senhora',
                    'prof.': 'professor',
                    'av.': 'avenida',
                    'r.': 'rua',
                    'tel.': 'telefone',
                    'cel.': 'celular'
                }

                for abbr, expansion in abbreviations.items():
                    text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)

                return text

            def complete_normalization(self, text):
                """Normalização completa"""
                text = self.unicode_normalization(text)
                text = self.case_normalization(text)
                text = self.number_normalization(text)
                text = self.abbreviation_expansion(text)

                return text

        normalizer = TextNormalizer(normalization_type)
        normalized_text = normalizer.complete_normalization(text)

        return {
            'text_normalizer': normalizer,
            'normalized_text': normalized_text
        }

    def stop_words_handling(self, text, language='portuguese'):
        """
        Tratamento de palavras vazias (stop words) para português
        """
        class StopWordsHandler:
            def __init__(self, lang='portuguese'):
                self.language = lang
                self.stop_words = self._load_stop_words()

            def _load_stop_words(self):
                """Carregar lista de stop words para português"""
                # Lista abrangente de stop words em português
                pt_stop_words = [
                    'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas',
                    'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas',
                    'por', 'para', 'pelo', 'pela', 'pelos', 'pelas', 'ao', 'à', 'aos', 'às',
                    'com', 'sem', 'sob', 'sobre', 'entre', 'até', 'desde', 'durante',
                    'que', 'quem', 'qual', 'quais', 'quanto', 'quantos', 'quanta', 'quantas',
                    'seu', 'sua', 'seus', 'suas', 'meu', 'minha', 'meus', 'minhas',
                    'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas',
                    'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
                    'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'isso', 'aquilo',
                    'e', 'ou', 'mas', 'porém', 'contudo', 'todavia', 'entretanto',
                    'não', 'nem', 'também', 'já', 'ainda', 'sempre', 'nunca', 'agora',
                    'aqui', 'ali', 'lá', 'acolá', 'além', 'perto', 'longe',
                    'muito', 'pouco', 'tanto', 'quanto', 'mais', 'menos', 'muito', 'pouco',
                    'todo', 'toda', 'todos', 'todas', 'algum', 'alguma', 'alguns', 'algumas',
                    'nenhum', 'nenhuma', 'nenhuns', 'nenhumas', 'outro', 'outra', 'outros', 'outras'
                ]

                return set(pt_stop_words)

            def remove_stop_words(self, text):
                """Remover stop words do texto"""
                words = word_tokenize(text, language=self.language)
                filtered_words = [word for word in words if word.lower() not in self.stop_words]

                return ' '.join(filtered_words)

            def identify_domain_stop_words(self, corpus, domain_threshold=0.8):
                """Identificar stop words específicas de domínio"""
                # Tokenizar corpus
                all_words = []
                for doc in corpus:
                    words = word_tokenize(doc, language=self.language)
                    all_words.extend([word.lower() for word in words])

                # Calcular frequência
                word_freq = Counter(all_words)
                total_words = len(all_words)

                # Palavras muito frequentes em domínio específico
                domain_stop_words = [
                    word for word, freq in word_freq.items()
                    if freq / total_words > domain_threshold
                ]

                return domain_stop_words

            def weighted_stop_words_removal(self, text, weights=None):
                """Remoção ponderada de stop words"""
                if weights is None:
                    weights = {
                        'common': 1.0,    # Stop words comuns
                        'domain': 1.5,    # Stop words de domínio
                        'context': 0.5    # Dependente do contexto
                    }

                words = word_tokenize(text, language=self.language)
                filtered_words = []

                for word in words:
                    word_lower = word.lower()

                    weight = 0

                    if word_lower in self.stop_words:
                        weight += weights['common']

                    # Verificar se é palavra de domínio frequente
                    if hasattr(self, 'domain_stop_words') and word_lower in self.domain_stop_words:
                        weight += weights['domain']

                    # Critérios contextuais
                    if len(word) <= 2:  # Palavras muito curtas
                        weight += weights['context']

                    # Decidir se remove baseado no peso
                    if weight < 1.5:  # Threshold
                        filtered_words.append(word)

                return ' '.join(filtered_words)

        stop_handler = StopWordsHandler(language)
        filtered_text = stop_handler.remove_stop_words(text)

        return {
            'stop_words_handler': stop_handler,
            'filtered_text': filtered_text
        }

    def stemming_lemmatization_portuguese(self, text, method='stemming'):
        """
        Stemming e lematização para português
        """
        class PortugueseMorphology:
            def __init__(self, method='stemming'):
                self.method = method

            def rslp_stemmer(self, text):
                """Stemmer RSLP para português"""
                # Implementação simplificada do stemmer RSLP
                stems = []

                words = word_tokenize(text, language='portuguese')

                for word in words:
                    stem = self._apply_rslp_rules(word.lower())
                    stems.append(stem)

                return stems

            def _apply_rslp_rules(self, word):
                """Aplicar regras do stemmer RSLP"""
                # Regras simplificadas
                rules = [
                    (r'ções$', 'ção'),     # plurais
                    (r'cionários$', 'cionário'),
                    (r'mente$', ''),       # advérbios
                    (r'idades$', 'idade'), # substantivos
                    (r'ismos$', 'ismo'),
                    (r'istas$', 'ista'),
                    (r'ários$', 'ário'),
                    (r'áveis$', 'ável'),
                    (r'íveis$', 'ível'),
                    (r'istas$', 'ista'),
                    (r'osos$', 'oso'),
                    (r'osas$', 'osa')
                ]

                for pattern, replacement in rules:
                    word = re.sub(pattern, replacement, word)

                return word

            def portuguese_lemmatizer(self, text):
                """Lematizador para português"""
                # Dicionário de lemas (simplificado)
                lemma_dict = {
                    'correu': 'correr',
                    'correu': 'correr',
                    'corre': 'correr',
                    'corrido': 'correr',
                    'casa': 'casa',
                    'casas': 'casa',
                    'feliz': 'feliz',
                    'felizes': 'feliz',
                    'bom': 'bom',
                    'bons': 'bom',
                    'boa': 'bom',
                    'boas': 'bom'
                }

                words = word_tokenize(text, language='portuguese')
                lemmas = []

                for word in words:
                    lemma = lemma_dict.get(word.lower(), word.lower())
                    lemmas.append(lemma)

                return lemmas

            def morphological_analysis(self, text):
                """Análise morfológica"""
                words = word_tokenize(text, language='portuguese')
                analysis = []

                for word in words:
                    morph_features = self._extract_morphological_features(word)
                    analysis.append({
                        'word': word,
                        'stem': self._apply_rslp_rules(word.lower()),
                        'lemma': self.portuguese_lemmatizer(' '.join([word]))[0],
                        'morphological_features': morph_features
                    })

                return analysis

            def _extract_morphological_features(self, word):
                """Extrair características morfológicas"""
                features = {}

                # Sufixos comuns em português
                if word.endswith(('ção', 'são', 'gem')):
                    features['type'] = 'substantivo'
                    features['gender'] = 'feminino' if word.endswith('ção') else 'masculino'
                elif word.endswith(('ar', 'er', 'ir')):
                    features['type'] = 'verbo'
                    features['infinitive'] = word
                elif word.endswith(('mente',)):
                    features['type'] = 'advérbio'
                elif word.endswith(('al', 'ar', 'oso', 'osa', 'ável', 'ível')):
                    features['type'] = 'adjetivo'

                # Número
                if word.endswith('s') and not word.endswith(('as', 'es', 'is', 'os', 'us')):
                    features['number'] = 'plural'
                else:
                    features['number'] = 'singular'

                # Tempo verbal (simplificado)
                if word.endswith(('ei', 'ou', 'ava', 'ia')):
                    features['tense'] = 'pretérito'

                return features

        morphology = PortugueseMorphology(method)

        if method == 'stemming':
            processed = morphology.rslp_stemmer(text)
        else:
            processed = morphology.portuguese_lemmatizer(text)

        morphological_analysis = morphology.morphological_analysis(text)

        return {
            'morphology_processor': morphology,
            'processed_words': processed,
            'morphological_analysis': morphological_analysis
        }
```

**Processamento de Linguagem Natural:**
- Tokenização avançada considerando clíticos em português
- Normalização de texto com tratamento de caracteres especiais
- Tratamento de palavras vazias (stop words) para português
- Stemming e lematização usando RSLP para português

### 1.2 Análise Sintática e Semântica
```python
import numpy as np
from collections import defaultdict
import re
from nltk import CFG
from nltk.parse import RecursiveDescentParser

class SyntacticSemanticAnalysis:
    """
    Análise sintática e semântica para português
    """

    def __init__(self):
        self.grammar_rules = self._load_portuguese_grammar()

    def _load_portuguese_grammar(self):
        """Carregar gramática do português"""
        grammar = CFG.fromstring("""
            S -> NP VP
            NP -> Det N | Det N PP | N
            VP -> V NP | V PP | V NP PP | V
            PP -> P NP
            Det -> 'o' | 'a' | 'os' | 'as' | 'um' | 'uma' | 'uns' | 'umas'
            N -> 'cachorro' | 'gato' | 'casa' | 'jardim' | 'homem' | 'mulher'
            V -> 'corre' | 'anda' | 'come' | 'dorme' | 'pula' | 'late'
            P -> 'em' | 'no' | 'na' | 'para' | 'com' | 'sobre'
        """)

        return grammar

    def dependency_parsing_portuguese(self, sentence, dependency_model='stanford'):
        """
        Análise de dependências para português
        """
        class DependencyParser:
            def __init__(self, model='stanford'):
                self.model = model
                self.portuguese_dependencies = self._load_dependency_patterns()

            def _load_dependency_patterns(self):
                """Padrões de dependência para português"""
                patterns = {
                    'subject': [
                        ('nsubj', 'Sujeito'),
                        ('nsubjpass', 'Sujeito passivo'),
                        ('csubj', 'Sujeito clausal')
                    ],
                    'object': [
                        ('dobj', 'Objeto direto'),
                        ('iobj', 'Objeto indireto'),
                        ('pobj', 'Objeto de preposição')
                    ],
                    'modifier': [
                        ('amod', 'Modificador de adjetivo'),
                        ('advmod', 'Modificador adverbial'),
                        ('nmod', 'Modificador nominal')
                    ],
                    'clausal': [
                        ('ccomp', 'Complemento clausal'),
                        ('xcomp', 'Complemento de extensão'),
                        ('advcl', 'Cláusula adverbial')
                    ]
                }

                return patterns

            def parse_dependencies(self, sentence):
                """Analisar dependências da sentença"""
                # Tokenização
                tokens = self._tokenize_sentence(sentence)

                # POS tagging (simplificado)
                pos_tags = self._pos_tagging(tokens)

                # Análise de dependências (simplificada)
                dependencies = self._extract_dependencies(tokens, pos_tags)

                return {
                    'tokens': tokens,
                    'pos_tags': pos_tags,
                    'dependencies': dependencies,
                    'dependency_tree': self._build_dependency_tree(dependencies)
                }

            def _tokenize_sentence(self, sentence):
                """Tokenização de sentença"""
                # Remover pontuação e tokenizar
                clean_sentence = re.sub(r'[^\w\s]', '', sentence)
                tokens = clean_sentence.lower().split()

                return tokens

            def _pos_tagging(self, tokens):
                """POS tagging simplificado para português"""
                pos_dict = {
                    'o': 'DET', 'a': 'DET', 'os': 'DET', 'as': 'DET',
                    'um': 'DET', 'uma': 'DET', 'uns': 'DET', 'umas': 'DET',
                    'cachorro': 'NOUN', 'gato': 'NOUN', 'casa': 'NOUN',
                    'homem': 'NOUN', 'mulher': 'NOUN', 'jardim': 'NOUN',
                    'corre': 'VERB', 'anda': 'VERB', 'come': 'VERB',
                    'dorme': 'VERB', 'pula': 'VERB', 'late': 'VERB',
                    'em': 'ADP', 'no': 'ADP', 'na': 'ADP', 'para': 'ADP',
                    'com': 'ADP', 'sobre': 'ADP', 'muito': 'ADV',
                    'rapidamente': 'ADV', 'grande': 'ADJ', 'pequeno': 'ADJ'
                }

                pos_tags = []
                for token in tokens:
                    pos = pos_dict.get(token, 'UNK')
                    pos_tags.append((token, pos))

                return pos_tags

            def _extract_dependencies(self, tokens, pos_tags):
                """Extrair relações de dependência"""
                dependencies = []

                # Regras simplificadas de dependência
                for i, (token, pos) in enumerate(pos_tags):
                    if pos == 'VERB':
                        # Procurar sujeito (antes do verbo)
                        for j in range(i):
                            if pos_tags[j][1] in ['NOUN', 'PRON']:
                                dependencies.append({
                                    'head': token,
                                    'dependent': pos_tags[j][0],
                                    'relation': 'nsubj',
                                    'head_index': i,
                                    'dep_index': j
                                })
                                break

                        # Procurar objeto (depois do verbo)
                        for j in range(i + 1, len(pos_tags)):
                            if pos_tags[j][1] in ['NOUN', 'PRON']:
                                dependencies.append({
                                    'head': token,
                                    'dependent': pos_tags[j][0],
                                    'relation': 'dobj',
                                    'head_index': i,
                                    'dep_index': j
                                })
                                break

                    elif pos == 'ADP':
                        # Preposição seguida de substantivo
                        if i + 1 < len(pos_tags) and pos_tags[i + 1][1] in ['NOUN', 'PRON']:
                            dependencies.append({
                                'head': pos_tags[i + 1][0],
                                'dependent': token,
                                'relation': 'case',
                                'head_index': i + 1,
                                'dep_index': i
                            })

                return dependencies

            def _build_dependency_tree(self, dependencies):
                """Construir árvore de dependências"""
                tree = defaultdict(list)

                for dep in dependencies:
                    tree[dep['head']].append({
                        'dependent': dep['dependent'],
                        'relation': dep['relation']
                    })

                return dict(tree)

            def extract_semantic_roles(self, dependencies):
                """Extrair papéis semânticos"""
                semantic_roles = {
                    'agent': None,
                    'patient': None,
                    'instrument': None,
                    'location': None,
                    'time': None
                }

                for dep in dependencies:
                    if dep['relation'] == 'nsubj':
                        semantic_roles['agent'] = dep['dependent']
                    elif dep['relation'] == 'dobj':
                        semantic_roles['patient'] = dep['dependent']
                    elif dep['relation'] == 'nmod' and 'local' in dep.get('dependent', ''):
                        semantic_roles['location'] = dep['dependent']

                return semantic_roles

        dep_parser = DependencyParser(dependency_model)
        parse_result = dep_parser.parse_dependencies(sentence)
        semantic_roles = dep_parser.extract_semantic_roles(parse_result['dependencies'])

        return {
            'dependency_parser': dep_parser,
            'parse_result': parse_result,
            'semantic_roles': semantic_roles
        }

    def constituent_parsing_portuguese(self, sentence):
        """
        Análise de constituintes para português
        """
        class ConstituentParser:
            def __init__(self):
                self.grammar = self._create_portuguese_grammar()

            def _create_portuguese_grammar(self):
                """Criar gramática de constituintes para português"""
                grammar = CFG.fromstring("""
                    S -> NP VP
                    NP -> Det N | Det N PP | Det Adj N | N
                    VP -> V | V NP | V PP | V NP PP | V Adj
                    PP -> P NP
                    Det -> 'o' | 'a' | 'os' | 'as' | 'um' | 'uma' | 'uns' | 'umas' | 'este' | 'esta' | 'esse' | 'essa'
                    N -> 'cachorro' | 'gato' | 'casa' | 'homem' | 'mulher' | 'livro' | 'mesa' | 'carro'
                    V -> 'corre' | 'anda' | 'come' | 'dorme' | 'pula' | 'late' | 'estuda' | 'trabalha'
                    Adj -> 'grande' | 'pequeno' | 'bonito' | 'feio' | 'velho' | 'novo' | 'inteligente'
                    P -> 'em' | 'no' | 'na' | 'para' | 'com' | 'sobre' | 'de' | 'do' | 'da'
                    Adv -> 'muito' | 'pouco' | 'rapidamente' | 'devagar' | 'bem' | 'mal'
                """)

                return grammar

            def parse_constituents(self, sentence):
                """Analisar constituintes da sentença"""
                # Tokenizar
                tokens = self._tokenize_sentence(sentence)

                # Parser recursivo
                parser = RecursiveDescentParser(self.grammar)

                try:
                    parses = list(parser.parse(tokens))
                    if parses:
                        parse_tree = parses[0]  # Pegar primeira análise
                        return {
                            'parse_success': True,
                            'parse_tree': self._tree_to_dict(parse_tree),
                            'n_parses': len(parses),
                            'constituents': self._extract_constituents(parse_tree)
                        }
                    else:
                        return {
                            'parse_success': False,
                            'error': 'No valid parse found'
                        }
                except Exception as e:
                    return {
                        'parse_success': False,
                        'error': str(e)
                    }

            def _tokenize_sentence(self, sentence):
                """Tokenização simples"""
                # Remover pontuação e tokenizar
                clean = re.sub(r'[^\w\s]', '', sentence)
                tokens = clean.lower().split()

                return tokens

            def _tree_to_dict(self, tree):
                """Converter árvore NLTK para dicionário"""
                if isinstance(tree, str):
                    return tree

                result = {
                    'label': tree.label(),
                    'children': [self._tree_to_dict(child) for child in tree]
                }

                return result

            def _extract_constituents(self, tree):
                """Extrair constituintes da árvore"""
                constituents = []

                def traverse_tree(node):
                    if isinstance(node, dict):
                        constituents.append({
                            'label': node['label'],
                            'span': self._get_tree_span(node)
                        })

                        for child in node['children']:
                            traverse_tree(child)
                    else:
                        constituents.append({
                            'label': 'word',
                            'word': node,
                            'span': (0, 1)  # Placeholder
                        })

                traverse_tree(tree)
                return constituents

            def _get_tree_span(self, tree_dict):
                """Obter extensão da árvore"""
                # Placeholder - implementação simplificada
                return (0, len(tree_dict.get('children', [])))

            def grammar_induction(self, sentences):
                """Indução de gramática a partir de dados"""
                # Contar padrões de bigramas
                bigram_counts = defaultdict(int)

                for sentence in sentences:
                    tokens = self._tokenize_sentence(sentence)
                    for i in range(len(tokens) - 1):
                        bigram = (tokens[i], tokens[i + 1])
                        bigram_counts[bigram] += 1

                # Induzir regras baseadas em frequência
                induced_rules = []

                for (word1, word2), count in bigram_counts.items():
                    if count > 2:  # Threshold
                        # Criar regra NP -> word1 word2 se ambos substantivos
                        if self._is_noun(word1) and self._is_noun(word2):
                            induced_rules.append(f"NP -> '{word1}' '{word2}'")

                return induced_rules

            def _is_noun(self, word):
                """Verificar se palavra é substantivo (simplificado)"""
                nouns = ['cachorro', 'gato', 'casa', 'homem', 'mulher', 'livro', 'mesa', 'carro']
                return word in nouns

        const_parser = ConstituentParser()
        parse_result = const_parser.parse_constituents(sentence)

        return {
            'constituent_parser': const_parser,
            'parse_result': parse_result
        }

    def semantic_role_labeling_portuguese(self, sentence):
        """
        Rotulagem de papéis semânticos para português
        """
        class SemanticRoleLabeler:
            def __init__(self):
                self.frame_elements = {
                    'agent': 'Entidade que realiza a ação',
                    'patient': 'Entidade que sofre a ação',
                    'theme': 'Entidade que muda de estado ou localização',
                    'instrument': 'Objeto usado para realizar a ação',
                    'location': 'Local onde a ação ocorre',
                    'time': 'Tempo quando a ação ocorre',
                    'cause': 'Causa da ação',
                    'purpose': 'Propósito da ação'
                }

            def label_semantic_roles(self, sentence, parse_tree=None):
                """Rotular papéis semânticos"""
                # Tokenização e POS tagging
                tokens = self._tokenize_and_tag(sentence)

                # Identificar predicado principal
                predicate = self._identify_predicate(tokens)

                if not predicate:
                    return {'error': 'No predicate found'}

                # Extrair argumentos
                arguments = self._extract_arguments(tokens, predicate)

                # Atribuir papéis semânticos
                semantic_roles = self._assign_semantic_roles(predicate, arguments)

                return {
                    'predicate': predicate,
                    'arguments': arguments,
                    'semantic_roles': semantic_roles,
                    'frame_elements': self.frame_elements
                }

            def _tokenize_and_tag(self, sentence):
                """Tokenizar e fazer POS tagging"""
                tokens = re.findall(r'\b\w+\b', sentence.lower())

                # POS tagging simplificado
                pos_tags = []
                for token in tokens:
                    if token in ['o', 'a', 'os', 'as', 'um', 'uma']:
                        pos = 'DET'
                    elif token in ['corre', 'anda', 'come', 'dorme', 'pula', 'late', 'estuda']:
                        pos = 'VERB'
                    elif token in ['cachorro', 'gato', 'casa', 'homem', 'mulher', 'livro']:
                        pos = 'NOUN'
                    else:
                        pos = 'UNK'

                    pos_tags.append((token, pos))

                return pos_tags

            def _identify_predicate(self, tokens):
                """Identificar predicado principal"""
                for token, pos in tokens:
                    if pos == 'VERB':
                        return {'word': token, 'position': tokens.index((token, pos))}

                return None

            def _extract_arguments(self, tokens, predicate):
                """Extrair argumentos do predicado"""
                arguments = []

                pred_pos = predicate['position']

                # Argumentos antes do verbo (sujeito)
                for i in range(pred_pos):
                    if tokens[i][1] in ['NOUN', 'PRON']:
                        arguments.append({
                            'word': tokens[i][0],
                            'position': i,
                            'relation': 'before_predicate'
                        })

                # Argumentos depois do verbo (objeto)
                for i in range(pred_pos + 1, len(tokens)):
                    if tokens[i][1] in ['NOUN', 'PRON']:
                        arguments.append({
                            'word': tokens[i][0],
                            'position': i,
                            'relation': 'after_predicate'
                        })

                return arguments

            def _assign_semantic_roles(self, predicate, arguments):
                """Atribuir papéis semânticos"""
                semantic_roles = {}

                # Papéis baseados na posição e tipo de verbo
                verb_type = self._classify_verb(predicate['word'])

                for arg in arguments:
                    if arg['relation'] == 'before_predicate':
                        if verb_type == 'action':
                            semantic_roles['agent'] = arg['word']
                        else:
                            semantic_roles['theme'] = arg['word']
                    elif arg['relation'] == 'after_predicate':
                        if verb_type == 'action':
                            semantic_roles['patient'] = arg['word']
                        else:
                            semantic_roles['location'] = arg['word']

                return semantic_roles

            def _classify_verb(self, verb):
                """Classificar tipo do verbo"""
                action_verbs = ['corre', 'anda', 'come', 'pula', 'late', 'estuda']
                state_verbs = ['dorme', 'existe', 'parece']

                if verb in action_verbs:
                    return 'action'
                elif verb in state_verbs:
                    return 'state'
                else:
                    return 'unknown'

            def semantic_frame_matching(self, sentence, frame_library):
                """Corresponder com frames semânticos"""
                # Biblioteca de frames (simplificada)
                frames = {
                    'motion': {
                        'predicate': 'corre',
                        'roles': ['agent', 'location'],
                        'example': 'O cachorro corre no parque'
                    },
                    'consumption': {
                        'predicate': 'come',
                        'roles': ['agent', 'patient'],
                        'example': 'O gato come o peixe'
                    }
                }

                tokens = self._tokenize_and_tag(sentence)
                predicate = self._identify_predicate(tokens)

                if not predicate:
                    return None

                # Encontrar frame correspondente
                for frame_name, frame_info in frames.items():
                    if predicate['word'] == frame_info['predicate']:
                        return {
                            'matched_frame': frame_name,
                            'frame_info': frame_info,
                            'semantic_roles': self.label_semantic_roles(sentence)['semantic_roles']
                        }

                return None

        srl = SemanticRoleLabeler()
        semantic_analysis = srl.label_semantic_roles(sentence)

        return {
            'semantic_role_labeler': srl,
            'semantic_analysis': semantic_analysis
        }

    def discourse_analysis_portuguese(self, text):
        """
        Análise de discurso para português
        """
        class DiscourseAnalyzer:
            def __init__(self):
                self.discourse_markers = {
                    'portuguese': {
                        'contrast': ['mas', 'porém', 'contudo', 'entretanto', 'todavia', 'no entanto'],
                        'cause': ['porque', 'pois', 'por isso', 'portanto', 'consequentemente'],
                        'addition': ['e', 'também', 'além disso', 'ademais', 'mais ainda'],
                        'temporal': ['depois', 'então', 'em seguida', 'posteriormente', 'anteriormente'],
                        'condition': ['se', 'caso', 'desde que', 'a menos que', 'contanto que']
                    }
                }

            def analyze_discourse_structure(self, text):
                """Analisar estrutura discursiva"""
                sentences = self._split_sentences(text)

                discourse_relations = []

                for i, sentence in enumerate(sentences):
                    markers = self._find_discourse_markers(sentence)

                    if markers:
                        for marker in markers:
                            relation = {
                                'marker': marker['word'],
                                'type': marker['type'],
                                'sentence_index': i,
                                'position': marker['position']
                            }

                            # Identificar sentença relacionada
                            if i > 0:
                                relation['related_sentence'] = i - 1

                            discourse_relations.append(relation)

                return {
                    'sentences': sentences,
                    'discourse_relations': discourse_relations,
                    'discourse_structure': self._build_discourse_tree(sentences, discourse_relations)
                }

            def _split_sentences(self, text):
                """Dividir texto em sentenças"""
                # Usar NLTK ou regex simples
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]

                return sentences

            def _find_discourse_markers(self, sentence):
                """Encontrar marcadores discursivos"""
                words = re.findall(r'\b\w+\b', sentence.lower())
                markers = []

                for i, word in enumerate(words):
                    for relation_type, marker_list in self.discourse_markers['portuguese'].items():
                        if word in marker_list:
                            markers.append({
                                'word': word,
                                'type': relation_type,
                                'position': i
                            })

                return markers

            def _build_discourse_tree(self, sentences, relations):
                """Construir árvore discursiva"""
                tree = {'root': 'main_discourse'}

                for i, sentence in enumerate(sentences):
                    tree[f'sentence_{i}'] = {
                        'text': sentence,
                        'relations': [r for r in relations if r['sentence_index'] == i]
                    }

                return tree

            def coherence_analysis(self, text):
                """Análise de coerência textual"""
                sentences = self._split_sentences(text)

                coherence_measures = {
                    'lexical_cohesion': self._measure_lexical_cohesion(sentences),
                    'semantic_coherence': self._measure_semantic_coherence(sentences),
                    'temporal_coherence': self._measure_temporal_coherence(sentences)
                }

                return coherence_measures

            def _measure_lexical_cohesion(self, sentences):
                """Medir coesão lexical"""
                # Contar palavras repetidas entre sentenças
                cohesion_score = 0
                total_comparisons = 0

                for i in range(len(sentences) - 1):
                    words1 = set(re.findall(r'\b\w+\b', sentences[i].lower()))
                    words2 = set(re.findall(r'\b\w+\b', sentences[i + 1].lower()))

                    overlap = len(words1 & words2)
                    union = len(words1 | words2)

                    if union > 0:
                        cohesion_score += overlap / union
                    total_comparisons += 1

                return cohesion_score / total_comparisons if total_comparisons > 0 else 0

            def _measure_semantic_coherence(self, sentences):
                """Medir coerência semântica"""
                # Placeholder - usaria embeddings semânticos
                return 0.7  # Valor simulado

            def _measure_temporal_coherence(self, sentences):
                """Medir coerência temporal"""
                temporal_markers = ['depois', 'antes', 'durante', 'então', 'agora']

                temporal_score = 0

                for sentence in sentences:
                    words = re.findall(r'\b\w+\b', sentence.lower())
                    temporal_words = [w for w in words if w in temporal_markers]

                    temporal_score += len(temporal_words)

                return temporal_score / len(sentences)

        discourse_analyzer = DiscourseAnalyzer()
        discourse_structure = discourse_analyzer.analyze_discourse_structure(text)
        coherence = discourse_analyzer.coherence_analysis(text)

        return {
            'discourse_analyzer': discourse_analyzer,
            'discourse_structure': discourse_structure,
            'coherence_measures': coherence
        }
```

**Análise Sintática e Semântica:**
- Análise de dependências com padrões específicos do português
- Parsing de constituintes usando gramática CFG
- Rotulagem de papéis semânticos (Semantic Role Labeling)
- Análise de discurso com marcadores específicos do português

### 1.3 Tradução Automática e Avaliação
```python
import numpy as np
from collections import Counter, defaultdict
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

class MachineTranslationEvaluation:
    """
    Tradução automática e avaliação para português
    """

    def __init__(self):
        self.translation_models = {}

    def statistical_machine_translation(self, source_text, target_language='english'):
        """
        Tradução automática estatística
        """
        class StatisticalTranslator:
            def __init__(self, target_lang='english'):
                self.target_language = target_lang
                self.translation_table = self._load_translation_probabilities()

            def _load_translation_probabilities(self):
                """Carregar tabela de probabilidades de tradução"""
                # Tabela simplificada português-inglês
                translation_probs = {
                    'casa': {'house': 0.8, 'home': 0.2},
                    'cachorro': {'dog': 0.9, 'hound': 0.1},
                    'corre': {'runs': 0.6, 'run': 0.4},
                    'no': {'in': 0.7, 'on': 0.3},
                    'parque': {'park': 0.9, 'garden': 0.1},
                    'o': {'the': 1.0},
                    'gato': {'cat': 0.8, 'feline': 0.2},
                    'come': {'eats': 0.7, 'eat': 0.3},
                    'peixe': {'fish': 0.9, 'fishing': 0.1}
                }

                return translation_probs

            def translate_phrase(self, source_phrase):
                """Traduzir frase usando modelo estatístico"""
                source_words = source_phrase.lower().split()
                translation_options = []

                # Gerar traduções para cada palavra
                for word in source_words:
                    if word in self.translation_table:
                        translations = self.translation_table[word]
                        translation_options.append(list(translations.keys()))
                    else:
                        # Palavra desconhecida - copiar como está
                        translation_options.append([word])

                # Gerar todas as combinações possíveis (simplificado)
                # Para demonstração, usar apenas a tradução mais provável
                best_translation = []

                for options in translation_options:
                    # Escolher tradução mais provável
                    if options and isinstance(options[0], str):
                        best_translation.append(options[0])
                    else:
                        best_translation.append(options[0] if options else 'UNK')

                return ' '.join(best_translation)

            def phrase_based_translation(self, source_phrase):
                """Tradução baseada em frases"""
                # Dicionário de frases
                phrase_translations = {
                    'o cachorro corre': 'the dog runs',
                    'no parque': 'in the park',
                    'o gato come': 'the cat eats',
                    'o peixe': 'the fish'
                }

                # Procurar frases correspondentes
                for pt_phrase, en_phrase in phrase_translations.items():
                    if pt_phrase in source_phrase:
                        translated = source_phrase.replace(pt_phrase, en_phrase)
                        return translated

                # Fallback para tradução palavra por palavra
                return self.translate_phrase(source_phrase)

            def alignment_model(self, source_sentence, target_sentence):
                """Modelo de alinhamento entre línguas"""
                source_words = source_sentence.split()
                target_words = target_sentence.split()

                # Matriz de alinhamento (simplificada)
                alignment_matrix = np.zeros((len(source_words), len(target_words)))

                # Alinhamentos baseados em posição
                for i, src_word in enumerate(source_words):
                    for j, tgt_word in enumerate(target_words):
                        # Similaridade simples baseada em tradução
                        if src_word in self.translation_table:
                            translations = self.translation_table[src_word]
                            if tgt_word in translations:
                                alignment_matrix[i, j] = translations[tgt_word]

                return {
                    'source_words': source_words,
                    'target_words': target_words,
                    'alignment_matrix': alignment_matrix,
                    'best_alignments': self._extract_best_alignments(alignment_matrix)
                }

            def _extract_best_alignments(self, alignment_matrix):
                """Extrair melhores alinhamentos"""
                alignments = []

                for i in range(alignment_matrix.shape[0]):
                    best_j = np.argmax(alignment_matrix[i])
                    if alignment_matrix[i, best_j] > 0:
                        alignments.append({
                            'source_index': i,
                            'target_index': best_j,
                            'probability': alignment_matrix[i, best_j]
                        })

                return alignments

        stat_translator = StatisticalTranslator(target_language)
        translation = stat_translator.translate_phrase(source_text)
        alignment = stat_translator.alignment_model(source_text, translation)

        return {
            'statistical_translator': stat_translator,
            'translation': translation,
            'word_alignment': alignment
        }

    def neural_machine_translation(self, source_text, model_type='transformer'):
        """
        Tradução automática neural
        """
        class NeuralTranslator:
            def __init__(self, model_type='transformer'):
                self.model_type = model_type
                self.vocab_size = 1000
                self.embedding_dim = 256
                self.num_heads = 8
                self.num_layers = 6

            def transformer_translation(self, source_text):
                """Tradução usando arquitetura Transformer"""
                # Simulação simplificada do processo de tradução

                # 1. Codificação da sentença fonte
                source_tokens = self._tokenize_text(source_text)
                source_embeddings = self._positional_encoding(source_tokens)

                # 2. Codificação através de múltiplas camadas
                encoder_output = self._transformer_encoder(source_embeddings)

                # 3. Decodificação autoregressiva
                target_tokens = self._greedy_decode(encoder_output)

                # 4. Converter tokens para texto
                translation = self._tokens_to_text(target_tokens)

                return {
                    'source_tokens': source_tokens,
                    'target_tokens': target_tokens,
                    'translation': translation,
                    'attention_weights': self._compute_attention_weights(encoder_output)
                }

            def _tokenize_text(self, text):
                """Tokenização simplificada"""
                words = text.lower().split()
                # Converter para índices (simulado)
                tokens = [hash(word) % self.vocab_size for word in words]

                return tokens

            def _positional_encoding(self, tokens):
                """Codificação posicional"""
                positions = np.arange(len(tokens))
                pos_encoding = np.zeros((len(tokens), self.embedding_dim))

                for pos in range(len(tokens)):
                    for i in range(0, self.embedding_dim, 2):
                        pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.embedding_dim)))
                        if i + 1 < self.embedding_dim:
                            pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.embedding_dim)))

                # Embeddings de palavra (simulados)
                word_embeddings = np.random.randn(len(tokens), self.embedding_dim)

                return word_embeddings + pos_encoding

            def _transformer_encoder(self, embeddings):
                """Codificador Transformer simplificado"""
                # Múltiplas camadas de atenção e feed-forward
                output = embeddings

                for layer in range(self.num_layers):
                    # Auto-atenção
                    attention_output = self._multihead_attention(output, output, output)

                    # Feed-forward
                    ff_output = self._feed_forward(attention_output)

                    # Residual connection e layer norm (simplificado)
                    output = output + ff_output

                return output

            def _multihead_attention(self, query, key, value):
                """Atenção multi-cabeça"""
                # Simplificação: apenas uma cabeça
                d_k = self.embedding_dim // self.num_heads

                # Pesos de atenção (simulados)
                scores = np.random.randn(query.shape[0], key.shape[0])
                attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

                # Output da atenção
                attention_output = np.dot(attention_weights, value)

                return attention_output

            def _feed_forward(self, x):
                """Rede feed-forward"""
                # Duas camadas lineares com ReLU
                hidden = np.maximum(0, np.dot(x, np.random.randn(self.embedding_dim, 4 * self.embedding_dim)))
                output = np.dot(hidden, np.random.randn(4 * self.embedding_dim, self.embedding_dim))

                return output

            def _greedy_decode(self, encoder_output):
                """Decodificação greedy"""
                # Começar com token de início
                target_tokens = [0]  # Token <SOS>

                max_length = 20

                for _ in range(max_length):
                    # Codificação posicional para tokens alvo
                    target_embeddings = self._positional_encoding(target_tokens)

                    # Atenção cruzada (encoder-decoder)
                    decoder_output = self._multihead_attention(target_embeddings, encoder_output, encoder_output)

                    # Previsão do próximo token (simulado)
                    next_token_logits = np.random.randn(self.vocab_size)
                    next_token = np.argmax(next_token_logits)

                    target_tokens.append(next_token)

                    # Parar se token de fim
                    if next_token == 1:  # Token <EOS>
                        break

                return target_tokens

            def _tokens_to_text(self, tokens):
                """Converter tokens para texto"""
                # Dicionário reverso simplificado
                id_to_word = {
                    0: '<SOS>',
                    1: '<EOS>',
                    10: 'the', 11: 'dog', 12: 'runs', 13: 'in', 14: 'park',
                    15: 'cat', 16: 'eats', 17: 'fish', 18: 'house', 19: 'beautiful'
                }

                words = []
                for token in tokens:
                    if token in id_to_word and id_to_word[token] not in ['<SOS>', '<EOS>']:
                        words.append(id_to_word[token])

                return ' '.join(words)

            def _compute_attention_weights(self, encoder_output):
                """Computar pesos de atenção"""
                # Simplificado - retornar matriz aleatória
                n_tokens = encoder_output.shape[0]
                attention_weights = np.random.rand(n_tokens, n_tokens)
                attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

                return attention_weights

        neural_translator = NeuralTranslator(model_type)
        translation_result = neural_translator.transformer_translation(source_text)

        return {
            'neural_translator': neural_translator,
            'translation_result': translation_result
        }

    def translation_quality_evaluation(self, reference_translations, candidate_translations):
        """
        Avaliação da qualidade da tradução
        """
        class TranslationEvaluator:
            def __init__(self):
                self.scorers = {
                    'bleu': self._compute_bleu,
                    'meteor': self._compute_meteor,
                    'rouge': self._compute_rouge,
                    'ter': self._compute_ter
                }

            def evaluate_translations(self, references, candidates):
                """Avaliar conjunto de traduções"""
                evaluation_results = defaultdict(list)

                for ref, cand in zip(references, candidates):
                    for metric_name, scorer in self.scorers.items():
                        score = scorer(ref, cand)
                        evaluation_results[metric_name].append(score)

                # Estatísticas agregadas
                summary_stats = {}

                for metric_name, scores in evaluation_results.items():
                    summary_stats[metric_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'median': np.median(scores)
                    }

                return {
                    'individual_scores': dict(evaluation_results),
                    'summary_statistics': summary_stats,
                    'n_translations': len(references)
                }

            def _compute_bleu(self, reference, candidate):
                """Computar BLEU score"""
                reference_tokens = [reference.split()]
                candidate_tokens = candidate.split()

                # Suavização para evitar problemas com n-grams curtos
                smoothing = SmoothingFunction().method1

                try:
                    bleu_score = sentence_bleu(reference_tokens, candidate_tokens,
                                             smoothing_function=smoothing)
                    return bleu_score
                except:
                    return 0.0

            def _compute_meteor(self, reference, candidate):
                """Computar METEOR score"""
                try:
                    # METEOR requer tokenização específica
                    meteor = meteor_score([reference.split()], candidate.split())
                    return meteor
                except:
                    return 0.0

            def _compute_rouge(self, reference, candidate):
                """Computar ROUGE scores"""
                try:
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
                    scores = scorer.score(reference, candidate)

                    # Retornar ROUGE-L F1
                    return scores['rougeL'].fmeasure
                except:
                    return 0.0

            def _compute_ter(self, reference, candidate):
                """Computar Translation Error Rate (TER)"""
                # Implementação simplificada
                ref_words = reference.split()
                cand_words = candidate.split()

                # Contar edições necessárias (simplificado)
                edits = abs(len(ref_words) - len(cand_words))

                # Palavras diferentes
                for i in range(min(len(ref_words), len(cand_words))):
                    if ref_words[i] != cand_words[i]:
                        edits += 1

                ter = edits / len(ref_words) if ref_words else 1.0

                return 1.0 - ter  # Converter para acurácia

            def human_evaluation_metrics(self):
                """Métricas de avaliação humana"""
                # Métricas baseadas em juízes humanos
                metrics = {
                    'adequacy': 'Grau em que a tradução transmite o significado correto',
                    'fluency': 'Naturalidade e fluência da tradução',
                    'comprehension': 'Facilidade de compreensão',
                    'cultural_adaptation': 'Adequação cultural da tradução'
                }

                return metrics

            def error_analysis(self, references, candidates):
                """Análise de erros de tradução"""
                error_categories = {
                    'omission': 0,      # Palavras omitidas
                    'addition': 0,      # Palavras adicionadas
                    'substitution': 0,  # Substituições incorretas
                    'reordering': 0     # Reordenação incorreta
                }

                for ref, cand in zip(references, candidates):
                    ref_words = ref.split()
                    cand_words = cand.split()

                    # Detectar omissões e adições
                    ref_set = set(ref_words)
                    cand_set = set(cand_words)

                    omissions = len(ref_set - cand_set)
                    additions = len(cand_set - ref_set)

                    # Substituições (palavras em comum mas em posições diferentes)
                    common_words = ref_set & cand_set
                    substitutions = len(common_words) - self._count_aligned_words(ref_words, cand_words)

                    error_categories['omission'] += omissions
                    error_categories['addition'] += additions
                    error_categories['substitution'] += substitutions

                return error_categories

            def _count_aligned_words(self, ref_words, cand_words):
                """Contar palavras alinhadas corretamente"""
                aligned = 0

                for i, ref_word in enumerate(ref_words):
                    if i < len(cand_words) and ref_word == cand_words[i]:
                        aligned += 1

                return aligned

        evaluator = TranslationEvaluator()
        evaluation_results = evaluator.evaluate_translations(reference_translations, candidate_translations)
        error_analysis = evaluator.error_analysis(reference_translations, candidate_translations)

        return {
            'translation_evaluator': evaluator,
            'evaluation_results': evaluation_results,
            'error_analysis': error_analysis
        }

    def multilingual_corpus_analysis(self, corpus, languages=['portuguese', 'english', 'spanish']):
        """
        Análise de corpus multilíngue
        """
        class MultilingualAnalyzer:
            def __init__(self, languages):
                self.languages = languages
                self.language_detectors = self._load_language_detectors()

            def _load_language_detectors(self):
                """Carregar detectores de idioma"""
                detectors = {}

                for lang in self.languages:
                    if lang == 'portuguese':
                        detectors[lang] = {
                            'keywords': ['que', 'não', 'como', 'mais', 'mas', 'seu', 'sua'],
                            'suffixes': ['ção', 'mente', 'inho', 'inha']
                        }
                    elif lang == 'english':
                        detectors[lang] = {
                            'keywords': ['the', 'and', 'or', 'but', 'in', 'on', 'at'],
                            'suffixes': ['tion', 'ment', 'ness', 'ship']
                        }
                    elif lang == 'spanish':
                        detectors[lang] = {
                            'keywords': ['que', 'no', 'como', 'pero', 'su', 'con'],
                            'suffixes': ['ción', 'mente', 'ito', 'ita']
                        }

                return detectors

            def analyze_multilingual_corpus(self, corpus):
                """Analisar corpus multilíngue"""
                language_distribution = Counter()
                code_switching = []
                lexical_borrowing = []

                for text in corpus:
                    # Detectar idioma
                    detected_lang = self._detect_language(text)
                    language_distribution[detected_lang] += 1

                    # Detectar alternância de código
                    switches = self._detect_code_switching(text)
                    code_switching.extend(switches)

                    # Detectar empréstimos lexicais
                    borrowings = self._detect_lexical_borrowing(text)
                    lexical_borrowing.extend(borrowings)

                return {
                    'language_distribution': dict(language_distribution),
                    'code_switching_instances': code_switching,
                    'lexical_borrowing': lexical_borrowing,
                    'multilingual_statistics': self._compute_multilingual_stats(corpus)
                }

            def _detect_language(self, text):
                """Detectar idioma do texto"""
                scores = {}

                words = re.findall(r'\b\w+\b', text.lower())

                for lang, detector in self.language_detectors.items():
                    score = 0

                    # Pontuar palavras-chave
                    for word in words:
                        if word in detector['keywords']:
                            score += 2

                        # Pontuar sufixos
                        for suffix in detector['suffixes']:
                            if word.endswith(suffix):
                                score += 1

                    scores[lang] = score

                return max(scores, key=scores.get) if scores else 'unknown'

            def _detect_code_switching(self, text):
                """Detectar alternância de código"""
                words = re.findall(r'\b\w+\b', text)
                switches = []

                for i in range(1, len(words)):
                    lang1 = self._classify_word_language(words[i-1])
                    lang2 = self._classify_word_language(words[i])

                    if lang1 != lang2 and lang1 != 'unknown' and lang2 != 'unknown':
                        switches.append({
                            'word1': words[i-1],
                            'word2': words[i],
                            'lang1': lang1,
                            'lang2': lang2,
                            'position': i
                        })

                return switches

            def _classify_word_language(self, word):
                """Classificar idioma de uma palavra"""
                for lang, detector in self.language_detectors.items():
                    if word in detector['keywords']:
                        return lang

                    for suffix in detector['suffixes']:
                        if word.endswith(suffix):
                            return lang

                return 'unknown'

            def _detect_lexical_borrowing(self, text):
                """Detectar empréstimos lexicais"""
                # Palavras emprestadas comuns português-inglês
                borrowed_words = ['computer', 'internet', 'email', 'software', 'hardware']

                borrowings = []

                words = re.findall(r'\b\w+\b', text.lower())

                for word in words:
                    if word in borrowed_words:
                        borrowings.append({
                            'word': word,
                            'source_language': 'english',
                            'target_language': 'portuguese'
                        })

                return borrowings

            def _compute_multilingual_stats(self, corpus):
                """Computar estatísticas multilíngues"""
                total_texts = len(corpus)
                avg_text_length = np.mean([len(text.split()) for text in corpus])

                # Diversidade lexical por idioma
                lexical_diversity = {}

                for lang in self.languages:
                    lang_texts = [text for text in corpus if self._detect_language(text) == lang]
                    if lang_texts:
                        all_words = []
                        for text in lang_texts:
                            all_words.extend(re.findall(r'\b\w+\b', text.lower()))

                        unique_words = len(set(all_words))
                        total_words = len(all_words)
                        lexical_diversity[lang] = unique_words / total_words if total_words > 0 else 0

                return {
                    'total_texts': total_texts,
                    'avg_text_length': avg_text_length,
                    'lexical_diversity': lexical_diversity
                }

        multilingual_analyzer = MultilingualAnalyzer(languages)
        analysis_results = multilingual_analyzer.analyze_multilingual_corpus(corpus)

        return {
            'multilingual_analyzer': multilingual_analyzer,
            'analysis_results': analysis_results
        }
```

**Tradução Automática e Avaliação:**
- Tradução automática estatística com alinhamento de palavras
- Tradução neural usando arquitetura Transformer
- Avaliação de qualidade com métricas BLEU, METEOR, ROUGE
- Análise de corpus multilíngue com detecção de idiomas

---

## 4. CONSIDERAÇÕES FINAIS

A linguística computacional oferece ferramentas poderosas para processar, analisar e gerar linguagem natural, com aplicações que vão desde a tradução automática até a compreensão de texto. Os modelos apresentados fornecem uma base sólida para:

1. **Pré-processamento**: Tokenização avançada, normalização e tratamento de palavras vazias
2. **Análise Sintática**: Parsing de constituintes e dependências para português
3. **Semântica**: Rotulagem de papéis semânticos e análise de discurso
4. **Tradução**: Modelos estatísticos e neurais para tradução automática
5. **Avaliação**: Métricas abrangentes para avaliar qualidade de tradução

**Próximos Passos Recomendados**:
1. Dominar técnicas de pré-processamento específicas do português
2. Explorar modelos de análise sintática e semântica
3. Implementar sistemas de tradução neural para pares de idiomas
4. Desenvolver aplicações práticas em PLN para português
5. Integrar múltiplas técnicas em sistemas de processamento de linguagem

---

*Documento preparado para fine-tuning de IA em Linguística Computacional*
*Versão 1.0 - Preparado para implementação prática*
