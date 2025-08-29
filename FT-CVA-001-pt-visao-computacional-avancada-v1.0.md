# FT-CVA-001: Fine-Tuning para IA em Visão Computacional Avançada

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados em visão computacional avançada, abrangendo desde processamento de imagens fundamentais até técnicas de deep learning para reconhecimento, segmentação e compreensão visual.

### Contexto Filosófico
A visão computacional representa a ponte entre a percepção visual humana e o processamento computacional, buscando não apenas replicar, mas aprimorar nossas capacidades de interpretação visual através de algoritmos matemáticos e arquiteturas neurais profundas.

### Metodologia de Aprendizado Recomendada
1. **Fundamentos Visuais**: Compreensão de processamento de imagens e visão biológica
2. **Abordagem Arquitetural**: Design de redes neurais para tarefas visuais
3. **Otimização de Performance**: Técnicas de eficiência computacional
4. **Integração Multimodal**: Combinação de visão com outras modalidades
5. **Aplicações Práticas**: Implementação em problemas do mundo real

---

## 1. PROCESSAMENTO DE IMAGENS AVANÇADO

### 1.1 Transformadas e Representações
```python
import numpy as np
import cv2
from scipy.fft import fft2, ifft2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

class AdvancedImageProcessing:
    """
    Técnicas avançadas de processamento de imagens
    """

    def __init__(self, image):
        self.image = np.array(image, dtype=np.float32)
        if len(self.image.shape) == 2:
            self.is_grayscale = True
        else:
            self.is_grayscale = False

    def wavelet_transform(self, wavelet_type='haar', levels=4):
        """
        Transformada wavelet para análise multiresolução
        """
        class WaveletTransform:
            def __init__(self, image, wavelet, levels):
                self.image = image
                self.wavelet = wavelet
                self.levels = levels

            def decompose(self):
                """Decomposição wavelet"""
                current_image = self.image.copy()
                coefficients = []

                for level in range(self.levels):
                    # Análise horizontal e vertical
                    h_coeff, v_coeff, d_coeff = self._wavelet_decomposition(current_image)

                    # Aproximação (baixo-baixo)
                    approx = self._wavelet_reconstruction(h_coeff, v_coeff, d_coeff, mode='approx')

                    coefficients.append({
                        'horizontal': h_coeff,
                        'vertical': v_coeff,
                        'diagonal': d_coeff,
                        'approximation': approx
                    })

                    current_image = approx

                return coefficients

            def _wavelet_decomposition(self, image):
                """Decomposição em uma nível"""
                # Filtros Haar simplificados
                if self.wavelet == 'haar':
                    # Filtros de decomposição
                    h = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Low-pass
                    g = np.array([-1/np.sqrt(2), 1/np.sqrt(2)])  # High-pass

                    # Convolução horizontal
                    temp = convolve(image, h.reshape(1, -1))
                    h_out = convolve(temp, h.reshape(-1, 1))

                    temp = convolve(image, g.reshape(1, -1))
                    g_out = convolve(temp, h.reshape(-1, 1))

                    # Convolução vertical
                    temp = convolve(image, h.reshape(-1, 1))
                    v_out = convolve(temp, g.reshape(1, -1))

                    # Diagonal
                    temp = convolve(image, g.reshape(-1, 1))
                    d_out = convolve(temp, g.reshape(1, -1))

                    return h_out, v_out, d_out

                return None, None, None

            def reconstruct(self, coefficients):
                """Reconstrução da imagem"""
                current_approx = coefficients[-1]['approximation']

                # Reconstruir nível por nível (do fino para o grosso)
                for level in reversed(range(len(coefficients))):
                    coeff = coefficients[level]

                    # Reconstrução usando filtros
                    reconstructed = self._wavelet_reconstruction(
                        coeff['horizontal'],
                        coeff['vertical'],
                        coeff['diagonal'],
                        coeff['approximation']
                    )

                    if level > 0:
                        # Combinar com nível anterior
                        current_approx = reconstructed

                return current_approx

            def _wavelet_reconstruction(self, h, v, d, approx=None, mode='full'):
                """Reconstrução wavelet"""
                # Implementação simplificada da reconstrução
                if mode == 'approx':
                    # Apenas a aproximação
                    return 0.25 * (h + v + d + (h * v * d))
                else:
                    # Reconstrução completa
                    return approx  # Simplificado

        wavelet_transformer = WaveletTransform(self.image, wavelet_type, levels)
        coefficients = wavelet_transformer.decompose()

        return {
            'coefficients': coefficients,
            'transformer': wavelet_transformer
        }

    def steerable_filters(self, orientations=[0, 45, 90, 135]):
        """
        Filtros direcionais orientáveis
        """
        class SteerableFilters:
            def __init__(self, image, orientations):
                self.image = image
                self.orientations = orientations

            def apply_steerable_pyramid(self):
                """Pirâmide de filtros orientáveis"""
                responses = {}

                for theta in self.orientations:
                    # Filtro Gabor orientado
                    kernel = self._gabor_kernel(theta, sigma=2.0, frequency=0.5)

                    # Convolução
                    response = convolve(self.image, kernel)
                    responses[f'orientation_{theta}'] = response

                # Magnitude total
                magnitude = np.sqrt(sum(response**2 for response in responses.values()))

                return {
                    'responses': responses,
                    'magnitude': magnitude,
                    'phase': np.arctan2(responses['orientation_0'], responses['orientation_90'])
                }

            def _gabor_kernel(self, theta, sigma=2.0, frequency=0.5):
                """Kernel Gabor orientado"""
                ksize = int(6 * sigma)
                if ksize % 2 == 0:
                    ksize += 1

                kernel = np.zeros((ksize, ksize))

                for i in range(ksize):
                    for j in range(ksize):
                        x = i - ksize//2
                        y = j - ksize//2

                        # Rotação
                        x_theta = x * np.cos(theta) + y * np.sin(theta)
                        y_theta = -x * np.sin(theta) + y * np.cos(theta)

                        # Função Gabor
                        gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
                        sinusoidal = np.cos(2 * np.pi * frequency * x_theta)

                        kernel[i, j] = gaussian * sinusoidal

                return kernel / np.sum(np.abs(kernel))  # Normalizar

        steerable_processor = SteerableFilters(self.image, orientations)
        results = steerable_processor.apply_steerable_pyramid()

        return results

    def phase_congruency(self):
        """
        Medida de congruência de fase para detecção de características
        """
        class PhaseCongruency:
            def __init__(self, image):
                self.image = image

            def compute_phase_congruency(self, n_scales=4, n_orientations=6):
                """Computa congruência de fase"""
                # Escalas
                scales = [1.0 / (1.5 ** i) for i in range(n_scales)]

                # Orientações
                orientations = [i * np.pi / n_orientations for i in range(n_orientations)]

                pc_responses = []
                energy_responses = []

                for scale in scales:
                    scale_responses = []

                    for theta in orientations:
                        # Filtro log-Gabor
                        log_gabor = self._log_gabor_filter(scale, theta)

                        # Resposta do filtro
                        response = convolve(self.image, log_gabor)
                        scale_responses.append(response)

                    # Congruência de fase para esta escala
                    pc_scale = self._compute_scale_phase_congruency(scale_responses)
                    pc_responses.append(pc_scale)

                    # Energia total
                    energy = np.sqrt(sum(response**2 for response in scale_responses))
                    energy_responses.append(energy)

                # Congruência total (média ponderada)
                weights = [1.0 / (i + 1) for i in range(n_scales)]
                total_pc = sum(w * pc for w, pc in zip(weights, pc_responses))

                return {
                    'phase_congruency': total_pc,
                    'energy': energy_responses,
                    'scale_responses': pc_responses
                }

            def _log_gabor_filter(self, scale, theta):
                """Filtro log-Gabor"""
                ksize = 31
                kernel = np.zeros((ksize, ksize))

                for i in range(ksize):
                    for j in range(ksize):
                        x = i - ksize//2
                        y = j - ksize//2

                        # Coordenadas polares
                        radius = np.sqrt(x**2 + y**2)
                        angle = np.arctan2(y, x)

                        if radius == 0:
                            continue

                        # Frequência central
                        fo = 1.0 / scale

                        # Largura de banda
                        sigma = 0.65

                        # Filtro angular
                        angular_component = np.exp(-0.5 * ((angle - theta) / (sigma * np.pi/6))**2)

                        # Filtro radial (log-Gabor)
                        radial_component = np.exp(-0.5 * ((np.log(radius/fo)) / sigma)**2)

                        kernel[i, j] = radial_component * angular_component

                return kernel / np.sum(kernel)

            def _compute_scale_phase_congruency(self, responses):
                """Computa congruência de fase para uma escala"""
                # Magnitude das respostas
                magnitudes = [np.abs(response) for response in responses]

                # Energia total
                total_energy = np.sqrt(sum(mag**2 for mag in magnitudes))

                # Congruência de fase
                phase_deviation = np.std([np.angle(response) for response in responses])

                # Medida de congruência
                pc = total_energy / (np.sum(magnitudes) + 1e-10) * (1 - phase_deviation / np.pi)

                return pc

        pc_processor = PhaseCongruency(self.image)
        results = pc_processor.compute_phase_congruency()

        return results

    def spectral_analysis(self):
        """
        Análise espectral de imagens
        """
        class SpectralAnalyzer:
            def __init__(self, image):
                self.image = image

            def fourier_spectrum(self):
                """Espectro de Fourier"""
                # Transformada de Fourier 2D
                f_transform = fft2(self.image)

                # Centralizar
                f_transform_shifted = np.fft.fftshift(f_transform)

                # Magnitude do espectro
                magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)

                # Fase
                phase_spectrum = np.angle(f_transform_shifted)

                return {
                    'magnitude': magnitude_spectrum,
                    'phase': phase_spectrum,
                    'f_transform': f_transform_shifted
                }

            def cepstral_analysis(self):
                """Análise cepstral para detecção de periodicidades"""
                # Log do espectro de magnitude
                log_magnitude = np.log(np.abs(fft2(self.image)) + 1e-10)

                # Transformada de Fourier do log-espectro
                cepstrum = np.real(ifft2(log_magnitude))

                # Cepstro centralizado
                cepstrum_shifted = np.fft.fftshift(cepstrum)

                return cepstrum_shifted

            def wavelet_spectral_analysis(self, wavelet_type='morlet'):
                """Análise espectral usando wavelets"""
                # Implementação simplificada
                scales = np.arange(1, 128, 2)
                wavelet_coeffs = []

                for scale in scales:
                    # Filtro wavelet
                    wavelet = self._generate_wavelet(wavelet_type, scale)

                    # Convolução
                    coeff = convolve(self.image, wavelet)
                    wavelet_coeffs.append(coeff)

                return {
                    'scales': scales,
                    'coefficients': wavelet_coeffs,
                    'scalogram': np.array(wavelet_coeffs)
                }

            def _generate_wavelet(self, wavelet_type, scale):
                """Gera wavelet"""
                if wavelet_type == 'morlet':
                    # Wavelet de Morlet simplificado
                    t = np.linspace(-4, 4, 32)
                    morlet = np.exp(1j * 2 * np.pi * t) * np.exp(-t**2 / 2)
                    morlet = morlet / np.sum(np.abs(morlet))

                    # Escalar
                    scaled_morlet = morlet * (1 / np.sqrt(scale))

                    return scaled_morlet.reshape(-1, 1)  # Para convolução 2D

                return np.ones((1, 1))

        spectral_analyzer = SpectralAnalyzer(self.image)
        fourier_results = spectral_analyzer.fourier_spectrum()
        cepstral_results = spectral_analyzer.cepstral_analysis()
        wavelet_results = spectral_analyzer.wavelet_spectral_analysis()

        return {
            'fourier': fourier_results,
            'cepstral': cepstral_results,
            'wavelet': wavelet_results
        }
```

**Técnicas de Processamento:**
- Transformada wavelet multiresolução
- Filtros direcionais orientáveis
- Congruência de fase
- Análise espectral avançada

### 1.2 Visão Computacional Biológica
```python
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

class BiologicalVision:
    """
    Modelos inspirados na visão biológica
    """

    def __init__(self, image):
        self.image = image

    def retinal_processing(self):
        """
        Processamento retinal inspirado na biologia
        """
        class RetinaModel:
            def __init__(self, image):
                self.image = image
                self.center_surround_ratio = 0.8

            def on_center_off_surround(self):
                """Campo receptivo on-center off-surround"""
                # Centro excitatório
                center_kernel = np.array([
                    [-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]
                ])

                # Surround inibitório
                surround_kernel = np.array([
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]
                ])

                # Resposta center-surround
                center_response = correlate2d(self.image, center_kernel, mode='same')
                surround_response = correlate2d(self.image, surround_kernel, mode='same')

                # Resposta total
                total_response = center_response - self.center_surround_ratio * surround_response

                return total_response

            def off_center_on_surround(self):
                """Campo receptivo off-center on-surround"""
                # Centro inibitório
                center_kernel = np.array([
                    [1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]
                ])

                # Surround excitatório
                surround_kernel = np.array([
                    [-1, -1, -1, -1, -1],
                    [-1, 0, 0, 0, -1],
                    [-1, 0, 0, 0, -1],
                    [-1, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1]
                ])

                center_response = correlate2d(self.image, center_kernel, mode='same')
                surround_response = correlate2d(self.image, surround_kernel, mode='same')

                total_response = center_response - self.center_surround_ratio * surround_response

                return total_response

            def ganglion_cell_responses(self):
                """Respostas de células ganglionares"""
                on_response = self.on_center_off_surround()
                off_response = self.off_center_on_surround()

                # Magnitude
                magnitude = np.sqrt(on_response**2 + off_response**2)

                # Orientação preferencial (simplificada)
                orientation = np.arctan2(on_response, off_response)

                return {
                    'on_response': on_response,
                    'off_response': off_response,
                    'magnitude': magnitude,
                    'orientation': orientation
                }

        retina_model = RetinaModel(self.image)
        retinal_responses = retina_model.ganglion_cell_responses()

        return retinal_responses

    def cortical_processing(self):
        """
        Processamento cortical inspirado no córtex visual primário
        """
        class CortexModel:
            def __init__(self, image):
                self.image = image

            def simple_cells(self):
                """Células simples (resposta a orientação)"""
                orientations = [0, 45, 90, 135]
                frequencies = [0.1, 0.2, 0.3]

                simple_responses = {}

                for theta in orientations:
                    for freq in frequencies:
                        # Filtro Gabor para simular células simples
                        gabor_kernel = self._gabor_kernel(theta, freq, sigma=2.0)

                        response = correlate2d(self.image, gabor_kernel, mode='same')

                        key = f'orientation_{theta}_freq_{freq}'
                        simple_responses[key] = response

                return simple_responses

            def complex_cells(self, simple_responses):
                """Células complexas (invariante a posição)"""
                complex_responses = {}

                # Pooling de células simples
                orientations = [0, 45, 90, 135]

                for theta in orientations:
                    # Máximo sobre frequências e posições próximas
                    responses_theta = [response for key, response in simple_responses.items()
                                     if f'orientation_{theta}' in key]

                    if responses_theta:
                        # Máximo local
                        max_response = np.maximum.reduce(responses_theta)

                        # Pooling espacial
                        kernel_size = 5
                        pooled = self._max_pooling(max_response, kernel_size)

                        complex_responses[f'complex_{theta}'] = pooled

                return complex_responses

            def end_stopped_cells(self, simple_responses):
                """Células end-stopped (detecção de extremos)"""
                end_stopped_responses = {}

                orientations = [0, 45, 90, 135]

                for theta in orientations:
                    responses_theta = [response for key, response in simple_responses.items()
                                     if f'orientation_{theta}' in key]

                    if responses_theta:
                        # Média das respostas
                        avg_response = np.mean(responses_theta, axis=0)

                        # Máscara de extremos (diferença de segunda derivada)
                        second_derivative = self._second_derivative(avg_response)

                        # Ativação para extremos
                        end_stopped = np.maximum(0, -second_derivative) * avg_response

                        end_stopped_responses[f'end_stopped_{theta}'] = end_stopped

                return end_stopped_responses

            def _gabor_kernel(self, theta, frequency, sigma=2.0):
                """Kernel Gabor para simulação de células simples"""
                ksize = 15
                kernel = np.zeros((ksize, ksize))

                for i in range(ksize):
                    for j in range(ksize):
                        x = i - ksize//2
                        y = j - ksize//2

                        # Rotação
                        x_theta = x * np.cos(theta) + y * np.sin(theta)
                        y_theta = -x * np.sin(theta) + y * np.cos(theta)

                        # Função Gabor
                        gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
                        sinusoidal = np.cos(2 * np.pi * frequency * x_theta)

                        kernel[i, j] = gaussian * sinusoidal

                return kernel / np.sum(np.abs(kernel))

            def _max_pooling(self, image, kernel_size):
                """Max pooling para células complexas"""
                pooled = np.zeros_like(image)

                for i in range(0, image.shape[0], kernel_size):
                    for j in range(0, image.shape[1], kernel_size):
                        patch = image[i:i+kernel_size, j:j+kernel_size]
                        pooled[i//kernel_size, j//kernel_size] = np.max(patch)

                return pooled

            def _second_derivative(self, image):
                """Segunda derivada para detecção de extremos"""
                # Laplaciano simples
                laplacian_kernel = np.array([
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]
                ])

                second_deriv = correlate2d(image, laplacian_kernel, mode='same')

                return second_deriv

        cortex_model = CortexModel(self.image)

        # Processamento hierárquico
        simple_responses = cortex_model.simple_cells()
        complex_responses = cortex_model.complex_cells(simple_responses)
        end_stopped_responses = cortex_model.end_stopped_cells(simple_responses)

        return {
            'simple_cells': simple_responses,
            'complex_cells': complex_responses,
            'end_stopped_cells': end_stopped_responses
        }

    def attention_mechanisms(self, saliency_map):
        """
        Mecanismos de atenção inspirados na visão biológica
        """
        class AttentionModel:
            def __init__(self, image, saliency):
                self.image = image
                self.saliency = saliency

            def bottom_up_attention(self):
                """Atenção bottom-up (estímulos)"""
                # Mapa de saliência baseado em características
                intensity_contrast = self._intensity_contrast()
                orientation_contrast = self._orientation_contrast()
                color_contrast = self._color_contrast()

                # Combinar contrastes
                combined_saliency = (intensity_contrast + orientation_contrast + color_contrast) / 3

                return combined_saliency

            def top_down_attention(self, target_features):
                """Atenção top-down (conhecimento prévio)"""
                # Atenção guiada por características alvo
                feature_map = np.zeros_like(self.image)

                for feature in target_features:
                    if feature == 'vertical_lines':
                        # Detectar linhas verticais
                        kernel = np.array([[-1, 2, -1]])
                        feature_response = np.abs(correlate2d(self.image, kernel, mode='same'))
                        feature_map += feature_response
                    elif feature == 'red_objects':
                        # Se imagem colorida, detectar objetos vermelhos
                        if len(self.image.shape) == 3:
                            red_channel = self.image[:, :, 0]
                            feature_map += red_channel

                return feature_map

            def _intensity_contrast(self):
                """Contraste de intensidade"""
                # Diferença local de intensidade
                kernel = np.array([
                    [-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]
                ])

                contrast = np.abs(correlate2d(self.image, kernel, mode='same'))

                return contrast

            def _orientation_contrast(self):
                """Contraste de orientação"""
                orientations = [0, 45, 90, 135]
                orientation_maps = []

                for theta in orientations:
                    kernel = np.array([
                        [-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]
                    ]) * np.cos(theta)  # Simplificado

                    response = np.abs(correlate2d(self.image, kernel, mode='same'))
                    orientation_maps.append(response)

                # Máximo sobre orientações
                orientation_contrast = np.maximum.reduce(orientation_maps)

                return orientation_contrast

            def _color_contrast(self):
                """Contraste de cor"""
                if len(self.image.shape) == 2:
                    return np.zeros_like(self.image)

                # Contraste baseado na diferença de canais
                r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]

                color_contrast = np.sqrt((r - g)**2 + (g - b)**2 + (b - r)**2)

                return color_contrast

        attention_model = AttentionModel(self.image, saliency_map)

        bottom_up = attention_model.bottom_up_attention()
        top_down = attention_model.top_down_attention(['vertical_lines'])

        # Atenção combinada
        combined_attention = 0.7 * bottom_up + 0.3 * top_down

        return {
            'bottom_up': bottom_up,
            'top_down': top_down,
            'combined': combined_attention
        }
```

**Modelos Biológicos:**
- Processamento retinal com campos receptivos
- Arquitetura cortical hierárquica
- Mecanismos de atenção visual
- Células simples, complexas e end-stopped

---

## 2. REDES NEURAIS PARA VISÃO COMPUTACIONAL

### 2.1 Arquiteturas CNN Avançadas
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class AdvancedCNNArchitectures:
    """
    Arquiteturas avançadas de Redes Neurais Convolucionais
    """

    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape

    def residual_network(self, num_classes=1000, num_blocks=[3, 4, 6, 3]):
        """
        Rede Residual (ResNet) para classificação de imagens
        """
        class ResidualBlock(keras.layers.Layer):
            def __init__(self, filters, stride=1, downsample=None):
                super().__init__()
                self.conv1 = keras.layers.Conv2D(filters, 3, stride, padding='same')
                self.bn1 = keras.layers.BatchNormalization()
                self.conv2 = keras.layers.Conv2D(filters, 3, padding='same')
                self.bn2 = keras.layers.BatchNormalization()
                self.downsample = downsample
                self.relu = keras.layers.ReLU()

            def call(self, inputs):
                identity = inputs

                out = self.conv1(inputs)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(inputs)

                out += identity
                out = self.relu(out)

                return out

        def make_layer(filters, blocks, stride=1):
            downsample = None
            if stride != 1 or self.input_shape[-1] != filters:
                downsample = keras.Sequential([
                    keras.layers.Conv2D(filters, 1, stride),
                    keras.layers.BatchNormalization()
                ])

            layers = [ResidualBlock(filters, stride, downsample)]

            for _ in range(1, blocks):
                layers.append(ResidualBlock(filters))

            return keras.Sequential(layers)

        # Arquitetura ResNet
        inputs = keras.Input(shape=self.input_shape)

        # Camada inicial
        x = keras.layers.Conv2D(64, 7, 2, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, 2, padding='same')(x)

        # Camadas residuais
        x = make_layer(64, num_blocks[0])(x)
        x = make_layer(128, num_blocks[1], stride=2)(x)
        x = make_layer(256, num_blocks[2], stride=2)(x)
        x = make_layer(512, num_blocks[3], stride=2)(x)

        # Classificação
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, x)

        return model

    def densely_connected_network(self, num_classes=1000, growth_rate=32, num_blocks=4):
        """
        Rede Densamente Conectada (DenseNet)
        """
        class DenseBlock(keras.layers.Layer):
            def __init__(self, num_layers, growth_rate):
                super().__init__()
                self.layers_list = []

                for _ in range(num_layers):
                    self.layers_list.append(
                        keras.Sequential([
                            keras.layers.BatchNormalization(),
                            keras.layers.ReLU(),
                            keras.layers.Conv2D(growth_rate, 3, padding='same'),
                        ])
                    )

            def call(self, inputs):
                features = [inputs]

                for layer in self.layers_list:
                    concat_features = keras.layers.Concatenate()(features)
                    new_feature = layer(concat_features)
                    features.append(new_feature)

                return keras.layers.Concatenate()(features)

        class TransitionLayer(keras.layers.Layer):
            def __init__(self, reduction=0.5):
                super().__init__()
                self.bn = keras.layers.BatchNormalization()
                self.relu = keras.layers.ReLU()
                self.conv = keras.layers.Conv2D(int(keras.backend.int_shape(self.bn).output[-1] * reduction), 1)
                self.avg_pool = keras.layers.AveragePooling2D(2)

            def call(self, inputs):
                x = self.bn(inputs)
                x = self.relu(x)
                x = self.conv(x)
                x = self.avg_pool(x)
                return x

        # Arquitetura DenseNet
        inputs = keras.Input(shape=self.input_shape)

        # Camada inicial
        x = keras.layers.Conv2D(64, 7, 2, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, 2, padding='same')(x)

        # Blocos densos
        num_layers_per_block = [6, 12, 24, 16][:num_blocks]

        for i, num_layers in enumerate(num_layers_per_block):
            x = DenseBlock(num_layers, growth_rate)(x)

            if i != len(num_layers_per_block) - 1:
                x = TransitionLayer()(x)

        # Classificação
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, x)

        return model

    def attention_augmented_cnn(self, num_classes=1000):
        """
        CNN com Mecanismos de Atenção
        """
        class AttentionBlock(keras.layers.Layer):
            def __init__(self, filters):
                super().__init__()
                self.filters = filters

                # Convoluções para Q, K, V
                self.query_conv = keras.layers.Conv2D(filters // 8, 1)
                self.key_conv = keras.layers.Conv2D(filters // 8, 1)
                self.value_conv = keras.layers.Conv2D(filters, 1)

                self.gamma = self.add_weight(shape=(1,), initializer='zeros')

            def call(self, inputs):
                batch_size, height, width, channels = inputs.shape

                # Projeções
                query = self.query_conv(inputs)  # (B, H, W, C/8)
                key = self.key_conv(inputs)     # (B, H, W, C/8)
                value = self.value_conv(inputs) # (B, H, W, C)

                # Rearranjar para atenção
                query = tf.reshape(query, [batch_size, -1, channels // 8])  # (B, HW, C/8)
                key = tf.reshape(key, [batch_size, -1, channels // 8])
                value = tf.reshape(value, [batch_size, -1, channels])

                # Atenção
                attention = tf.matmul(query, key, transpose_b=True)  # (B, HW, HW)
                attention = tf.nn.softmax(attention / np.sqrt(channels // 8))

                out = tf.matmul(attention, value)  # (B, HW, C)
                out = tf.reshape(out, [batch_size, height, width, channels])

                # Saída com skip connection
                return inputs + self.gamma * out

        # Arquitetura com atenção
        inputs = keras.Input(shape=self.input_shape)

        # Backbone CNN
        x = keras.layers.Conv2D(64, 7, 2, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, 2, padding='same')(x)

        # Blocos com atenção
        for filters in [128, 256, 512]:
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = AttentionBlock(filters)(x)
            x = keras.layers.MaxPooling2D(2)(x)

        # Classificação
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, x)

        return model

    def efficient_net(self, num_classes=1000, compound_coefficient=1):
        """
        EfficientNet com escalonamento composto
        """
        def swish(x):
            return x * tf.nn.sigmoid(x)

        def efficient_block(x, filters, expand_ratio, stride, kernel_size):
            input_filters = x.shape[-1]

            # Expansão
            if expand_ratio != 1:
                x = keras.layers.Conv2D(input_filters * expand_ratio, 1, padding='same')(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation(swish)(x)

            # Profundidade-wise convolution
            x = keras.layers.DepthwiseConv2D(kernel_size, stride, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(swish)(x)

            # Squeeze-and-excitation
            se_filters = max(1, input_filters // 4)
            se = keras.layers.GlobalAveragePooling2D()(x)
            se = keras.layers.Dense(se_filters, activation=swish)(se)
            se = keras.layers.Dense(input_filters, activation='sigmoid')(se)
            se = keras.layers.Reshape((1, 1, input_filters))(se)
            x = x * se

            # Projeção
            x = keras.layers.Conv2D(filters, 1, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)

            return x

        # Parâmetros de escalonamento
        alpha = 1.2 ** compound_coefficient
        beta = 1.1 ** compound_coefficient

        # Configuração da arquitetura
        config = [
            # (expand_ratio, filters, repeats, stride, kernel_size)
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3)
        ]

        inputs = keras.Input(shape=self.input_shape)

        # Camada inicial
        x = keras.layers.Conv2D(int(32 * alpha), 3, 2, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(swish)(x)

        # Blocos EfficientNet
        for expand_ratio, filters, repeats, stride, kernel_size in config:
            filters = int(filters * alpha)

            for i in range(repeats):
                current_stride = stride if i == 0 else 1
                x = efficient_block(x, filters, expand_ratio, current_stride, kernel_size)

        # Cabeçalho
        x = keras.layers.Conv2D(int(1280 * alpha), 1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(swish)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, x)

        return model

    def vision_transformer(self, num_classes=1000, patch_size=16, num_layers=12, num_heads=12):
        """
        Vision Transformer (ViT) para classificação de imagens
        """
        class PatchEmbedding(keras.layers.Layer):
            def __init__(self, patch_size, embed_dim):
                super().__init__()
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                self.projection = keras.layers.Conv2D(embed_dim, patch_size, patch_size)

            def call(self, images):
                patches = self.projection(images)  # (B, H/p, W/p, embed_dim)
                batch_size = tf.shape(patches)[0]
                patch_dims = patches.shape[1] * patches.shape[2]
                patches = tf.reshape(patches, [batch_size, patch_dims, self.embed_dim])
                return patches

        class MultiHeadAttention(keras.layers.Layer):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.num_heads = num_heads
                self.embed_dim = embed_dim
                self.head_dim = embed_dim // num_heads

                self.query_dense = keras.layers.Dense(embed_dim)
                self.key_dense = keras.layers.Dense(embed_dim)
                self.value_dense = keras.layers.Dense(embed_dim)
                self.combine_heads = keras.layers.Dense(embed_dim)

            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]

                # Projeções lineares
                query = self.query_dense(inputs)
                key = self.key_dense(inputs)
                value = self.value_dense(inputs)

                # Dividir em cabeças
                query = tf.reshape(query, [batch_size, -1, self.num_heads, self.head_dim])
                key = tf.reshape(key, [batch_size, -1, self.num_heads, self.head_dim])
                value = tf.reshape(value, [batch_size, -1, self.num_heads, self.head_dim])

                # Atenção
                score = tf.matmul(query, key, transpose_b=True)
                dim_key = tf.cast(self.head_dim, tf.float32)
                scaled_score = score / tf.math.sqrt(dim_key)

                weights = tf.nn.softmax(scaled_score, axis=-1)
                attention = tf.matmul(weights, value)

                # Combinar cabeças
                attention = tf.reshape(attention, [batch_size, -1, self.embed_dim])
                output = self.combine_heads(attention)

                return output

        class TransformerBlock(keras.layers.Layer):
            def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
                super().__init__()
                self.attention = MultiHeadAttention(embed_dim, num_heads)
                self.mlp = keras.Sequential([
                    keras.layers.Dense(mlp_dim, activation='gelu'),
                    keras.layers.Dropout(dropout),
                    keras.layers.Dense(embed_dim),
                    keras.layers.Dropout(dropout)
                ])
                self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = keras.layers.Dropout(dropout)
                self.dropout2 = keras.layers.Dropout(dropout)

            def call(self, inputs):
                # Atenção
                attn_output = self.attention(self.layernorm1(inputs))
                attn_output = self.dropout1(attn_output)
                out1 = inputs + attn_output

                # MLP
                mlp_output = self.mlp(self.layernorm2(out1))
                mlp_output = self.dropout2(mlp_output)
                out2 = out1 + mlp_output

                return out2

        # Parâmetros ViT
        image_size = self.input_shape[0]
        embed_dim = 768
        mlp_dim = 3072

        # Embedding de patches
        inputs = keras.Input(shape=self.input_shape)
        patches = PatchEmbedding(patch_size, embed_dim)(inputs)

        # Positional encoding
        num_patches = (image_size // patch_size) ** 2
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )(positions)

        # Adicionar token [CLS]
        cls_token = tf.Variable(tf.zeros((1, 1, embed_dim)))
        patches = tf.concat([tf.broadcast_to(cls_token, [tf.shape(patches)[0], 1, embed_dim]), patches], axis=1)
        patches += tf.concat([tf.zeros((1, 1, embed_dim)), position_embedding[tf.newaxis, ...]], axis=1)

        # Camadas Transformer
        for _ in range(num_layers):
            patches = TransformerBlock(embed_dim, num_heads, mlp_dim)(patches)

        # Classificação
        cls_output = patches[:, 0, :]
        x = keras.layers.Dense(num_classes, activation='softmax')(cls_output)

        model = keras.Model(inputs, x)

        return model
```

**Arquiteturas Avançadas:**
- ResNet com conexões residuais
- DenseNet com conectividade densa
- CNNs com atenção
- EfficientNet com escalonamento composto
- Vision Transformer

### 2.2 Técnicas de Segmentação Avançadas
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class AdvancedSegmentation:
    """
    Técnicas avançadas de segmentação de imagens
    """

    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape

    def unet_architecture(self, num_classes=21):
        """
        Arquitetura U-Net para segmentação semântica
        """
        def conv_block(x, filters, kernel_size=3, strides=1):
            x = keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            return x

        def encoder_block(x, filters):
            x = conv_block(x, filters)
            x = conv_block(x, filters)
            p = keras.layers.MaxPooling2D(2)(x)
            return x, p

        def decoder_block(x, skip_features, filters):
            x = keras.layers.Conv2DTranspose(filters, 2, 2, padding='same')(x)
            x = keras.layers.Concatenate()([x, skip_features])
            x = conv_block(x, filters)
            x = conv_block(x, filters)
            return x

        # Encoder
        inputs = keras.Input(shape=self.input_shape)

        s1, p1 = encoder_block(inputs, 64)
        s2, p2 = encoder_block(p1, 128)
        s3, p3 = encoder_block(p2, 256)
        s4, p4 = encoder_block(p3, 512)

        # Bridge
        b1 = conv_block(p4, 1024)
        b1 = conv_block(b1, 1024)

        # Decoder
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)

        # Saída
        outputs = keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(d4)

        model = keras.Model(inputs, outputs)

        return model

    def deeplab_v3_plus(self, num_classes=21, output_stride=16):
        """
        DeepLab v3+ para segmentação semântica precisa
        """
        def atrous_conv_block(x, filters, rate):
            x = keras.layers.Conv2D(filters, 3, padding='same', dilation_rate=rate)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            return x

        def aspp_block(x, filters):
            # Múltiplas taxas de dilatação
            conv_1x1 = keras.layers.Conv2D(filters, 1, padding='same')(x)
            conv_1x1 = keras.layers.BatchNormalization()(x)
            conv_1x1 = keras.layers.ReLU()(x)

            conv_3x3_rate6 = atrous_conv_block(x, filters, 6)
            conv_3x3_rate12 = atrous_conv_block(x, filters, 12)
            conv_3x3_rate18 = atrous_conv_block(x, filters, 18)

            # Pooling global
            global_pool = keras.layers.GlobalAveragePooling2D()(x)
            global_pool = keras.layers.Reshape((1, 1, -1))(global_pool)
            global_pool = keras.layers.Conv2D(filters, 1, padding='same')(global_pool)
            global_pool = keras.layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(global_pool)

            # Concatenação
            x = keras.layers.Concatenate()([
                conv_1x1, conv_3x3_rate6, conv_3x3_rate12, conv_3x3_rate18, global_pool
            ])

            x = keras.layers.Conv2D(filters, 1, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

            return x

        # Backbone (ResNet-like)
        inputs = keras.Input(shape=self.input_shape)

        # Encoder
        x = keras.layers.Conv2D(64, 7, 2, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, 2, padding='same')(x)

        # Blocos residuais simplificados
        for filters in [64, 128, 256]:
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        # ASPP
        x = aspp_block(x, 256)

        # Decoder
        x = keras.layers.Conv2DTranspose(256, 2, 2, padding='same')(x)
        x = keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)

        model = keras.Model(inputs, x)

        return model

    def mask_rcnn(self, num_classes=80):
        """
        Mask R-CNN para segmentação de instâncias
        """
        class RegionProposalNetwork(keras.layers.Layer):
            def __init__(self, num_anchors=9):
                super().__init__()
                self.num_anchors = num_anchors

                # Convoluções compartilhadas
                self.conv = keras.layers.Conv2D(512, 3, padding='same')
                self.cls_conv = keras.layers.Conv2D(num_anchors * 2, 1)
                self.reg_conv = keras.layers.Conv2D(num_anchors * 4, 1)

            def call(self, feature_map):
                x = self.conv(feature_map)
                x = keras.layers.ReLU()(x)

                # Classificação de âncoras
                cls_scores = self.cls_conv(x)
                cls_scores = keras.layers.Reshape((-1, 2))(cls_scores)

                # Regressão de bounding boxes
                bbox_regs = self.reg_conv(x)
                bbox_regs = keras.layers.Reshape((-1, 4))(bbox_regs)

                return cls_scores, bbox_regs

        class RoIAlign(keras.layers.Layer):
            def __init__(self, pool_size=(7, 7)):
                super().__init__()
                self.pool_size = pool_size

            def call(self, feature_map, rois):
                # Implementação simplificada do RoI Align
                # Em produção, usar implementação TensorFlow completa
                return tf.image.crop_and_resize(
                    feature_map, rois, tf.range(tf.shape(rois)[0]),
                    self.pool_size
                )

        # Backbone (ResNet)
        inputs = keras.Input(shape=self.input_shape)

        # Feature extraction
        x = keras.layers.Conv2D(64, 7, 2, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        feature_map = keras.layers.MaxPooling2D(3, 2, padding='same')(x)

        # RPN
        rpn = RegionProposalNetwork()
        cls_scores, bbox_regs = rpn(feature_map)

        # RoI Align (simplificado)
        roi_align = RoIAlign()
        pooled_features = roi_align(feature_map, bbox_regs)  # Simplificado

        # Cabeçalhos para classificação e máscara
        cls_head = keras.layers.Dense(num_classes + 1, activation='softmax')(pooled_features)
        mask_head = keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(pooled_features)

        model = keras.Model(inputs, [cls_head, mask_head, cls_scores, bbox_regs])

        return model

    def panoptic_segmentation(self, num_classes=21, num_instances=100):
        """
        Segmentação Panóptica (semântica + instâncias)
        """
        def semantic_branch(feature_map):
            """Ramo semântico"""
            x = keras.layers.Conv2D(256, 3, padding='same')(feature_map)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv2D(num_classes, 1, activation='softmax')(x)
            return x

        def instance_branch(feature_map):
            """Ramo de instâncias"""
            x = keras.layers.Conv2D(256, 3, padding='same')(feature_map)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

            # Centros de instâncias
            centers = keras.layers.Conv2D(num_instances, 1, activation='sigmoid')(x)

            # Bounding boxes
            bboxes = keras.layers.Conv2D(num_instances * 4, 1)(x)

            return centers, bboxes

        # Backbone
        inputs = keras.Input(shape=self.input_shape)

        x = keras.layers.Conv2D(64, 7, 2, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        feature_map = keras.layers.MaxPooling2D(3, 2, padding='same')(x)

        # Ramos
        semantic_output = semantic_branch(feature_map)
        centers_output, bboxes_output = instance_branch(feature_map)

        model = keras.Model(inputs, [semantic_output, centers_output, bboxes_output])

        return model

    def optical_flow_estimation(self):
        """
        Estimativa de fluxo óptico para segmentação temporal
        """
        def flow_net_block(x, filters):
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            return x

        # Entradas: frame atual e anterior
        frame1 = keras.Input(shape=self.input_shape)
        frame2 = keras.Input(shape=self.input_shape)

        # Concatenação
        combined = keras.layers.Concatenate()([frame1, frame2])

        # Encoder
        x = flow_net_block(combined, 64)
        x = keras.layers.MaxPooling2D(2)(x)
        x = flow_net_block(x, 128)
        x = keras.layers.MaxPooling2D(2)(x)
        x = flow_net_block(x, 256)
        x = keras.layers.MaxPooling2D(2)(x)

        # Decoder com skip connections
        x = keras.layers.Conv2DTranspose(128, 2, 2, padding='same')(x)
        x = flow_net_block(x, 128)
        x = keras.layers.Conv2DTranspose(64, 2, 2, padding='same')(x)
        x = flow_net_block(x, 64)

        # Fluxo óptico (2 canais: u, v)
        flow = keras.layers.Conv2D(2, 3, padding='same')(x)

        model = keras.Model([frame1, frame2], flow)

        return model
```

**Técnicas de Segmentação:**
- U-Net para segmentação médica
- DeepLab v3+ para segmentação semântica
- Mask R-CNN para segmentação de instâncias
- Segmentação panóptica
- Estimativa de fluxo óptico

---

## 3. APRENDIZADO PROFUNDO AVANÇADO

### 3.1 Redes Generativas para Visão
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class GenerativeVisionModels:
    """
    Modelos generativos para tarefas de visão computacional
    """

    def __init__(self, image_shape=(64, 64, 3)):
        self.image_shape = image_shape

    def variational_autoencoder(self, latent_dim=128):
        """
        Autoencoder Variacional (VAE) para geração de imagens
        """
        class Sampling(keras.layers.Layer):
            """Camada de amostragem para VAE"""
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Encoder
        encoder_inputs = keras.Input(shape=self.image_shape)
        x = keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
        x = keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(16, activation='relu')(x)

        z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = keras.layers.Dense(16 * 16 * 64, activation='relu')(latent_inputs)
        x = keras.layers.Reshape((16, 16, 64))(x)
        x = keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        x = keras.layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        decoder_outputs = keras.layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

        decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

        # VAE completo
        vae_outputs = decoder(encoder(encoder_inputs)[2])
        vae = keras.Model(encoder_inputs, vae_outputs, name='vae')

        # Função de perda
        def vae_loss(y_true, y_pred):
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y_true, y_pred), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            return reconstruction_loss + kl_loss

        vae.compile(optimizer='adam', loss=vae_loss)

        return vae, encoder, decoder

    def generative_adversarial_network(self, latent_dim=100):
        """
        Rede Generativa Antagônica (GAN) para geração de imagens
        """
        def build_generator(latent_dim):
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_dim=latent_dim),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(np.prod(self.image_shape), activation='tanh'),
                keras.layers.Reshape(self.image_shape)
            ])
            return model

        def build_discriminator():
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=self.image_shape),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            return model

        # Construir modelos
        generator = build_generator(latent_dim)
        discriminator = build_discriminator()

        # Compilar discriminador
        discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Construir GAN
        discriminator.trainable = False

        gan_input = keras.Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        gan = keras.Model(gan_input, gan_output)

        gan.compile(optimizer='adam', loss='binary_crossentropy')

        return gan, generator, discriminator

    def style_gan(self, latent_dim=512):
        """
        StyleGAN para geração de imagens de alta qualidade
        """
        class AdaIN(keras.layers.Layer):
            """Adaptive Instance Normalization"""
            def __init__(self, epsilon=1e-7):
                super().__init__()
                self.epsilon = epsilon

            def call(self, inputs):
                content, style = inputs

                # Estatísticas do conteúdo
                content_mean, content_var = tf.nn.moments(content, axes=[1, 2], keepdims=True)

                # Estatísticas do estilo
                style_mean = style[:, :content.shape[-1]]
                style_var = style[:, content.shape[-1]:]

                # Normalização adaptativa
                normalized = (content - content_mean) / tf.sqrt(content_var + self.epsilon)
                adain = normalized * tf.reshape(style_var, [-1, 1, 1, content.shape[-1]]) + \
                       tf.reshape(style_mean, [-1, 1, 1, content.shape[-1]])

                return adain

        def style_block(x, filters, latent_dim):
            # Mapeamento de estilo
            style = keras.layers.Dense(2 * filters)(keras.Input(shape=(latent_dim,)))

            # Convolução com AdaIN
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = AdaIN()([x, style])

            return x

        # Generator simplificado
        latent_input = keras.Input(shape=(latent_dim,))

        # Mapeamento inicial
        x = keras.layers.Dense(4 * 4 * 512)(latent_input)
        x = keras.layers.Reshape((4, 4, 512))(x)

        # Blocos de aumento de resolução
        for filters in [256, 128, 64]:
            x = keras.layers.UpSampling2D(2)(x)
            x = style_block(x, filters, latent_dim)

        # Saída
        output = keras.layers.Conv2D(3, 3, padding='same', activation='tanh')(x)

        generator = keras.Model(latent_input, output)

        return generator

    def diffusion_model(self, num_timesteps=1000):
        """
        Modelo de Difusão para geração de imagens
        """
        class DiffusionModel:
            def __init__(self, image_shape, num_timesteps):
                self.image_shape = image_shape
                self.num_timesteps = num_timesteps

                # Redes neurais para difusão
                self.forward_process = self.build_forward_process()
                self.reverse_process = self.build_reverse_process()

            def build_forward_process(self):
                """Processo de difusão forward (adiciona ruído)"""
                inputs = keras.Input(shape=self.image_shape)
                timestep = keras.Input(shape=(1,))

                # Incorporação de tempo
                time_embedding = keras.layers.Dense(128)(timestep)
                time_embedding = keras.layers.ReLU()(time_embedding)

                # U-Net simplificada para denoising
                x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
                x = keras.layers.Concatenate()([x, time_embedding[:, tf.newaxis, tf.newaxis, :]])
                x = keras.layers.Conv2D(64, 3, padding='same')(x)

                # Adicionar ruído
                noise = keras.layers.GaussianNoise(0.1)(x)

                model = keras.Model([inputs, timestep], noise)
                return model

            def build_reverse_process(self):
                """Processo reverso (remove ruído)"""
                inputs = keras.Input(shape=self.image_shape)
                timestep = keras.Input(shape=(1,))

                # Mesma arquitetura do forward
                time_embedding = keras.layers.Dense(128)(timestep)
                time_embedding = keras.layers.ReLU()(time_embedding)

                x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
                x = keras.layers.Concatenate()([x, time_embedding[:, tf.newaxis, tf.newaxis, :]])
                x = keras.layers.Conv2D(64, 3, padding='same')(x)

                # Predição de ruído
                predicted_noise = keras.layers.Conv2D(3, 3, padding='same')(x)

                model = keras.Model([inputs, timestep], predicted_noise)
                return model

            def diffuse_image(self, image, t):
                """Adiciona ruído a uma imagem no passo t"""
                # Coeficientes de difusão
                beta_t = 0.02  # Simplificado
                alpha_t = 1 - beta_t

                # Adicionar ruído
                noise = tf.random.normal(shape=image.shape)
                diffused = tf.sqrt(alpha_t) * image + tf.sqrt(1 - alpha_t) * noise

                return diffused, noise

            def denoise_image(self, noisy_image, t):
                """Remove ruído de uma imagem"""
                predicted_noise = self.reverse_process([noisy_image, tf.constant([[t]])])

                # Coeficientes de reversão
                beta_t = 0.02
                alpha_t = 1 - beta_t
                alpha_t_minus_1 = 1 - 0.02  # Simplificado

                # Reversão do processo
                denoised = (noisy_image - tf.sqrt(1 - alpha_t) * predicted_noise) / tf.sqrt(alpha_t)
                denoised = tf.sqrt(alpha_t_minus_1) * denoised

                return denoised

        diffusion_model = DiffusionModel(self.image_shape, num_timesteps)

        return diffusion_model

    def normalizing_flows(self, num_layers=8):
        """
        Normalizing Flows para geração de imagens
        """
        class CouplingLayer(keras.layers.Layer):
            """Camada de acoplamento para Normalizing Flow"""
            def __init__(self, input_dim, hidden_dim=256):
                super().__init__()
                self.input_dim = input_dim
                self.scale_net = keras.Sequential([
                    keras.layers.Dense(hidden_dim, activation='relu'),
                    keras.layers.Dense(hidden_dim, activation='relu'),
                    keras.layers.Dense(input_dim // 2)
                ])
                self.translate_net = keras.Sequential([
                    keras.layers.Dense(hidden_dim, activation='relu'),
                    keras.layers.Dense(hidden_dim, activation='relu'),
                    keras.layers.Dense(input_dim // 2)
                ])

            def call(self, inputs):
                x1, x2 = tf.split(inputs, 2, axis=-1)

                # Transformações
                scale = tf.exp(self.scale_net(x1))
                translate = self.translate_net(x1)

                y1 = x1
                y2 = x2 * scale + translate

                # Log-determinante
                log_det = tf.reduce_sum(tf.math.log(scale), axis=-1)

                return tf.concat([y1, y2], axis=-1), log_det

        class NormalizingFlow:
            def __init__(self, input_dim, num_layers):
                self.layers = [CouplingLayer(input_dim) for _ in range(num_layers)]

            def forward(self, x):
                """Transformação forward (dados -> latente)"""
                log_det_total = 0

                for layer in self.layers:
                    x, log_det = layer(x)
                    log_det_total += log_det

                return x, log_det_total

            def inverse(self, z):
                """Transformação inversa (latente -> dados)"""
                log_det_total = 0

                for layer in reversed(self.layers):
                    z, log_det = layer.inverse(z)
                    log_det_total += log_det

                return z, log_det_total

            def log_probability(self, x):
                """Log-probabilidade dos dados"""
                z, log_det = self.forward(x)

                # Prior (distribuição normal)
                log_prior = -0.5 * tf.reduce_sum(z**2 + tf.math.log(2 * np.pi), axis=-1)

                return log_prior + log_det

        # Flatten image dimensions
        input_dim = np.prod(self.image_shape)
        flow = NormalizingFlow(input_dim, num_layers)

        return flow
```

**Modelos Generativos:**
- Autoencoders Variacionais (VAE)
- Redes Generativas Antagônicas (GAN)
- StyleGAN para geração de alta qualidade
- Modelos de Difusão
- Normalizing Flows

---

## 4. CONSIDERAÇÕES FINAIS

A visão computacional avançada representa a convergência entre processamento de imagens clássico, aprendizado profundo e modelos generativos. Os métodos apresentados fornecem ferramentas para:

1. **Processamento Avançado**: Técnicas de transformadas e filtros orientáveis
2. **Modelos Biológicos**: Inspiração na visão humana para melhor desempenho
3. **Arquiteturas Profundas**: CNNs, Transformers e modelos de atenção
4. **Segmentação Inteligente**: Técnicas precisas para diversas aplicações
5. **Geração Criativa**: Modelos generativos para síntese de imagens

**Próximos Passos Recomendados**:
1. Dominar fundamentos de processamento de imagens
2. Explorar arquiteturas de redes neurais profundas
3. Implementar modelos biológicos inspirados
4. Experimentar com tarefas de segmentação avançadas
5. Desenvolver aplicações generativas criativas

---

*Documento preparado para fine-tuning de IA em Visão Computacional Avançada*
*Versão 1.0 - Preparado para implementação prática*
