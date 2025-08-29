# FT-MAT-001: Fine-Tuning para IA em Materiais Avançados

## Visão Geral do Projeto

Este documento estabelece diretrizes para o fine-tuning de modelos de IA especializados em materiais avançados, integrando modelagem computacional de materiais funcionais, compósitos, cerâmicas, metais e polímeros com princípios da física dos materiais, química computacional e mecânica quântica.

### Contexto Filosófico
Os materiais avançados representam a materialização da inovação tecnológica, onde propriedades emergentes surgem da combinação inteligente de constituintes em diferentes escalas. Esta abordagem reconhece que o comportamento dos materiais não é simplesmente a soma das partes, mas emerge da interação complexa entre estrutura, processamento e ambiente.

### Metodologia de Aprendizado Recomendada
1. **Fundamentos de Materiais**: Compreensão de estrutura, processamento e propriedades
2. **Modelagem Computacional**: Desenvolvimento de modelos multiescala
3. **Caracterização Avançada**: Técnicas experimentais e computacionais
4. **Projeto e Otimização**: Integração de ML para descoberta de materiais
5. **Aplicações Integradas**: Síntese de conhecimento teórico com aplicações práticas

---

## 1. MODELAGEM COMPUTACIONAL DE MATERIAIS

### 1.1 Mecânica Quântica de Materiais
```python
import numpy as np
from scipy.linalg import eigh
from scipy.constants import hbar, e, m_e
import matplotlib.pyplot as plt

class QuantumMaterialsModeling:
    """
    Modelagem quântica de materiais avançados
    """

    def __init__(self):
        self.hbar = hbar
        self.e = e
        self.m_e = m_e

    def tight_binding_model(self, lattice_sites, hopping_parameters):
        """
        Modelo tight-binding para bandas eletrônicas
        """
        class TightBindingModel:
            def __init__(self, sites, hopping):
                self.sites = sites
                self.hopping = hopping
                self.hamiltonian = self._build_hamiltonian()

            def _build_hamiltonian(self):
                """Construir matriz Hamiltoniana"""
                n_sites = len(self.sites)
                H = np.zeros((n_sites, n_sites), dtype=complex)

                # Energia no sítio (diagonal)
                for i in range(n_sites):
                    H[i, i] = self.hopping.get('onsite', 0)

                # Hopping entre sítios (off-diagonal)
                for i in range(n_sites):
                    for j in range(n_sites):
                        if i != j:
                            distance = np.linalg.norm(self.sites[i] - self.sites[j])
                            # Parâmetro de hopping baseado na distância
                            t_ij = self.hopping.get('t', -1.0) * np.exp(-distance / self.hopping.get('decay', 1.0))
                            H[i, j] = t_ij

                return H

            def calculate_band_structure(self, k_points):
                """Calcular estrutura de bandas"""
                eigenvalues_k = []
                eigenvectors_k = []

                for k in k_points:
                    # Hamiltoniana no espaço-k
                    H_k = self._fourier_transform_hamiltonian(k)
                    eigenvalues, eigenvectors = eigh(H_k)

                    eigenvalues_k.append(eigenvalues)
                    eigenvectors_k.append(eigenvectors)

                return {
                    'k_points': k_points,
                    'eigenvalues': np.array(eigenvalues_k),
                    'eigenvectors': eigenvectors_k,
                    'fermi_level': self._calculate_fermi_level(np.array(eigenvalues_k))
                }

            def _fourier_transform_hamiltonian(self, k):
                """Transformada de Fourier da Hamiltoniana"""
                n_sites = len(self.sites)
                H_k = np.zeros((n_sites, n_sites), dtype=complex)

                for i in range(n_sites):
                    for j in range(n_sites):
                        phase = np.exp(-1j * k * (self.sites[i][0] - self.sites[j][0]))
                        H_k[i, j] = self.hamiltonian[i, j] * phase

                return H_k

            def _calculate_fermi_level(self, eigenvalues_k):
                """Calcular nível de Fermi"""
                # Para meio preenchimento (aproximado)
                all_eigenvalues = eigenvalues_k.flatten()
                fermi_idx = len(all_eigenvalues) // 2
                fermi_level = np.sort(all_eigenvalues)[fermi_idx]

                return fermi_level

            def calculate_density_of_states(self, eigenvalues_k, energy_range):
                """Calcular densidade de estados"""
                energies = np.linspace(energy_range[0], energy_range[1], 1000)
                dos = np.zeros_like(energies)

                all_eigenvalues = eigenvalues_k.flatten()

                for eigenvalue in all_eigenvalues:
                    # Distribuição Gaussiana
                    dos += np.exp(-((energies - eigenvalue) / 0.1)**2)

                dos = dos / (np.sqrt(2 * np.pi) * 0.1)  # Normalização

                return energies, dos

        tb_model = TightBindingModel(lattice_sites, hopping_parameters)
        k_points = np.linspace(0, 2*np.pi, 50)
        band_structure = tb_model.calculate_band_structure(k_points)

        return {
            'tight_binding_model': tb_model,
            'band_structure': band_structure,
            'density_of_states': tb_model.calculate_density_of_states(
                band_structure['eigenvalues'], [-3, 3]
            )
        }

    def density_functional_theory_dft(self, atomic_positions, atomic_numbers):
        """
        Teoria do Funcional da Densidade (DFT) simplificada
        """
        class DFTCalculator:
            def __init__(self, positions, numbers):
                self.positions = positions
                self.atomic_numbers = numbers
                self.grid = self._create_real_space_grid()

            def _create_real_space_grid(self):
                """Criar grade no espaço real"""
                # Grade simples 3D
                n_points = 20
                x = np.linspace(-5, 5, n_points)
                y = np.linspace(-5, 5, n_points)
                z = np.linspace(-5, 5, n_points)

                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

                return {
                    'x': X, 'y': Y, 'z': Z,
                    'shape': X.shape,
                    'n_points': n_points**3
                }

            def calculate_electron_density(self):
                """Calcular densidade eletrônica"""
                grid = self.grid

                # Densidade inicial (superposição de orbitais atômicos)
                rho = np.zeros(grid['shape'])

                for pos, Z in zip(self.positions, self.atomic_numbers):
                    # Orbital atômico simplificado (Gaussiana)
                    r_squared = (grid['x'] - pos[0])**2 + (grid['y'] - pos[1])**2 + (grid['z'] - pos[2])**2
                    atomic_rho = Z * np.exp(-r_squared)  # Densidade atômica aproximada
                    rho += atomic_rho

                return rho

            def calculate_kinetic_energy(self, wavefunctions):
                """Calcular energia cinética"""
                T = 0

                for psi in wavefunctions:
                    # Laplaciano numérico
                    grad_psi = np.gradient(psi)
                    laplacian = np.sum([np.gradient(grad)[i] for i, grad in enumerate(np.gradient(psi))], axis=0)

                    T += -0.5 * np.sum(psi * laplacian) * (10/20)**3  # Integral numérica

                return T

            def calculate_exchange_correlation_energy(self, rho):
                """Calcular energia de troca-correlação (aproximada LDA)"""
                # Aproximação LDA simples
                alpha = (4/(9*np.pi))**(1/3)
                rs = (3/(4*np.pi * rho))**(1/3)

                # Energia de troca
                epsilon_x = -0.916 / rs

                # Energia de correlação (aproximação)
                epsilon_c = -0.096 + 0.0622 * np.log(rs)

                E_xc = np.sum((epsilon_x + epsilon_c) * rho) * (10/20)**3

                return E_xc

            def calculate_total_energy(self):
                """Calcular energia total do sistema"""
                rho = self.calculate_electron_density()

                # Energia cinética (estimada)
                T_s = 0.5 * np.sum(rho**(5/3)) * (10/20)**3

                # Energia de troca-correlação
                E_xc = self.calculate_exchange_correlation_energy(rho)

                # Energia potencial (nucleo-eletron)
                E_ext = 0
                for pos, Z in zip(self.positions, self.atomic_numbers):
                    r_squared = (self.grid['x'] - pos[0])**2 + (self.grid['y'] - pos[1])**2 + (self.grid['z'] - pos[2])**2
                    E_ext += -Z * np.sum(rho / np.sqrt(r_squared + 1)) * (10/20)**3

                # Energia de Hartree
                E_hartree = 0.5 * np.sum(rho * self._calculate_hartree_potential(rho)) * (10/20)**3

                total_energy = T_s + E_ext + E_hartree + E_xc

                return {
                    'total_energy': total_energy,
                    'kinetic_energy': T_s,
                    'external_energy': E_ext,
                    'hartree_energy': E_hartree,
                    'xc_energy': E_xc,
                    'electron_density': rho
                }

            def _calculate_hartree_potential(self, rho):
                """Calcular potencial de Hartree"""
                # Solução aproximada da equação de Poisson
                # ∇²φ = -4πρ
                # Solução no espaço de Fourier
                rho_fft = np.fft.fftn(rho)
                k_squared = np.sum(np.meshgrid(*[np.fft.fftfreq(s) * 2*np.pi/s for s in rho.shape], indexing='ij'), axis=0)**2

                # Evitar divisão por zero
                k_squared = np.where(k_squared == 0, 1e-10, k_squared)

                phi_fft = -4 * np.pi * rho_fft / k_squared
                phi = np.fft.ifftn(phi_fft).real

                return phi

        dft_calc = DFTCalculator(atomic_positions, atomic_numbers)
        energy_calculation = dft_calc.calculate_total_energy()

        return energy_calculation

    def molecular_dynamics_materials(self, material_structure, temperature):
        """
        Dinâmica molecular para materiais
        """
        class MaterialsMD:
            def __init__(self, structure, temp):
                self.structure = structure
                self.temperature = temp
                self.velocities = self._initialize_velocities()

            def _initialize_velocities(self):
                """Inicializar velocidades segundo Maxwell-Boltzmann"""
                n_atoms = len(self.structure['positions'])
                velocities = np.random.normal(0, np.sqrt(self.temperature), (n_atoms, 3))

                # Remover momento linear total
                total_momentum = np.sum(velocities, axis=0)
                velocities -= total_momentum / n_atoms

                return velocities

            def calculate_forces(self, positions):
                """Calcular forças usando potencial de Lennard-Jones"""
                forces = np.zeros_like(positions)
                n_atoms = len(positions)

                epsilon = 1.0  # Profundidade do poço
                sigma = 1.0    # Diâmetro atômico

                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        r_ij = positions[j] - positions[i]
                        r = np.linalg.norm(r_ij)

                        if r > 0.1:  # Evitar singularidade
                            # Força de Lennard-Jones
                            sr6 = (sigma / r)**6
                            sr12 = sr6**2

                            force_magnitude = 48 * epsilon * (sr12 - 0.5 * sr6) / r**2
                            force_ij = force_magnitude * r_ij / r

                            forces[i] -= force_ij
                            forces[j] += force_ij

                return forces

            def verlet_integration(self, n_steps=1000, dt=0.001):
                """Integração Verlet para dinâmica molecular"""
                positions = self.structure['positions'].copy()
                velocities = self.velocities.copy()
                masses = np.ones(len(positions))  # Massas unitárias

                trajectory = [positions.copy()]
                energies = []

                # Forças iniciais
                forces = self.calculate_forces(positions)

                for step in range(n_steps):
                    # Atualizar posições
                    positions += velocities * dt + 0.5 * forces / masses[:, np.newaxis] * dt**2

                    # Forças nas novas posições
                    new_forces = self.calculate_forces(positions)

                    # Atualizar velocidades
                    velocities += 0.5 * (forces + new_forces) / masses[:, np.newaxis] * dt

                    # Atualizar forças
                    forces = new_forces

                    # Calcular energia
                    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
                    potential_energy = self._calculate_potential_energy(positions)
                    total_energy = kinetic_energy + potential_energy

                    energies.append({
                        'kinetic': kinetic_energy,
                        'potential': potential_energy,
                        'total': total_energy,
                        'temperature': 2 * kinetic_energy / (3 * len(positions))
                    })

                    # Salvar trajetória
                    if step % 10 == 0:
                        trajectory.append(positions.copy())

                return {
                    'trajectory': trajectory,
                    'energies': energies,
                    'final_positions': positions,
                    'final_velocities': velocities
                }

            def _calculate_potential_energy(self, positions):
                """Calcular energia potencial"""
                energy = 0
                n_atoms = len(positions)

                epsilon = 1.0
                sigma = 1.0

                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        r = np.linalg.norm(positions[i] - positions[j])

                        if r > 0.1:
                            sr6 = (sigma / r)**6
                            energy += 4 * epsilon * (sr6**2 - sr6)

                return energy

        md_sim = MaterialsMD(material_structure, temperature)
        simulation_results = md_sim.verlet_integration()

        return simulation_results

    def phonon_spectra_calculation(self, crystal_structure, force_constants):
        """
        Cálculo de espectros fonônicos
        """
        class PhononCalculator:
            def __init__(self, structure, force_constants):
                self.structure = structure
                self.force_constants = force_constants

            def build_dynamical_matrix(self, q_point):
                """Construir matriz dinâmica"""
                n_atoms = len(self.structure['positions'])
                dynamical_matrix = np.zeros((3*n_atoms, 3*n_atoms), dtype=complex)

                # Constantes de força entre átomos
                for i in range(n_atoms):
                    for j in range(n_atoms):
                        if i != j:
                            r_ij = self.structure['positions'][j] - self.structure['positions'][i]
                            distance = np.linalg.norm(r_ij)

                            # Fase de Bloch
                            phase = np.exp(-1j * np.dot(q_point, r_ij))

                            # Matriz de constantes de força (simplificada)
                            phi_ij = self.force_constants.get(distance, np.zeros((3, 3)))

                            # Preencher matriz dinâmica
                            for alpha in range(3):
                                for beta in range(3):
                                    dynamical_matrix[3*i + alpha, 3*j + beta] = phi_ij[alpha, beta] * phase

                return dynamical_matrix

            def calculate_phonon_frequencies(self, q_points):
                """Calcular frequências fonônicas"""
                frequencies_q = []

                for q in q_points:
                    dyn_matrix = self.build_dynamical_matrix(q)

                    # Diagonalizar matriz dinâmica
                    eigenvalues, eigenvectors = eigh(dyn_matrix)

                    # Frequências (positivas)
                    frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)  # Em Hz

                    frequencies_q.append(frequencies)

                return {
                    'q_points': q_points,
                    'frequencies': np.array(frequencies_q),
                    'phonon_branches': len(frequencies_q[0]) if frequencies_q else 0
                }

            def calculate_phonon_dos(self, frequencies_q):
                """Calcular densidade de estados fonônicos"""
                all_frequencies = frequencies_q.flatten()

                # Histograma das frequências
                freq_bins = np.linspace(0, np.max(all_frequencies), 100)
                dos, _ = np.histogram(all_frequencies, bins=freq_bins, density=True)

                return freq_bins[:-1], dos

            def calculate_thermal_properties(self, frequencies_q, temperatures):
                """Calcular propriedades térmicas"""
                thermal_props = {}

                for T in temperatures:
                    # Capacidade calorífica
                    cv = self._calculate_heat_capacity(frequencies_q, T)

                    # Energia livre de Helmholtz
                    f_helmholtz = self._calculate_helmholtz_free_energy(frequencies_q, T)

                    thermal_props[T] = {
                        'heat_capacity': cv,
                        'helmholtz_free_energy': f_helmholtz
                    }

                return thermal_props

            def _calculate_heat_capacity(self, frequencies_q, temperature):
                """Calcular capacidade calorífica"""
                hbar_omega = frequencies_q * 2 * np.pi * hbar
                kT = 8.617e-5 * temperature  # eV

                # Função de Bose
                n_bose = 1 / (np.exp(hbar_omega / kT) - 1)

                # Capacidade calorífica por modo
                cv_modes = (hbar_omega / kT)**2 * np.exp(hbar_omega / kT) / (np.exp(hbar_omega / kT) - 1)**2

                return np.sum(cv_modes) * 1.602e-19  # J/mol/K

            def _calculate_helmholtz_free_energy(self, frequencies_q, temperature):
                """Calcular energia livre de Helmholtz"""
                hbar_omega = frequencies_q * 2 * np.pi * hbar
                kT = 8.617e-5 * temperature

                f = 0.5 * np.sum(hbar_omega) + kT * np.sum(np.log(1 - np.exp(-hbar_omega / kT)))

                return f

        phonon_calc = PhononCalculator(crystal_structure, force_constants)
        q_points = np.random.rand(10, 3) * 2 * np.pi  # Pontos q aleatórios
        phonon_spectra = phonon_calc.calculate_phonon_frequencies(q_points)

        return {
            'phonon_calculator': phonon_calc,
            'phonon_spectra': phonon_spectra,
            'phonon_dos': phonon_calc.calculate_phonon_dos(phonon_spectra['frequencies'])
        }
```

**Mecânica Quântica de Materiais:**
- Modelo tight-binding para bandas eletrônicas
- Teoria do Funcional da Densidade (DFT)
- Dinâmica molecular para materiais
- Cálculo de espectros fonônicos

### 1.2 Modelos de Microestrutura
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MicrostructureModeling:
    """
    Modelagem de microestrutura de materiais
    """

    def __init__(self):
        self.phase_field_params = {
            'interface_width': 1.0,
            'mobility': 1.0,
            'epsilon': 1.0
        }

    def phase_field_microstructure(self, initial_microstructure, time_steps=100):
        """
        Evolução de microestrutura usando campo de fase
        """
        class PhaseFieldModel:
            def __init__(self, microstructure, params):
                self.microstructure = microstructure.copy()
                self.params = params

            def evolve_microstructure(self, n_steps):
                """Evoluir microestrutura no tempo"""
                evolution_history = [self.microstructure.copy()]

                for step in range(n_steps):
                    # Calcular energia livre
                    free_energy = self._calculate_free_energy(self.microstructure)

                    # Calcular força química
                    chemical_force = -np.gradient(free_energy)

                    # Evolução temporal (equação de Allen-Cahn)
                    dphi_dt = self.params['mobility'] * chemical_force

                    # Integração temporal simples
                    self.microstructure += dphi_dt * 0.01

                    # Condições de contorno periódicas
                    self.microstructure = np.mod(self.microstructure + 0.5, 1) - 0.5

                    evolution_history.append(self.microstructure.copy())

                return evolution_history

            def _calculate_free_energy(self, phi):
                """Calcular energia livre do campo de fase"""
                # Energia interfacial
                grad_phi = np.gradient(phi)
                interface_energy = 0.5 * self.params['epsilon'] * np.sum(grad_phi**2)

                # Energia bulk
                bulk_energy = np.sum(phi**2 * (1 - phi)**2)

                total_energy = interface_energy + bulk_energy

                return total_energy

            def calculate_microstructural_parameters(self):
                """Calcular parâmetros microestruturais"""
                # Fração volumétrica de fases
                phase_fractions = self._calculate_phase_fractions()

                # Área interfacial específica
                interfacial_area = self._calculate_interfacial_area()

                # Tamanho médio de grão
                grain_size = self._calculate_grain_size()

                return {
                    'phase_fractions': phase_fractions,
                    'interfacial_area': interfacial_area,
                    'grain_size': grain_size
                }

            def _calculate_phase_fractions(self):
                """Calcular fração volumétrica de cada fase"""
                phi_threshold = 0.5

                phase1_fraction = np.mean(self.microstructure > phi_threshold)
                phase2_fraction = 1 - phase1_fraction

                return {
                    'phase_1': phase1_fraction,
                    'phase_2': phase2_fraction
                }

            def _calculate_interfacial_area(self):
                """Calcular área interfacial"""
                grad_phi = np.gradient(self.microstructure)
                interface_density = np.sqrt(np.sum(grad_phi**2, axis=0))

                interfacial_area = np.sum(interface_density)

                return interfacial_area

            def _calculate_grain_size(self):
                """Calcular tamanho médio de grão"""
                # Usar análise de watershed ou método simplificado
                # Aqui usamos um método baseado na autocorrelação

                # Função de autocorrelação
                autocorr = np.correlate(self.microstructure.flatten(),
                                      self.microstructure.flatten(), mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                # Encontrar primeira raiz da autocorrelação
                zero_crossing = np.where(np.diff(np.sign(autocorr)))[0]

                if len(zero_crossing) > 0:
                    grain_size = zero_crossing[0]
                else:
                    grain_size = len(self.microstructure) // 10  # Valor padrão

                return grain_size

        pf_model = PhaseFieldModel(initial_microstructure, self.phase_field_params)
        evolution = pf_model.evolve_microstructure(time_steps)
        parameters = pf_model.calculate_microstructural_parameters()

        return {
            'microstructure_evolution': evolution,
            'final_microstructure': evolution[-1],
            'microstructural_parameters': parameters
        }

    def grain_boundary_energy(self, grain_orientations, boundary_angles):
        """
        Cálculo de energia de contorno de grão
        """
        class GrainBoundaryModel:
            def __init__(self, orientations, angles):
                self.orientations = orientations
                self.angles = angles

            def calculate_read_shockley_energy(self, misorientation_angle):
                """Calcular energia usando modelo Read-Shockley"""
                # Energia de contorno de grão
                gamma_0 = 1.0  # Energia para ângulo alto
                theta_0 = np.pi / 6  # Ângulo característico

                if misorientation_angle < theta_0:
                    energy = gamma_0 * misorientation_angle * (1 - np.log(misorientation_angle / theta_0))
                else:
                    energy = gamma_0

                return energy

            def calculate_anisotropic_energy(self, orientation1, orientation2):
                """Calcular energia anisotrópica"""
                # Diferença de orientação
                delta_orientation = np.abs(orientation1 - orientation2)

                # Energia baseada na diferença
                energy = 1.0 + 0.5 * np.sin(2 * delta_orientation)

                return energy

            def simulate_grain_growth(self, initial_grains, n_steps=50):
                """Simular crescimento de grãos"""
                grains = initial_grains.copy()
                grain_areas = np.ones(len(grains))  # Áreas iniciais

                growth_history = [grains.copy()]

                for step in range(n_steps):
                    new_areas = grain_areas.copy()

                    for i in range(len(grains)):
                        # Taxa de crescimento proporcional à curvatura
                        curvature = self._calculate_grain_curvature(grains, i)

                        # Energia de contorno
                        boundary_energy = self.calculate_read_shockley_energy(
                            np.abs(grains[i] - np.mean(grains))
                        )

                        # Taxa de crescimento
                        growth_rate = curvature * boundary_energy
                        new_areas[i] += growth_rate

                    grain_areas = new_areas

                    # Remover grãos muito pequenos
                    keep_indices = grain_areas > 0.1
                    grains = grains[keep_indices]
                    grain_areas = grain_areas[keep_indices]

                    growth_history.append(grains.copy())

                return {
                    'grain_evolution': growth_history,
                    'final_grains': grains,
                    'final_areas': grain_areas
                }

            def _calculate_grain_curvature(self, grains, grain_idx):
                """Calcular curvatura de um grão"""
                # Curvatura baseada na diferença com grãos vizinhos
                differences = [np.abs(grains[grain_idx] - g) for g in grains if g != grains[grain_idx]]

                if differences:
                    curvature = np.mean(differences)
                else:
                    curvature = 0

                return curvature

        gb_model = GrainBoundaryModel(grain_orientations, boundary_angles)

        # Simular crescimento de grãos
        initial_grains = np.random.uniform(0, 2*np.pi, 20)  # Orientações aleatórias
        grain_growth = gb_model.simulate_grain_growth(initial_grains)

        return {
            'grain_boundary_model': gb_model,
            'grain_growth_simulation': grain_growth
        }

    def precipitate_microstructure(self, supersaturation_ratio, nucleation_sites):
        """
        Modelagem de microestrutura de precipitados
        """
        class PrecipitationModel:
            def __init__(self, supersaturation, sites):
                self.supersaturation = supersaturation
                self.nucleation_sites = sites
                self.precipitates = []

            def simulate_nucleation_growth(self, time_steps=100):
                """Simular nucleação e crescimento de precipitados"""
                concentration = self.supersaturation

                for step in range(time_steps):
                    # Nucleação
                    self._nucleation_step(concentration)

                    # Crescimento
                    self._growth_step(concentration)

                    # Atualização da concentração
                    concentration = self._update_concentration(concentration)

                return {
                    'precipitates': self.precipitates,
                    'final_concentration': concentration,
                    'precipitate_sizes': [p['size'] for p in self.precipitates],
                    'precipitate_positions': [p['position'] for p in self.precipitates]
                }

            def _nucleation_step(self, concentration):
                """Passo de nucleação"""
                # Taxa de nucleação baseada na supersaturação
                nucleation_rate = 1e-6 * (concentration - 1)**2

                n_new_nuclei = np.random.poisson(nucleation_rate * len(self.nucleation_sites))

                for _ in range(min(n_new_nuclei, len(self.nucleation_sites))):
                    site = np.random.choice(self.nucleation_sites)

                    new_precipitate = {
                        'position': site,
                        'size': 1.0,  # Tamanho inicial
                        'formation_time': len(self.precipitates)
                    }

                    self.precipitates.append(new_precipitate)

            def _growth_step(self, concentration):
                """Passo de crescimento"""
                for precipitate in self.precipitates:
                    # Taxa de crescimento proporcional à supersaturação
                    growth_rate = 0.01 * (concentration - 1)

                    precipitate['size'] += growth_rate

            def _update_concentration(self, concentration):
                """Atualizar concentração da matriz"""
                # Depleção devido ao crescimento dos precipitados
                total_precipitate_volume = sum(p['size']**3 for p in self.precipitates)

                depletion = total_precipitate_volume * 1e-6  # Fator de conversão

                new_concentration = concentration - depletion

                return max(new_concentration, 1.0)  # Não abaixo da solubilidade

            def calculate_aging_kinetics(self):
                """Calcular cinética de envelhecimento"""
                sizes = [p['size'] for p in self.precipitates]
                times = [p['formation_time'] for p in self.precipitates]

                # Distribuição de tamanhos
                size_bins = np.linspace(0, max(sizes) + 1, 20)
                size_distribution, _ = np.histogram(sizes, bins=size_bins)

                return {
                    'size_distribution': {
                        'bins': size_bins[:-1],
                        'counts': size_distribution
                    },
                    'average_size': np.mean(sizes),
                    'total_precipitate_volume': sum(s**3 for s in sizes)
                }

        pm = PrecipitationModel(supersaturation_ratio, nucleation_sites)
        simulation = pm.simulate_nucleation_growth()
        aging = pm.calculate_aging_kinetics()

        return {
            'precipitation_simulation': simulation,
            'aging_kinetics': aging
        }

    def fracture_mechanics_microstructure(self, crack_tip_stress, microstructural_features):
        """
        Mecânica da fratura em materiais microestruturados
        """
        class FractureMechanicsModel:
            def __init__(self, stress_intensity, features):
                self.stress_intensity = stress_intensity
                self.features = features

            def calculate_crack_propagation(self, initial_crack_length):
                """Calcular propagação de trinca"""
                # Fator de intensidade de tensão
                K_I = self.stress_intensity

                # Tenacidade à fratura
                K_IC = self.features.get('fracture_toughness', 50)  # MPa√m

                # Critério de propagação
                if K_I > K_IC:
                    # Velocidade de propagação (Lei de Paris simplificada)
                    da_dn = 1e-10 * (K_I / K_IC)**4  # m/ciclo

                    final_crack_length = initial_crack_length + da_dn * 1000
                else:
                    final_crack_length = initial_crack_length

                return {
                    'initial_crack_length': initial_crack_length,
                    'final_crack_length': final_crack_length,
                    'crack_growth': final_crack_length - initial_crack_length,
                    'failure_probability': min(1.0, K_I / K_IC)
                }

            def simulate_microstructurally_sensitive_cracking(self):
                """Simular trinca sensível à microestrutura"""
                # Obstáculos microestruturais
                grain_boundaries = self.features.get('grain_boundaries', [])
                precipitates = self.features.get('precipitates', [])

                # Caminho da trinca influenciado pela microestrutura
                crack_path = self._calculate_crack_path(grain_boundaries, precipitates)

                return {
                    'crack_path': crack_path,
                    'deflection_angle': self._calculate_deflection_angle(crack_path),
                    'toughening_mechanisms': self._identify_toughening_mechanisms()
                }

            def _calculate_crack_path(self, grain_boundaries, precipitates):
                """Calcular caminho da trinca"""
                # Caminho simplificado: evitar precipitados, seguir contornos de grão
                path_points = [(0, 0)]  # Ponto inicial

                for i in range(10):
                    current_pos = path_points[-1]

                    # Influência dos precipitados
                    precipitate_influence = self._precipitate_influence(current_pos, precipitates)

                    # Influência dos contornos de grão
                    boundary_influence = self._boundary_influence(current_pos, grain_boundaries)

                    # Novo ponto na direção resultante
                    direction = precipitate_influence + boundary_influence
                    direction = direction / (np.linalg.norm(direction) + 1e-6)

                    new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                    path_points.append(new_pos)

                return path_points

            def _precipitate_influence(self, position, precipitates):
                """Calcular influência dos precipitados"""
                influence = np.zeros(2)

                for precip in precipitates:
                    dist = np.linalg.norm(np.array(position) - np.array(precip))
                    if dist < 5:  # Raio de influência
                        direction_away = np.array(position) - np.array(precip)
                        influence += direction_away / (dist + 1e-6)

                return influence

            def _boundary_influence(self, position, boundaries):
                """Calcular influência dos contornos de grão"""
                influence = np.zeros(2)

                for boundary in boundaries:
                    dist = np.abs(position[1] - boundary)  # Distância para contorno horizontal
                    if dist < 2:
                        influence[1] += np.sign(position[1] - boundary) * (2 - dist)

                return influence

            def _calculate_deflection_angle(self, crack_path):
                """Calcular ângulo de deflexão da trinca"""
                if len(crack_path) < 3:
                    return 0

                # Vetores de direção
                v1 = np.array(crack_path[1]) - np.array(crack_path[0])
                v2 = np.array(crack_path[-1]) - np.array(crack_path[-2])

                # Ângulo entre vetores
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))

                return angle

            def _identify_toughening_mechanisms(self):
                """Identificar mecanismos de aumento de tenacidade"""
                mechanisms = []

                if self.features.get('grain_size', 10) < 5:
                    mechanisms.append('grain_refinement')

                if self.features.get('precipitate_volume_fraction', 0) > 0.1:
                    mechanisms.append('precipitate_toughening')

                if self.features.get('ductile_phase', False):
                    mechanisms.append('ductile_phase_toughening')

                return mechanisms

        fm_model = FractureMechanicsModel(crack_tip_stress, microstructural_features)
        crack_propagation = fm_model.calculate_crack_propagation(1e-3)
        microstructurally_sensitive = fm_model.simulate_microstructurally_sensitive_cracking()

        return {
            'crack_propagation': crack_propagation,
            'microstructural_cracking': microstructurally_sensitive
        }
```

**Modelos de Microestrutura:**
- Campo de fase para evolução microestrutural
- Energia de contorno de grão
- Microestrutura de precipitados
- Mecânica da fratura microestrutural

### 1.3 Propriedades Mecânicas e Térmicas
```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MechanicalThermalProperties:
    """
    Propriedades mecânicas e térmicas de materiais avançados
    """

    def __init__(self):
        self.material_constants = {
            'bulk_modulus_range': (50, 400),  # GPa
            'shear_modulus_range': (20, 200),  # GPa
            'thermal_conductivity_range': (0.1, 500)  # W/m/K
        }

    def elastic_tensor_calculation(self, crystal_structure, interatomic_potential):
        """
        Cálculo do tensor elástico
        """
        class ElasticTensorCalculator:
            def __init__(self, structure, potential):
                self.structure = structure
                self.potential = potential

            def calculate_elastic_constants(self):
                """Calcular constantes elásticas"""
                # Deformações para cálculo das constantes elásticas
                strains = self._generate_strain_tensor()

                stress_strain_data = []

                for strain in strains:
                    # Aplicar deformação
                    deformed_structure = self._apply_strain(self.structure, strain)

                    # Calcular energia
                    energy = self._calculate_energy(deformed_structure)

                    # Calcular tensor de tensão (derivada segunda da energia)
                    stress = self._calculate_stress_tensor(energy, strain)

                    stress_strain_data.append({
                        'strain': strain,
                        'stress': stress,
                        'energy': energy
                    })

                # Ajustar constantes elásticas
                elastic_constants = self._fit_elastic_constants(stress_strain_data)

                return {
                    'elastic_constants': elastic_constants,
                    'stress_strain_data': stress_strain_data,
                    'bulk_modulus': self._calculate_bulk_modulus(elastic_constants),
                    'shear_modulus': self._calculate_shear_modulus(elastic_constants)
                }

            def _generate_strain_tensor(self):
                """Gerar tensores de deformação para teste"""
                strains = []

                # Deformações volumétricas
                for delta in [-0.01, 0.01]:
                    strain = np.eye(3) * delta
                    strains.append(strain)

                # Deformações de cisalhamento
                for i in range(3):
                    for j in range(i+1, 3):
                        strain = np.zeros((3, 3))
                        strain[i, j] = strain[j, i] = 0.01
                        strains.append(strain)

                return strains

            def _apply_strain(self, structure, strain_tensor):
                """Aplicar deformação à estrutura"""
                deformation = np.eye(3) + strain_tensor

                deformed_positions = []
                for pos in structure:
                    new_pos = np.dot(deformation, pos)
                    deformed_positions.append(new_pos)

                return deformed_positions

            def _calculate_energy(self, structure):
                """Calcular energia da estrutura deformada"""
                # Usar potencial de Lennard-Jones simplificado
                energy = 0
                n_atoms = len(structure)

                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        r_ij = np.array(structure[j]) - np.array(structure[i])
                        r = np.linalg.norm(r_ij)

                        if r > 0.1:
                            sr6 = (1.0 / r)**6
                            energy += 4.0 * (sr6**2 - sr6)

                return energy

            def _calculate_stress_tensor(self, energy, strain):
                """Calcular tensor de tensão"""
                # Derivada da energia em relação à deformação
                # Aproximação simples
                stress = 2 * energy * strain  # Relação aproximada

                return stress

            def _fit_elastic_constants(self, stress_strain_data):
                """Ajustar constantes elásticas aos dados"""
                # Modelo linear: σ = C * ε
                strains = [data['strain'].flatten() for data in stress_strain_data]
                stresses = [data['stress'].flatten() for data in stress_strain_data]

                # Ajuste linear
                from scipy.linalg import lstsq

                strain_matrix = np.array(strains)
                stress_vector = np.array(stresses).flatten()

                # Resolver sistema linear
                elastic_constants, residuals, rank, s = lstsq(strain_matrix, stress_vector)

                return elastic_constants.reshape((3, 3))

            def _calculate_bulk_modulus(self, elastic_constants):
                """Calcular módulo de compressibilidade"""
                # K = (C11 + C22 + C33 + 2(C12 + C13 + C23)) / 9
                c11 = elastic_constants[0, 0]
                c22 = elastic_constants[1, 1]
                c33 = elastic_constants[2, 2]
                c12 = elastic_constants[0, 1]
                c13 = elastic_constants[0, 2]
                c23 = elastic_constants[1, 2]

                bulk_modulus = (c11 + c22 + c33 + 2*(c12 + c13 + c23)) / 9

                return bulk_modulus

            def _calculate_shear_modulus(self, elastic_constants):
                """Calcular módulo de cisalhamento"""
                # G = (C11 + C22 + C33 - (C12 + C13 + C23) + 3(C44 + C55 + C66)) / 15
                c11 = elastic_constants[0, 0]
                c22 = elastic_constants[1, 1]
                c33 = elastic_constants[2, 2]
                c12 = elastic_constants[0, 1]
                c13 = elastic_constants[0, 2]
                c23 = elastic_constants[1, 2]
                c44 = elastic_constants[0, 1]  # Aproximação
                c55 = elastic_constants[1, 2]  # Aproximação
                c66 = elastic_constants[0, 2]  # Aproximação

                shear_modulus = (c11 + c22 + c33 - (c12 + c13 + c23) + 3*(c44 + c55 + c66)) / 15

                return shear_modulus

        etc = ElasticTensorCalculator(crystal_structure, interatomic_potential)
        elastic_properties = etc.calculate_elastic_constants()

        return elastic_properties

    def thermal_conductivity_calculation(self, phonon_spectra, temperature):
        """
        Cálculo de condutividade térmica
        """
        class ThermalConductivityCalculator:
            def __init__(self, phonon_freq, temp):
                self.phonon_frequencies = phonon_freq
                self.temperature = temp

            def calculate_thermal_conductivity(self):
                """Calcular condutividade térmica usando Boltzmann transport"""
                # Constante de Boltzmann
                kB = 1.38e-23

                # Frequências dos fonons
                omega = 2 * np.pi * self.phonon_frequencies

                # Função de Bose
                n_bose = 1 / (np.exp(hbar * omega / (kB * self.temperature)) - 1)

                # Capacidade calorífica por modo
                cv_modes = kB * (hbar * omega / (kB * self.temperature))**2 * \
                          np.exp(hbar * omega / (kB * self.temperature)) / \
                          (np.exp(hbar * omega / (kB * self.temperature)) - 1)**2

                # Tempo de relaxação (aproximação)
                tau = 1e-12  # s (tempo de relaxação típico)

                # Velocidade do som (aproximação)
                v_sound = 5000  # m/s

                # Condutividade térmica
                kappa = (1/3) * np.sum(cv_modes * v_sound**2 * tau)

                return {
                    'thermal_conductivity': kappa,
                    'phonon_contribution': cv_modes,
                    'relaxation_time': tau,
                    'sound_velocity': v_sound
                }

            def calculate_thermal_boundary_resistance(self, interface_properties):
                """Calcular resistência térmica de interface (Kapitza)"""
                # Condutância interfacial
                G = interface_properties.get('interface_conductance', 1e8)  # W/m²/K

                # Espessura efetiva
                d = 1e-9  # m (espessura da interface)

                # Resistência térmica
                R_kapitza = d / G

                return {
                    'kapitza_resistance': R_kapitza,
                    'interface_conductance': G,
                    'effective_thickness': d
                }

        tcc = ThermalConductivityCalculator(phonon_spectra, temperature)
        thermal_conductivity = tcc.calculate_thermal_conductivity()

        return thermal_conductivity

    def fracture_toughness_prediction(self, microstructural_parameters):
        """
        Previsão de tenacidade à fratura
        """
        class FractureToughnessPredictor:
            def __init__(self, micro_params):
                self.micro_params = micro_params

            def predict_toughness(self):
                """Prever tenacidade à fratura"""
                # Modelo baseado na microestrutura
                grain_size = self.micro_params.get('grain_size', 10e-6)  # m
                precipitate_size = self.micro_params.get('precipitate_size', 1e-6)  # m
                porosity = self.micro_params.get('porosity', 0.01)

                # Tenacidade intrínseca
                K0 = 20  # MPa√m (valor base)

                # Efeito do tamanho de grão (Hall-Petch)
                grain_contribution = 10 * grain_size**(-0.5)

                # Efeito dos precipitados
                precipitate_contribution = 5 * precipitate_size**0.5

                # Efeito da porosidade
                porosity_effect = 1 - 2 * porosity

                # Tenacidade total
                K_IC = K0 + grain_contribution + precipitate_contribution
                K_IC *= porosity_effect

                return {
                    'fracture_toughness': K_IC,
                    'grain_size_effect': grain_contribution,
                    'precipitate_effect': precipitate_contribution,
                    'porosity_effect': porosity_effect
                }

            def calculate_crack_growth_resistance(self):
                """Calcular resistência ao crescimento de trinca"""
                # Curva R
                crack_lengths = np.linspace(1e-6, 1e-3, 100)  # m

                # Resistência crescente com o tamanho da trinca
                R_curve = self.predict_toughness()['fracture_toughness'] * \
                         (1 + 0.1 * np.log(crack_lengths / 1e-6))

                return {
                    'crack_lengths': crack_lengths,
                    'resistance_curve': R_curve,
                    'stable_crack_growth': np.where(R_curve > self.predict_toughness()['fracture_toughness'])[0]
                }

        ftp = FractureToughnessPredictor(microstructural_parameters)
        toughness_prediction = ftp.predict_toughness()
        crack_growth = ftp.calculate_crack_growth_resistance()

        return {
            'toughness_prediction': toughness_prediction,
            'crack_growth_resistance': crack_growth
        }

    def fatigue_life_prediction(self, stress_amplitude, microstructural_factors):
        """
        Previsão de vida em fadiga
        """
        class FatigueLifePredictor:
            def __init__(self, stress_amp, micro_factors):
                self.stress_amplitude = stress_amp
                self.micro_factors = micro_factors

            def predict_fatigue_life(self):
                """Prever vida em fadiga usando Lei de Paris"""
                # Lei de Paris: da/dN = C * (ΔK)^m
                C = 1e-10  # Coeficiente
                m = 3.0    # Expoente

                # Fator de intensidade de tensão
                delta_K = self.stress_amplitude * np.sqrt(np.pi * 1e-6)  # MPa√m

                # Número de ciclos para propagação
                da_dN = C * (delta_K)**m

                # Comprimento inicial da trinca
                a_initial = 1e-6  # m

                # Comprimento crítico
                a_critical = (self.micro_factors.get('fracture_toughness', 50) / self.stress_amplitude)**2 / np.pi

                # Número total de ciclos
                N_f = (a_critical - a_initial) / da_dN

                return {
                    'fatigue_life': N_f,
                    'crack_growth_rate': da_dN,
                    'stress_intensity_factor': delta_K,
                    'critical_crack_length': a_critical
                }

            def calculate_s_n_curve(self, stress_levels):
                """Calcular curva S-N"""
                fatigue_lives = []

                for stress in stress_levels:
                    life = self.predict_fatigue_life()
                    # Ajustar para nível de tensão
                    adjusted_life = life * (self.stress_amplitude / stress)**3
                    fatigue_lives.append(adjusted_life)

                return {
                    'stress_levels': stress_levels,
                    'fatigue_lives': fatigue_lives,
                    'endurance_limit': min(stress_levels) if min(fatigue_lives) > 1e6 else None
                }

        flp = FatigueLifePredictor(stress_amplitude, microstructural_factors)
        fatigue_life = flp.predict_fatigue_life()

        return {
            'fatigue_life_prediction': fatigue_life,
            's_n_curve': flp.calculate_s_n_curve(np.linspace(0.1, 0.8, 10) * stress_amplitude)
        }
```

**Propriedades Mecânicas e Térmicas:**
- Cálculo do tensor elástico
- Condutividade térmica via transporte fonônico
- Previsão de tenacidade à fratura
- Previsão de vida em fadiga

---

## 2. MATERIAIS FUNCIONAIS E COMPÓSITOS

### 2.1 Materiais Inteligentes e Piezoelétricos
```python
import numpy as np
from scipy.constants import epsilon_0, e
import matplotlib.pyplot as plt

class SmartMaterials:
    """
    Materiais inteligentes e piezoelétricos
    """

    def __init__(self):
        self.piezoelectric_constants = {
            'quartz': {'d15': 2.3e-12, 'd22': -0.67e-12},  # C/N
            'PZT': {'d33': 400e-12, 'd31': -200e-12},     # C/N
            'PVDF': {'d31': 20e-12, 'd32': 3e-12}         # C/N
        }

    def piezoelectric_tensor_calculation(self, crystal_structure, material_type):
        """
        Cálculo do tensor piezoelétrico
        """
        class PiezoelectricCalculator:
            def __init__(self, structure, material):
                self.structure = structure
                self.material = material
                self.piezo_tensor = self._calculate_piezo_tensor()

            def _calculate_piezo_tensor(self):
                """Calcular tensor piezoelétrico"""
                # Constantes do material
                constants = self.piezoelectric_constants.get(self.material, {})

                # Tensor 3x6 para materiais 6mm (quartz-like)
                piezo_tensor = np.zeros((3, 6))

                # Preencher baseado no tipo de material
                if self.material == 'quartz':
                    piezo_tensor[0, 3] = constants.get('d15', 0)  # d15
                    piezo_tensor[1, 1] = constants.get('d22', 0)  # d22
                    piezo_tensor[1, 0] = -piezo_tensor[1, 1] / 2  # d21
                    piezo_tensor[2, 2] = piezo_tensor[1, 1]       # d22
                    piezo_tensor[2, 0] = piezo_tensor[1, 0]       # d21

                elif self.material == 'PZT':
                    piezo_tensor[2, 2] = constants.get('d33', 0)  # d33
                    piezo_tensor[0, 2] = constants.get('d31', 0)  # d31
                    piezo_tensor[1, 2] = piezo_tensor[0, 2]       # d32 = d31

                return piezo_tensor

            def calculate_polarization_charge(self, strain_tensor):
                """Calcular carga de polarização devido à deformação"""
                # P = d * σ (aproximação)
                # Converter tensor de tensão 6x1 para matriz 3x3
                strain_voigt = self._tensor_to_voigt(strain_tensor)

                # Calcular polarização
                polarization = np.dot(self.piezo_tensor, strain_voigt)

                return polarization

            def calculate_induced_strain(self, electric_field):
                """Calcular deformação induzida por campo elétrico"""
                # σ = d^T * E
                strain_voigt = np.dot(self.piezo_tensor.T, electric_field)

                # Converter para tensor 3x3
                strain_tensor = self._voigt_to_tensor(strain_voigt)

                return strain_tensor

            def _tensor_to_voigt(self, tensor):
                """Converter tensor 3x3 para notação Voigt 6x1"""
                voigt = np.array([
                    tensor[0, 0],
                    tensor[1, 1],
                    tensor[2, 2],
                    2 * tensor[1, 2],
                    2 * tensor[0, 2],
                    2 * tensor[0, 1]
                ])

                return voigt

            def _voigt_to_tensor(self, voigt):
                """Converter notação Voigt 6x1 para tensor 3x3"""
                tensor = np.array([
                    [voigt[0], voigt[5], voigt[4]],
                    [voigt[5], voigt[1], voigt[3]],
                    [voigt[4], voigt[3], voigt[2]]
                ])

                return tensor

            def calculate_electromechanical_coupling(self):
                """Calcular acoplamento eletromecânico"""
                # Fator de acoplamento k = d * sqrt(c^E / epsilon^T)
                # Aproximação simplificada

                piezoelectric_coeff = np.linalg.norm(self.piezo_tensor)
                elastic_stiffness = 1e11  # Pa (aproximado)
                dielectric_constant = 1000 * epsilon_0

                k_factor = piezoelectric_coeff * np.sqrt(elastic_stiffness / dielectric_constant)

                return {
                    'electromechanical_coupling_factor': k_factor,
                    'piezoelectric_coefficient': piezoelectric_coeff,
                    'effective_coupling_efficiency': k_factor**2
                }

        pz_calc = PiezoelectricCalculator(crystal_structure, material_type)
        coupling = pz_calc.calculate_electromechanical_coupling()

        return {
            'piezoelectric_calculator': pz_calc,
            'piezoelectric_tensor': pz_calc.piezo_tensor,
            'electromechanical_coupling': coupling
        }

    def shape_memory_alloys(self, alloy_composition, transformation_temperatures):
        """
        Ligas com memória de forma
        """
        class ShapeMemoryAlloy:
            def __init__(self, composition, temperatures):
                self.composition = composition
                self.temperatures = temperatures

            def martensitic_transformation(self, temperature, stress=0):
                """Simular transformação martensítica"""
                # Temperaturas de transformação
                Ms = self.temperatures.get('Ms', 300)  # Início martensita
                Mf = self.temperatures.get('Mf', 250)  # Fim martensita
                As = self.temperatures.get('As', 350)  # Início austenita
                Af = self.temperatures.get('Af', 400)  # Fim austenita

                # Fração martensítica
                if temperature <= Mf:
                    martensite_fraction = 1.0
                elif temperature >= Af:
                    martensite_fraction = 0.0
                elif Mf < temperature < Ms:
                    martensite_fraction = (Ms - temperature) / (Ms - Mf)
                elif As < temperature < Af:
                    martensite_fraction = (Af - temperature) / (Af - As)
                else:
                    martensite_fraction = 0.5

                # Efeito do estresse (pseudo-elastico)
                if stress > 0:
                    stress_effect = min(stress / 100e6, 0.3)  # MPa
                    martensite_fraction += stress_effect

                return {
                    'martensite_fraction': martensite_fraction,
                    'austenite_fraction': 1 - martensite_fraction,
                    'transformation_temperatures': self.temperatures,
                    'current_phase': 'martensite' if martensite_fraction > 0.5 else 'austenite'
                }

            def shape_memory_effect(self, deformation_history):
                """Simular efeito de memória de forma"""
                # Deformação residual
                residual_strain = 0

                # Aplicar deformação
                for deformation in deformation_history:
                    if deformation['type'] == 'loading':
                        residual_strain += deformation['strain']
                    elif deformation['type'] == 'heating':
                        # Recuperação de forma
                        recovery = min(residual_strain, deformation['temperature'] / 1000)
                        residual_strain -= recovery

                return {
                    'residual_strain': residual_strain,
                    'shape_recovery_ratio': (1 - residual_strain / max([d.get('strain', 0) for d in deformation_history] + [1e-6])),
                    'deformation_history': deformation_history
                }

            def superelastic_behavior(self, stress_history):
                """Simular comportamento superelástico"""
                strains = []
                phases = []

                for stress in stress_history:
                    # Temperatura efetiva baseada no estresse
                    effective_temp = 300 - stress / 1e7  # K

                    transformation = self.martensitic_transformation(effective_temp, stress)

                    # Deformação baseada na fração martensítica
                    strain = transformation['martensite_fraction'] * 0.06  # 6% deformação

                    strains.append(strain)
                    phases.append(transformation['current_phase'])

                return {
                    'stress_strain_curve': list(zip(stress_history, strains)),
                    'phase_history': phases,
                    'hysteresis_area': self._calculate_hysteresis_area(stress_history, strains)
                }

            def _calculate_hysteresis_area(self, stresses, strains):
                """Calcular área de histerese"""
                # Área do ciclo de histerese
                hysteresis_area = 0

                for i in range(1, len(stresses)):
                    delta_stress = stresses[i] - stresses[i-1]
                    avg_strain = (strains[i] + strains[i-1]) / 2
                    hysteresis_area += delta_stress * avg_strain

                return abs(hysteresis_area)

        sma = ShapeMemoryAlloy(alloy_composition, transformation_temperatures)
        transformation = sma.martensitic_transformation(320)

        return {
            'shape_memory_alloy': sma,
            'current_transformation': transformation,
            'superelastic_response': sma.superelastic_behavior(np.linspace(0, 800e6, 50))
        }

    def magnetostrictive_materials(self, material_composition, magnetic_field):
        """
        Materiais magnetorresistivos
        """
        class MagnetostrictiveMaterial:
            def __init__(self, composition, field):
                self.composition = composition
                self.magnetic_field = field

            def calculate_magnetostriction(self):
                """Calcular magnetostrição"""
                # Coeficiente de magnetostrição
                if 'Terfenol-D' in self.composition:
                    lambda_s = 2000e-6  # Magnetostrição de saturação
                    M_s = 800e3  # Magnetização de saturação (A/m)
                else:
                    lambda_s = 500e-6
                    M_s = 400e3

                # Campo magnético aplicado
                H = self.magnetic_field

                # Magnetização (Lei de Langevin aproximada)
                magnetization = M_s * np.tanh(H / 1000)  # A/m

                # Magnetostrição
                magnetostriction = lambda_s * (magnetization / M_s)**2

                return {
                    'magnetostriction': magnetostriction,
                    'magnetization': magnetization,
                    'saturation_magnetostriction': lambda_s,
                    'field_sensitivity': lambda_s * 2 * magnetization / (M_s**2 * 1000)
                }

            def villari_effect_simulation(self):
                """Simular efeito Villari (magnetostrição inversa)"""
                # Mudança na magnetização devido à tensão mecânica
                stress = np.linspace(0, 100e6, 50)  # Pa

                delta_magnetization = []

                for sigma in stress:
                    # Efeito piezomagnético
                    delta_m = -2e-9 * sigma  # A/m/Pa (coeficiente aproximado)
                    delta_magnetization.append(delta_m)

                return {
                    'applied_stress': stress,
                    'magnetization_change': delta_magnetization,
                    'piezomagnetic_coefficient': -2e-9
                }

            def magnetoelastic_coupling(self):
                """Calcular acoplamento magnetoelástico"""
                # Energia magnetoelástica
                B1 = 1e7  # Pa (constante magnetoelástica)

                # Energia de anisotropia
                K1 = 1e5  # J/m³

                # Acoplamento
                coupling_energy = B1 * (self.calculate_magnetostriction()['magnetostriction'] * 1e6)

                return {
                    'magnetoelastic_energy': coupling_energy,
                    'anisotropy_energy': K1,
                    'coupling_efficiency': coupling_energy / K1
                }

        ms_material = MagnetostrictiveMaterial(material_composition, magnetic_field)
        magnetostriction = ms_material.calculate_magnetostriction()

        return {
            'magnetostrictive_material': ms_material,
            'magnetostriction_response': magnetostriction,
            'villari_effect': ms_material.villari_effect_simulation(),
            'magnetoelastic_coupling': ms_material.magnetoelastic_coupling()
        }

    def electroactive_polymers(self, polymer_type, electric_field_strength):
        """
        Polímeros eletroativos
        """
        class ElectroactivePolymer:
            def __init__(self, polymer, field):
                self.polymer = polymer
                self.electric_field = field

            def maxwell_stress_actuation(self):
                """Simular atuação por estresse de Maxwell"""
                # Constante dielétrica
                if self.polymer == 'silicone':
                    epsilon_r = 3.0
                    young_modulus = 1e6  # Pa
                elif self.polymer == 'acrylic':
                    epsilon_r = 4.7
                    young_modulus = 0.5e6  # Pa
                else:
                    epsilon_r = 3.0
                    young_modulus = 1e6  # Pa

                # Estresse de Maxwell
                maxwell_stress = epsilon_0 * epsilon_r * self.electric_field**2

                # Deformação
                strain = maxwell_stress / young_modulus

                return {
                    'maxwell_stress': maxwell_stress,
                    'resulting_strain': strain,
                    'relative_permittivity': epsilon_r,
                    'electric_field': self.electric_field
                }

            def electrostriction_effect(self):
                """
                Simular efeito de eletrostrição
                """
                # Coeficiente de eletrostrição
                M = 1e-18  # m²/V²

                # Deformação de eletrostrição
                electrostriction_strain = M * self.electric_field**2

                return {
                    'electrostriction_coefficient': M,
                    'electrostriction_strain': electrostriction_strain,
                    'field_induced_strain': electrostriction_strain
                }

            def calculate_actuation_efficiency(self):
                """Calcular eficiência de atuação"""
                maxwell = self.maxwell_stress_actuation()
                electrostriction = self.electstriction_effect()

                total_strain = maxwell['resulting_strain'] + electrostriction['electrostriction_strain']

                # Eficiência (comparado com deformação máxima possível)
                max_possible_strain = 0.1  # 10%
                efficiency = total_strain / max_possible_strain

                return {
                    'total_actuation_strain': total_strain,
                    'actuation_efficiency': efficiency,
                    'maxwell_contribution': maxwell['resulting_strain'] / total_strain,
                    'electrostriction_contribution': electrostriction['electrostriction_strain'] / total_strain
                }

        eap = ElectroactivePolymer(polymer_type, electric_field_strength)
        actuation = eap.maxwell_stress_actuation()

        return {
            'electroactive_polymer': eap,
            'actuation_response': actuation,
            'efficiency_analysis': eap.calculate_actuation_efficiency()
        }
```

**Materiais Inteligentes:**
- Cálculo do tensor piezoelétrico
- Ligas com memória de forma
- Materiais magnetorresistivos
- Polímeros eletroativos

### 2.2 Compósitos Multifuncionais
```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CompositeMaterials:
    """
    Materiais compósitos multifuncionais
    """

    def __init__(self):
        self.fiber_properties = {
            'carbon': {'E': 230e9, 'rho': 1800, 'cost': 20},  # Pa, kg/m³, $/kg
            'glass': {'E': 70e9, 'rho': 2540, 'cost': 2},
            'aramid': {'E': 130e9, 'rho': 1440, 'cost': 15}
        }

    def composite_lamina_properties(self, fiber_type, matrix_type, fiber_volume_fraction):
        """
        Propriedades de lâmina compósita
        """
        class CompositeLamina:
            def __init__(self, fiber, matrix, vf):
                self.fiber = fiber
                self.matrix = matrix
                self.vf = vf
                self.vm = 1 - vf

            def calculate_stiffness_matrix(self):
                """Calcular matriz de rigidez (Q)"""
                # Propriedades das fases
                E_f = self.fiber_properties[self.fiber]['E']
                E_m = 3e9  # Pa (matriz epóxi típica)
                nu_f = 0.3
                nu_m = 0.35
                G_f = E_f / (2 * (1 + nu_f))
                G_m = E_m / (2 * (1 + nu_m))

                # Coeficientes de Halpin-Tsai
                eta_L = (E_f/E_m - 1) / (E_f/E_m + 2*self.vf)
                eta_T = (E_f/E_m - 1) / (E_f/E_m + 2)

                # Módulos longitudinal e transversal
                E_L = E_m * (1 + 2*self.vf*eta_L) / (1 - self.vf*eta_L)
                E_T = E_m * (1 + 2*self.vf*eta_T) / (1 - self.vf*eta_T)
                G_LT = G_m * (1 + self.vf*eta_T) / (1 - self.vf*eta_T)

                nu_LT = self.vf*nu_f + self.vm*nu_m
                nu_TT = nu_LT

                # Matriz de rigidez reduzida
                Q = np.array([
                    [E_L/(1-nu_LT*nu_TT), nu_LT*E_T/(1-nu_LT*nu_TT), 0],
                    [nu_LT*E_T/(1-nu_LT*nu_TT), E_T/(1-nu_LT*nu_TT), 0],
                    [0, 0, G_LT]
                ])

                return {
                    'stiffness_matrix': Q,
                    'longitudinal_modulus': E_L,
                    'transverse_modulus': E_T,
                    'shear_modulus': G_LT,
                    'poisson_ratios': {'nu_LT': nu_LT, 'nu_TT': nu_TT}
                }

            def calculate_strength_properties(self):
                """Calcular propriedades de resistência"""
                # Resistência à tração longitudinal
                sigma_Lt = self.vf * 3500e6 + self.vm * 50e6  # Pa (aproximado)

                # Resistência à tração transversal
                sigma_Tt = 50e6  # Pa (controlada pela matriz)

                # Resistência ao cisalhamento
                tau_LT = 70e6  # Pa

                return {
                    'longitudinal_tensile_strength': sigma_Lt,
                    'transverse_tensile_strength': sigma_Tt,
                    'shear_strength': tau_LT,
                    'failure_criteria': self._calculate_failure_envelope()
                }

            def _calculate_failure_envelope(self):
                """Calcular envelope de falha"""
                # Critério de Tsai-Hill simplificado
                sigma_L, sigma_T = np.meshgrid(np.linspace(0, 400e6, 50), np.linspace(0, 100e6, 50))

                # Critério de falha
                strength_props = self.calculate_strength_properties()
                sigma_L0 = strength_props['longitudinal_tensile_strength']
                sigma_T0 = strength_props['transverse_tensile_strength']
                tau_0 = strength_props['shear_strength']

                failure_index = (sigma_L/sigma_L0)**2 + (sigma_T/sigma_T0)**2 + (0/tau_0)**2 - 1

                return {
                    'sigma_L_range': sigma_L,
                    'sigma_T_range': sigma_T,
                    'failure_index': failure_index
                }

            def optimize_fiber_volume_fraction(self):
                """Otimizar fração volumétrica de fibra"""
                vf_range = np.linspace(0.1, 0.7, 20)
                properties = []

                for vf in vf_range:
                    self.vf = vf
                    self.vm = 1 - vf

                    stiffness = self.calculate_stiffness_matrix()
                    strength = self.calculate_strength_properties()

                    # Índice de performance
                    performance = stiffness['longitudinal_modulus'] * strength['longitudinal_tensile_strength']

                    properties.append({
                        'vf': vf,
                        'modulus': stiffness['longitudinal_modulus'],
                        'strength': strength['longitudinal_tensile_strength'],
                        'performance_index': performance
                    })

                # Encontrar ótimo
                best = max(properties, key=lambda x: x['performance_index'])

                return {
                    'optimization_results': properties,
                    'optimal_vf': best['vf'],
                    'maximum_performance': best['performance_index']
                }

        lamina = CompositeLamina(fiber_type, matrix_type, fiber_volume_fraction)
        stiffness = lamina.calculate_stiffness_matrix()
        strength = lamina.calculate_strength_properties()
        optimization = lamina.optimize_fiber_volume_fraction()

        return {
            'lamina_properties': lamina,
            'stiffness_properties': stiffness,
            'strength_properties': strength,
            'optimization': optimization
        }

    def laminate_theory_analysis(self, lamina_properties, stacking_sequence):
        """
        Análise de laminado usando teoria das lâminas
        """
        class LaminateAnalysis:
            def __init__(self, lamina_props, stacking):
                self.lamina = lamina_props
                self.stacking = stacking
                self.thickness = 0.125e-3  # Espessura de cada lâmina (mm)

            def calculate_laminate_stiffness(self):
                """Calcular rigidez do laminado [A], [B], [D]"""
                n_layers = len(self.stacking)
                total_thickness = n_layers * self.thickness

                # Matrizes de rigidez do laminado
                A = np.zeros((3, 3))
                B = np.zeros((3, 3))
                D = np.zeros((3, 3))

                z_positions = self._calculate_z_positions(total_thickness)

                for i, angle in enumerate(self.stacking):
                    # Matriz de rigidez transformada
                    Q_bar = self._transform_stiffness_matrix(float(angle))

                    # Posições z
                    z_k = z_positions[i]
                    z_k1 = z_positions[i+1]

                    # Matriz A
                    A += Q_bar * (z_k1 - z_k)

                    # Matriz B
                    B += Q_bar * (z_k1**2 - z_k**2) / 2

                    # Matriz D
                    D += Q_bar * (z_k1**3 - z_k**3) / 3

                return {
                    'A_matrix': A,
                    'B_matrix': B,
                    'D_matrix': D,
                    'total_thickness': total_thickness
                }

            def _calculate_z_positions(self, total_thickness):
                """Calcular posições z das interfaces"""
                z = [-total_thickness/2]

                for i in range(len(self.stacking)):
                    z_k = z[-1] + self.thickness
                    z.append(z_k)

                return z

            def _transform_stiffness_matrix(self, angle):
                """Transformar matriz de rigidez para ângulo θ"""
                theta = np.radians(angle)

                # Matriz de rotação
                c = np.cos(theta)
                s = np.sin(theta)

                T = np.array([
                    [c**2, s**2, 2*c*s],
                    [s**2, c**2, -2*c*s],
                    [-c*s, c*s, c**2 - s**2]
                ])

                # Matriz de rigidez da lâmina (Q)
                Q = self.lamina['stiffness_matrix']

                # Matriz transformada
                Q_bar = np.linalg.inv(T) @ Q @ np.linalg.inv(T.T)

                return Q_bar

            def calculate_laminate_strength(self, applied_loads):
                """Calcular resistência do laminado"""
                # Calcular tensões em cada lâmina
                lamina_stresses = self._calculate_lamina_stresses(applied_loads)

                # Verificar critérios de falha
                failure_analysis = []

                for i, stresses in enumerate(lamina_stresses):
                    failure_index = self._tsai_wu_criterion(stresses, i)
                    failure_analysis.append({
                        'layer': i,
                        'stresses': stresses,
                        'failure_index': failure_index,
                        'failed': failure_index > 1.0
                    })

                return {
                    'lamina_stresses': lamina_stresses,
                    'failure_analysis': failure_analysis,
                    'critical_layer': min(failure_analysis, key=lambda x: x['failure_index'])
                }

            def _calculate_lamina_stresses(self, applied_loads):
                """Calcular tensões em cada lâmina"""
                # Cargas aplicadas [Nx, Ny, Nxy, Mx, My, Mxy]
                N = applied_loads[:3]
                M = applied_loads[3:]

                laminate_matrices = self.calculate_laminate_stiffness()
                A = laminate_matrices['A_matrix']
                B = laminate_matrices['B_matrix']

                # Deformações e curvaturas
                epsilon_kappa = np.linalg.inv(A) @ N - np.linalg.inv(A) @ B @ np.linalg.inv(A) @ M

                epsilon = epsilon_kappa[:3]  # Deformações
                kappa = epsilon_kappa[3:]   # Curvaturas

                # Tensões em cada lâmina
                lamina_stresses = []

                for angle in self.stacking:
                    # Matriz de transformação
                    theta = np.radians(angle)
                    T_sigma = self._stress_transformation_matrix(theta)

                    # Tensões na lâmina
                    sigma_lamina = T_sigma @ self.lamina['stiffness_matrix'] @ epsilon
                    lamina_stresses.append(sigma_lamina)

                return lamina_stresses

            def _stress_transformation_matrix(self, theta):
                """Matriz de transformação de tensões"""
                c = np.cos(theta)
                s = np.sin(theta)

                T = np.array([
                    [c**2, s**2, 2*c*s],
                    [s**2, c**2, -2*c*s],
                    [-c*s, c*s, c**2 - s**2]
                ])

                return T

            def _tsai_wu_criterion(self, stresses, layer_idx):
                """Critério de falha de Tsai-Wu"""
                sigma_L, sigma_T, tau_LT = stresses

                # Coeficientes (simplificados)
                F1 = 1/3500e6  # 1/σ_Lt
                F2 = 1/50e6    # 1/σ_Tt
                F11 = -1/(3500e6)**2
                F22 = -1/(50e6)**2
                F66 = 1/(70e6)**2
                F12 = -0.5 * np.sqrt(F11 * F22)

                # Polinômio de falha
                failure_index = F1*sigma_L + F2*sigma_T + F11*sigma_L**2 + \
                              F22*sigma_T**2 + F66*tau_LT**2 + 2*F12*sigma_L*sigma_T

                return failure_index

        laminate = LaminateAnalysis(lamina_properties, stacking_sequence)
        stiffness = laminate.calculate_laminate_stiffness()

        return {
            'laminate_analysis': laminate,
            'laminate_stiffness': stiffness
        }

    def functionally_graded_materials(self, composition_profile, property_gradients):
        """
        Materiais com gradiente funcional
        """
        class FunctionallyGradedMaterial:
            def __init__(self, composition, gradients):
                self.composition = composition
                self.gradients = gradients

            def calculate_effective_properties(self, position):
                """Calcular propriedades efetivas em uma posição"""
                # Gradiente de composição
                if self.composition['type'] == 'exponential':
                    phi = self.composition['phi_0'] * np.exp(self.composition['k'] * position)
                elif self.composition['type'] == 'power_law':
                    phi = self.composition['phi_0'] * (position / self.composition['L'])**self.composition['n']
                else:
                    phi = 0.5  # Uniforme

                # Propriedades efetivas
                E1 = self.gradients['E1']  # Material 1
                E2 = self.gradients['E2']  # Material 2

                E_effective = phi * E1 + (1 - phi) * E2

                # Condutividade térmica
                k1 = self.gradients.get('k1', 10)
                k2 = self.gradients.get('k2', 100)

                k_effective = phi * k1 + (1 - phi) * k2

                return {
                    'position': position,
                    'composition_fraction': phi,
                    'effective_modulus': E_effective,
                    'effective_conductivity': k_effective
                }

            def optimize_gradation_profile(self):
                """Otimizar perfil de graduação"""
                positions = np.linspace(0, 1, 20)
                profiles = ['exponential', 'power_law', 'linear']

                optimal_profile = None
                max_performance = 0

                for profile_type in profiles:
                    self.composition['type'] = profile_type

                    if profile_type == 'exponential':
                        self.composition.update({'phi_0': 0.1, 'k': 2})
                    elif profile_type == 'power_law':
                        self.composition.update({'phi_0': 0.1, 'n': 2, 'L': 1})
                    elif profile_type == 'linear':
                        self.composition.update({'type': 'linear'})

                    # Calcular performance
                    properties_profile = [self.calculate_effective_properties(pos) for pos in positions]
                    avg_modulus = np.mean([p['effective_modulus'] for p in properties_profile])

                    if avg_modulus > max_performance:
                        max_performance = avg_modulus
                        optimal_profile = profile_type

                return {
                    'optimal_profile': optimal_profile,
                    'maximum_performance': max_performance,
                    'profile_comparison': profiles
                }

            def thermal_stress_analysis(self):
                """Análise de tensões térmicas"""
                positions = np.linspace(0, 1, 50)
                temperature_gradient = np.linspace(300, 800, 50)  # K

                thermal_stresses = []

                for i, pos in enumerate(positions):
                    # Propriedades locais
                    props = self.calculate_effective_properties(pos)

                    # Coeficiente de expansão térmica
                    alpha = 10e-6 + 5e-6 * props['composition_fraction']  # 1/K

                    # Diferença de temperatura
                    delta_T = temperature_gradient[i] - temperature_gradient[0]

                    # Tensão térmica
                    sigma_thermal = props['effective_modulus'] * alpha * delta_T

                    thermal_stresses.append(sigma_thermal)

                return {
                    'positions': positions,
                    'thermal_stresses': thermal_stresses,
                    'maximum_thermal_stress': max(thermal_stresses),
                    'thermal_stress_gradient': np.gradient(thermal_stresses)
                }

        fgm = FunctionallyGradedMaterial(composition_profile, property_gradients)
        properties = fgm.calculate_effective_properties(0.5)
        optimization = fgm.optimize_gradation_profile()
        thermal_analysis = fgm.thermal_stress_analysis()

        return {
            'fgm_material': fgm,
            'effective_properties': properties,
            'profile_optimization': optimization,
            'thermal_stress_analysis': thermal_analysis
        }

    def nanocomposite_reinforcement(self, nanoparticle_properties, matrix_material):
        """
        Reforço com nanopartículas em compósitos
        """
        class NanocompositeReinforcement:
            def __init__(self, nanoparticles, matrix):
                self.nanoparticles = nanoparticles
                self.matrix = matrix

            def calculate_reinforcement_mechanisms(self):
                """Calcular mecanismos de reforço"""
                # Mecanismo 1: Reforço por carga de transferência (shear lag)
                shear_lag_strength = self._shear_lag_model()

                # Mecanismo 2: Reforço por restrição térmica
                thermal_constraint_strength = self._thermal_constraint_model()

                # Mecanismo 3: Reforço por nucleação de deslocalizações
                dislocation_strength = self._dislocation_nucleation_model()

                total_reinforcement = shear_lag_strength + thermal_constraint_strength + dislocation_strength

                return {
                    'shear_lag_contribution': shear_lag_strength,
                    'thermal_constraint_contribution': thermal_constraint_strength,
                    'dislocation_contribution': dislocation_strength,
                    'total_reinforcement': total_reinforcement
                }

            def _shear_lag_model(self):
                """Modelo de shear lag para reforço"""
                # Comprimento crítico
                d = self.nanoparticles.get('diameter', 20e-9)  # m
                tau = self.nanoparticles.get('interface_strength', 100e6)  # Pa
                sigma_f = self.nanoparticles.get('fiber_strength', 2000e6)  # Pa

                l_critical = sigma_f * d / (2 * tau)

                # Eficiência de reforço
                aspect_ratio = self.nanoparticles.get('aspect_ratio', 10)
                efficiency = 1 - l_critical / (2 * aspect_ratio * d)

                reinforcement = efficiency * sigma_f * self.nanoparticles.get('volume_fraction', 0.05)

                return reinforcement

            def _thermal_constraint_model(self):
                """Modelo de restrição térmica"""
                # Diferença de coeficiente de expansão térmica
                alpha_np = self.nanoparticles.get('thermal_expansion', 5e-6)  # 1/K
                alpha_matrix = self.matrix.get('thermal_expansion', 50e-6)  # 1/K

                delta_alpha = alpha_matrix - alpha_np
                delta_T = self.nanoparticles.get('processing_temperature', 400) - 300  # K

                # Tensão térmica
                sigma_thermal = self.matrix.get('modulus', 3e9) * delta_alpha * delta_T

                reinforcement = sigma_thermal * self.nanoparticles.get('volume_fraction', 0.05)

                return reinforcement

            def _dislocation_nucleation_model(self):
                """Modelo de nucleação de deslocalizações"""
                # Tamanho da partícula
                r = self.nanoparticles.get('diameter', 20e-9) / 2

                # Módulo de cisalhamento
                G = self.matrix.get('shear_modulus', 1e9)  # Pa

                # Força de Orowan
                b = 0.25e-9  # Vetor de Burgers
                volume_fraction = self.nanoparticles.get('volume_fraction', 0.05)

                # Tensão de reforço
                sigma_orowan = (0.8 * G * b) / (np.pi * r * np.sqrt(1 - volume_fraction))

                return sigma_orowan

            def optimize_nanoparticle_size(self):
                """Otimizar tamanho das nanopartículas"""
                sizes = np.logspace(-9, -6, 20)  # 1nm a 1μm
                reinforcements = []

                for size in sizes:
                    self.nanoparticles['diameter'] = size

                    mechanisms = self.calculate_reinforcement_mechanisms()
                    total_reinforcement = mechanisms['total_reinforcement']
                    reinforcements.append(total_reinforcement)

                optimal_size = sizes[np.argmax(reinforcements)]

                return {
                    'optimal_size': optimal_size,
                    'maximum_reinforcement': max(reinforcements),
                    'size_reinforcement_curve': list(zip(sizes, reinforcements))
                }

        nc_reinforcement = NanocompositeReinforcement(nanoparticle_properties, matrix_material)
        mechanisms = nc_reinforcement.calculate_reinforcement_mechanisms()
        optimization = nc_reinforcement.optimize_nanoparticle_size()

        return {
            'reinforcement_analysis': nc_reinforcement,
            'reinforcement_mechanisms': mechanisms,
            'size_optimization': optimization
        }
```

**Compósitos Multifuncionais:**
- Propriedades de lâmina compósita
- Análise de laminado usando teoria das lâminas
- Materiais com gradiente funcional
- Reforço com nanopartículas em nanocompósitos

---

## 3. CONSIDERAÇÕES FINAIS

Os materiais avançados representam a convergência entre ciência dos materiais, física computacional e engenharia, oferecendo ferramentas para projetar materiais com propriedades específicas para aplicações desafiadoras. Os modelos apresentados fornecem uma base sólida para:

1. **Modelagem Quântica**: Bandas eletrônicas, DFT e dinâmica molecular
2. **Microestrutura**: Campo de fase, contorno de grão e precipitação
3. **Propriedades Mecânicas**: Tensor elástico e condutividade térmica
4. **Materiais Funcionais**: Piezoelétricos, memória de forma e magnetostrição
5. **Compósitos Avançados**: Laminados e materiais gradientes funcionais

**Próximos Passos Recomendados**:
1. Dominar mecânica quântica e DFT para materiais
2. Explorar simulações de microestrutura e crescimento de grãos
3. Aplicar ML para descoberta de novos materiais
4. Desenvolver aplicações práticas em compósitos estruturais
5. Integrar modelagem multiescala com experimentos

---

*Documento preparado para fine-tuning de IA em Materiais Avançados*
*Versão 1.0 - Preparado para implementação prática*
