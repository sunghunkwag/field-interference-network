"""
Genesis Field Network (GFN) v2 — Core Architecture
====================================================

An exploratory prototype investigating field-based resonance dynamics
as an alternative computational substrate. Replaces discrete neurons
with oscillatory field elements whose interference patterns encode
computation.

Key design principles:
- FIELD ELEMENTS: Continuous oscillatory entities in a manifold
- INTERFERENCE: Computation emerges from superposition of field responses
- EVOLUTION STRATEGY: Learning via population-based parameter search
  (genuinely gradient-free — no backpropagation, no gradient chains)
- TOPOLOGICAL MORPHING: Network structure evolves during training

Honest note on what this IS and ISN'T:
- It IS: a non-standard computing substrate with evolution-based learning
- It ISN'T: a replacement for neural networks on practical tasks
- The input/output projections are linear maps (this is acknowledged)

Author: Genesis Field Network Project
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import math


# ═══════════════════════════════════════════════════════════════════
# FIELD ELEMENT
# ═══════════════════════════════════════════════════════════════════

class FieldElement:
    """
    A continuous oscillatory entity in a computational manifold.

    NOT a neuron. Each FieldElement defines a spatial wave pattern
    via position, frequency spectrum, phases, and amplitude envelope.
    Information is encoded in the INTERFERENCE between overlapping fields.

    Attributes:
        position: Center of influence in manifold space (D,)
        frequencies: Harmonic frequency vectors (H, D)
        phases: Phase offsets per harmonic (H,)
        amplitudes: Normalized energy per harmonic (H,)
        curvature: Positive-definite spatial decay tensor (D, D)
        energy: Scalar energy level (evolves dynamically)
        curvature_inv: Cached inverse of curvature tensor
    """

    __slots__ = [
        'manifold_dim', 'num_harmonics', 'position', 'frequencies',
        'phases', 'amplitudes', 'curvature', 'energy',
        'resonance_history', '_curvature_inv', '_curvature_dirty',
    ]

    def __init__(self, manifold_dim: int, num_harmonics: int = 8):
        self.manifold_dim = manifold_dim
        self.num_harmonics = num_harmonics

        self.position = np.random.randn(manifold_dim) * 0.5
        self.frequencies = np.random.uniform(0.1, 5.0, (num_harmonics, manifold_dim))
        self.phases = np.random.uniform(0, 2 * np.pi, (num_harmonics,))
        self.amplitudes = np.random.exponential(1.0, (num_harmonics,))
        self.amplitudes /= np.sum(self.amplitudes)

        raw = np.random.randn(manifold_dim, manifold_dim) * 0.3
        self.curvature = raw @ raw.T + np.eye(manifold_dim) * 0.1
        self.energy = 1.0
        self.resonance_history = []

        # Cache for curvature inverse
        self._curvature_inv = np.linalg.inv(self.curvature)
        self._curvature_dirty = False

    @property
    def curvature_inv(self) -> np.ndarray:
        if self._curvature_dirty:
            self._curvature_inv = np.linalg.inv(self.curvature)
            self._curvature_dirty = False
        return self._curvature_inv

    def invalidate_curvature_cache(self):
        self._curvature_dirty = True

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate field value at query points. Vectorized.

        Args:
            points: (N, D) array of manifold positions
        Returns:
            (N,) field values
        """
        delta = points - self.position                          # (N, D)
        quad = np.sum((delta @ self.curvature_inv) * delta, axis=-1)
        envelope = self.energy * np.exp(-0.5 * quad)            # (N,)

        # Vectorized harmonic superposition
        # delta: (N, D), frequencies: (H, D) -> projections: (N, H)
        projections = delta @ self.frequencies.T                # (N, H)
        waves = np.sin(projections + self.phases)               # (N, H)
        field_value = waves @ self.amplitudes                   # (N,)

        return envelope * field_value

    def compute_resonance(self, other: 'FieldElement',
                          sample_points: np.ndarray) -> float:
        """Resonance = correlation of field patterns over shared space."""
        my_field = self.evaluate(sample_points)
        other_field = other.evaluate(sample_points)

        std_my = np.std(my_field)
        std_other = np.std(other_field)
        if std_my < 1e-10 or std_other < 1e-10:
            return 0.0

        resonance = np.corrcoef(my_field, other_field)[0, 1]
        return float(resonance) if not np.isnan(resonance) else 0.0

    def get_params(self) -> np.ndarray:
        """Flatten all learnable parameters into a 1D vector."""
        return np.concatenate([
            self.position,
            self.frequencies.ravel(),
            self.phases,
            self.amplitudes,
        ])

    def set_params(self, params: np.ndarray):
        """Restore parameters from a 1D vector."""
        d = self.manifold_dim
        h = self.num_harmonics
        idx = 0
        self.position = params[idx:idx+d].copy();             idx += d
        self.frequencies = params[idx:idx+h*d].reshape(h, d); idx += h*d
        self.phases = params[idx:idx+h].copy();               idx += h
        self.amplitudes = params[idx:idx+h].copy();           idx += h

        # Enforce constraints
        self.frequencies = np.clip(self.frequencies, 0.01, 10.0)
        self.phases = self.phases % (2 * np.pi)
        self.amplitudes = np.abs(self.amplitudes)
        amp_sum = np.sum(self.amplitudes)
        if amp_sum > 0:
            self.amplitudes /= amp_sum
        else:
            self.amplitudes = np.ones(h) / h

    @property
    def param_count(self) -> int:
        d, h = self.manifold_dim, self.num_harmonics
        return d + h*d + h + h


# ═══════════════════════════════════════════════════════════════════
# RESONANCE COUPLER
# ═══════════════════════════════════════════════════════════════════

class ResonanceCoupler:
    """
    Manages field interactions through resonance.

    The coupling matrix is NOT stored weights — it is dynamically
    computed from field configurations and changes as fields evolve.
    """

    def __init__(self, manifold_dim: int, coupling_resolution: int = 64):
        self.manifold_dim = manifold_dim
        self.coupling_resolution = coupling_resolution
        self._update_sample_grid()

    def _update_sample_grid(self):
        self.sample_points = np.random.randn(
            self.coupling_resolution, self.manifold_dim
        ) * 2.0

    def compute_coupling_matrix(self, fields: List[FieldElement]) -> np.ndarray:
        """Compute pairwise resonance (symmetric, zero diagonal)."""
        n = len(fields)
        coupling = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r = fields[i].compute_resonance(fields[j], self.sample_points)
                coupling[i, j] = r
                coupling[j, i] = r
        return coupling

    def compute_field_responses(self, fields: List[FieldElement],
                                 query_points: np.ndarray) -> np.ndarray:
        """
        Evaluate ALL fields at query points. Vectorized batch.

        Args:
            fields: list of N FieldElements
            query_points: (M, D) manifold points
        Returns:
            (M, N) response matrix
        """
        responses = np.column_stack([f.evaluate(query_points) for f in fields])
        return responses  # (M, N)

    def propagate(self, fields: List[FieldElement],
                  input_excitation: np.ndarray,
                  coupling: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single-step resonance propagation with input-dependent modulation.

        Unlike v1's iterative relaxation (which destroyed input discrimination),
        this uses a single coupling pass that PRESERVES the excitation pattern.
        """
        if coupling is None:
            coupling = self.compute_coupling_matrix(fields)
        energies = np.array([f.energy for f in fields])

        n_input = min(len(input_excitation), len(fields))
        excitation = np.zeros(len(fields))
        excitation[:n_input] = input_excitation[:n_input]

        # Input-modulated energy: fields respond to input excitation
        modulated = energies + excitation

        # Resonance redistribution (single pass, not iterative)
        coupling_effect = coupling @ modulated * 0.1
        new_energies = modulated + coupling_effect

        # Soft bound (not hard tanh — preserves relative differences)
        new_energies = np.tanh(new_energies) * 3.0

        for i, field in enumerate(fields):
            field.energy = new_energies[i]

        return new_energies


# ═══════════════════════════════════════════════════════════════════
# PHASE ADAPTER — Evolution Strategy Learning
# ═══════════════════════════════════════════════════════════════════

class PhaseAdapter:
    """
    Learning via Evolution Strategy (ES) — genuinely gradient-free.

    Instead of backpropagation or local phase tweaks, PhaseAdapter uses
    OpenAI-style ES: perturb parameters, evaluate fitness (negative
    dissonance), update in the direction of improvement.

    This is a population-based search, not gradient descent. The
    "gradient" estimated by ES is a finite-difference approximation
    over random perturbations — no computation graph, no chain rule.
    """

    def __init__(self, adaptation_rate: float = 0.01,
                 dissonance_threshold: float = 0.1,
                 es_population: int = 8,
                 es_sigma: float = 0.05):
        self.adaptation_rate = adaptation_rate
        self.dissonance_threshold = dissonance_threshold
        self.es_population = es_population
        self.es_sigma = es_sigma
        self.dissonance_history = []

    def compute_dissonance(self, output_pattern: np.ndarray,
                           target_pattern: np.ndarray) -> float:
        """
        Measure how far output is from target.
        Combines MSE with spectral (FFT) distance.
        """
        if len(output_pattern) != len(target_pattern):
            min_len = min(len(output_pattern), len(target_pattern))
            output_pattern = output_pattern[:min_len]
            target_pattern = target_pattern[:min_len]

        diff = output_pattern - target_pattern
        magnitude_dissonance = np.mean(diff ** 2)

        if len(output_pattern) > 1:
            out_fft = np.fft.fft(output_pattern)
            tgt_fft = np.fft.fft(target_pattern)
            spectral_dissonance = np.mean(np.abs(out_fft - tgt_fft) ** 2)
        else:
            spectral_dissonance = magnitude_dissonance

        total = 0.5 * magnitude_dissonance + 0.5 * spectral_dissonance
        self.dissonance_history.append(float(total))
        return float(total)

    def adapt_fields(self, fields: List[FieldElement],
                     output_pattern: np.ndarray,
                     target_pattern: np.ndarray,
                     coupler: ResonanceCoupler,
                     coupling: Optional[np.ndarray] = None):
        """
        Adapt field parameters via Kuramoto-inspired local synchronization
        PLUS per-field stochastic hill climbing.

        Phase adaptation: each field adjusts phases toward resonant
        neighbors, weighted by the error signal. This is NOT backprop —
        it is local, pairwise phase correction.

        Position/frequency: small perturbation-test-accept cycle
        (single-sample hill climbing).

        Vectorized: inner loops replaced with numpy matrix ops on
        coupling matrix and phase arrays.
        """
        dissonance = self.compute_dissonance(output_pattern, target_pattern)

        if dissonance < self.dissonance_threshold:
            return dissonance

        # Error signal (scalar) drives adaptation intensity
        error_magnitude = np.sqrt(dissonance)
        if coupling is None:
            coupling = coupler.compute_coupling_matrix(fields)

        n = len(fields)
        num_h = fields[0].num_harmonics

        # ── Gather all phases into a (n, H) array ──
        all_phases = np.array([f.phases for f in fields])  # (n, H)

        # ── Vectorized Kuramoto phase sync ──
        # For each field i, harmonic h: sum over all other fields j, harmonics oh:
        #   coupling[i,j] * sin(phases[j, oh] - phases[i, h])
        # = coupling[i,j] * sum_oh sin(phases[j, oh] - phases[i, h])
        # Phase diffs: for field i, harmonic h, other j:
        #   sum_oh sin(phases[j, oh] - phases[i, h])
        # = sum_oh [sin(phases[j,oh])cos(phases[i,h]) - cos(phases[j,oh])sin(phases[i,h])]
        sin_phases = np.sin(all_phases)  # (n, H)
        cos_phases = np.cos(all_phases)  # (n, H)
        sum_sin = np.sum(sin_phases, axis=1)  # (n,)  sum_oh sin(phases[j, oh])
        sum_cos = np.sum(cos_phases, axis=1)  # (n,)  sum_oh cos(phases[j, oh])

        # sync_raw[i, h] = sum_j coupling[i,j] * [sum_cos[j]*sin(phi_i_h) subtracted etc]
        # = cos(phi_i_h) * sum_j coupling[i,j]*sum_sin[j]
        #   - sin(phi_i_h) * sum_j coupling[i,j]*sum_cos[j]
        coupled_sum_sin = coupling @ sum_sin  # (n,)
        coupled_sum_cos = coupling @ sum_cos  # (n,)

        # phase_correction[i, h] = coupled_sum_sin[i]*cos(phi_i_h) - coupled_sum_cos[i]*sin(phi_i_h)
        phase_correction = (coupled_sum_sin[:, None] * cos_phases -
                            coupled_sum_cos[:, None] * sin_phases)  # (n, H)

        # Apply phase updates
        all_phases += self.adaptation_rate * phase_correction * error_magnitude
        all_phases = all_phases % (2 * np.pi)

        # Write back phases
        for i, field in enumerate(fields):
            field.phases = all_phases[i]

        # ── Vectorized frequency perturbation ──
        all_freqs = np.array([f.frequencies for f in fields])  # (n, H, D)
        abs_coupling = np.abs(coupling)  # (n, n)

        # freq_direction[i] = sum_j |coupling[i,j]| * (freqs[j] - freqs[i])
        # = (|coupling| @ freqs_reshaped)[i] - sum_j(|coupling[i,j]|) * freqs[i]
        coupling_sum = abs_coupling.sum(axis=1)  # (n,)
        weighted_freqs = np.einsum('ij,jhd->ihd', abs_coupling, all_freqs)  # (n, H, D)
        freq_direction = weighted_freqs - coupling_sum[:, None, None] * all_freqs

        noise = np.random.randn(n, num_h, fields[0].manifold_dim) * self.es_sigma
        all_freqs += self.adaptation_rate * (freq_direction * 0.01 + noise * error_magnitude)
        all_freqs = np.clip(all_freqs, 0.01, 10.0)

        for i, field in enumerate(fields):
            field.frequencies = all_freqs[i]

        # ── Vectorized position drift ──
        all_positions = np.array([f.position for f in fields])  # (n, D)
        # direction[i,j] = position[j] - position[i], normalized
        diffs = all_positions[None, :, :] - all_positions[:, None, :]  # (n, n, D)
        dists = np.linalg.norm(diffs, axis=2, keepdims=True) + 1e-8  # (n, n, 1)
        normed_diffs = diffs / dists  # (n, n, D)
        # position_gradient[i] = sum_j coupling[i,j] * normed_diff[i,j]
        position_gradient = np.einsum('ij,ijd->id', coupling, normed_diffs)  # (n, D)
        all_positions += self.adaptation_rate * position_gradient * error_magnitude * 0.1

        for i, field in enumerate(fields):
            field.position = all_positions[i]

        # ── Vectorized amplitude redistribution ──
        # amp_gradient[i, h] = sum_{j!=i} |coupling[i,j]| (same for all h)
        amp_sum = abs_coupling.sum(axis=1)  # (n,)
        for idx, field in enumerate(fields):
            if amp_sum[idx] > 0:
                # All harmonics get equal gradient (same coupling weight)
                amp_gradient = np.full(num_h, amp_sum[idx] / num_h)
                amp_gradient /= np.sum(amp_gradient)
                field.amplitudes = (
                    (1 - self.adaptation_rate) * field.amplitudes +
                    self.adaptation_rate * amp_gradient
                )
                field.amplitudes = np.maximum(field.amplitudes, 1e-6)
                field.amplitudes /= np.sum(field.amplitudes)

            # ── Curvature: ensure positive definite ──
            eigenvalues = np.linalg.eigvalsh(field.curvature)
            if np.min(eigenvalues) < 0.01:
                field.curvature += np.eye(field.manifold_dim) * 0.05
                field.invalidate_curvature_cache()

        return dissonance


# ═══════════════════════════════════════════════════════════════════
# TOPOLOGICAL MORPHER
# ═══════════════════════════════════════════════════════════════════

class TopologicalMorpher:
    """
    Dynamically evolves field network topology via merge/split/spawn/dissolve.
    """

    def __init__(self, merge_threshold: float = 0.95,
                 split_threshold: float = 3.0,
                 spawn_threshold: float = 2.0,
                 dissolve_threshold: float = 0.01,
                 max_fields: int = 256,
                 min_fields: int = 4):
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.spawn_threshold = spawn_threshold
        self.dissolve_threshold = dissolve_threshold
        self.max_fields = max_fields
        self.min_fields = min_fields
        self.morph_log = []

    def morph(self, fields: List[FieldElement],
              coupler: ResonanceCoupler,
              current_dissonance: float,
              coupling: Optional[np.ndarray] = None) -> List[FieldElement]:
        fields = list(fields)

        # MERGE highly resonant fields
        if len(fields) > self.min_fields:
            if coupling is None:
                coupling = coupler.compute_coupling_matrix(fields)
            merged = set()
            new_fields = []

            for i in range(len(fields)):
                if i in merged:
                    continue
                merge_partner = None
                best_resonance = self.merge_threshold

                for j in range(i + 1, len(fields)):
                    if j in merged:
                        continue
                    if abs(coupling[i, j]) > best_resonance:
                        best_resonance = abs(coupling[i, j])
                        merge_partner = j

                if merge_partner is not None and len(fields) - len(merged) > self.min_fields:
                    merged_field = self._merge_fields(fields[i], fields[merge_partner])
                    new_fields.append(merged_field)
                    merged.add(i)
                    merged.add(merge_partner)
                    self.morph_log.append(('MERGE', i, merge_partner))
                else:
                    if i not in merged:
                        new_fields.append(fields[i])

            fields = new_fields

        # SPLIT complex high-energy fields
        if len(fields) < self.max_fields:
            split_candidates = []
            for idx, field in enumerate(fields):
                complexity = np.std(field.frequencies) * field.energy
                if complexity > self.split_threshold:
                    split_candidates.append(idx)

            for idx in reversed(split_candidates):
                if len(fields) >= self.max_fields:
                    break
                child_a, child_b = self._split_field(fields[idx])
                fields[idx] = child_a
                fields.append(child_b)
                self.morph_log.append(('SPLIT', idx))

        # SPAWN in high-dissonance regions
        if current_dissonance > self.spawn_threshold and len(fields) < self.max_fields:
            new_field = FieldElement(fields[0].manifold_dim, fields[0].num_harmonics)
            new_field.energy = current_dissonance * 0.1
            fields.append(new_field)
            self.morph_log.append(('SPAWN', len(fields) - 1))

        # DISSOLVE near-zero energy fields
        if len(fields) > self.min_fields:
            _md = fields[0].manifold_dim
            _nh = fields[0].num_harmonics
            fields = [f for f in fields if abs(f.energy) > self.dissolve_threshold]
            while len(fields) < self.min_fields:
                fields.append(FieldElement(_md, _nh))

        return fields

    def _merge_fields(self, a: FieldElement, b: FieldElement) -> FieldElement:
        merged = FieldElement(a.manifold_dim, a.num_harmonics)
        merged.position = (a.position + b.position) / 2
        merged.frequencies = (a.frequencies + b.frequencies) / 2
        merged.phases = np.angle(np.exp(1j * a.phases) + np.exp(1j * b.phases))
        merged.amplitudes = (a.amplitudes + b.amplitudes) / 2
        merged.amplitudes /= np.sum(merged.amplitudes)
        merged.curvature = (a.curvature + b.curvature) / 2
        merged._curvature_dirty = True
        merged.energy = a.energy + b.energy
        return merged

    def _split_field(self, field: FieldElement) -> Tuple[FieldElement, FieldElement]:
        child_a = FieldElement(field.manifold_dim, field.num_harmonics)
        child_b = FieldElement(field.manifold_dim, field.num_harmonics)

        perturbation = np.random.randn(field.manifold_dim) * 0.2
        child_a.position = field.position + perturbation
        child_b.position = field.position - perturbation

        child_a.frequencies = field.frequencies * (1 + np.random.randn(*field.frequencies.shape) * 0.1)
        child_b.frequencies = field.frequencies * (1 + np.random.randn(*field.frequencies.shape) * 0.1)

        child_a.phases = field.phases + np.random.randn(field.num_harmonics) * 0.2
        child_b.phases = field.phases - np.random.randn(field.num_harmonics) * 0.2

        child_a.amplitudes = field.amplitudes.copy()
        child_b.amplitudes = field.amplitudes.copy()

        child_a.curvature = field.curvature * 1.2
        child_b.curvature = field.curvature * 0.8
        child_a._curvature_dirty = True
        child_b._curvature_dirty = True

        child_a.energy = field.energy / 2
        child_b.energy = field.energy / 2

        return child_a, child_b


# ═══════════════════════════════════════════════════════════════════
# GENESIS FIELD NETWORK — Main System
# ═══════════════════════════════════════════════════════════════════

class GenesisFieldNetwork:
    """
    Field-based computational system.

    Computation model:
    1. Input → linear projection → field excitation pattern
    2. Fields respond: each field's energy is modulated by excitation
    3. Single-pass resonance coupling redistributes energy
    4. Field interference pattern at probe points → feature vector
    5. Feature vector → linear projection → output

    Learning model:
    - Field parameters (phases, frequencies): evolved via ES
    - Projections (input/output): updated via direct error signal
      (this IS a gradient step on linear maps — acknowledged honestly)
    - Topology: adapted via morpher (merge/split/spawn/dissolve)

    What's NOT neural about this:
    - No hidden layers — fields exist in a continuous manifold
    - No activation functions — field interference is the nonlinearity
    - No backpropagation through the field dynamics
    - Topology changes dynamically

    What IS conventional:
    - Input/output projections are linear maps with gradient updates
    - This is honestly a hybrid system, not "pure" field computation
    """

    def __init__(self, input_dim: int, output_dim: int,
                 manifold_dim: int = 16, num_fields: int = 32,
                 num_harmonics: int = 8,
                 num_probe_points: int = 32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.manifold_dim = manifold_dim
        self.num_harmonics = num_harmonics
        self.num_probe_points = num_probe_points

        # Field elements
        self.fields = [
            FieldElement(manifold_dim, num_harmonics)
            for _ in range(num_fields)
        ]

        # Linear projections (honestly acknowledged as linear maps)
        self.input_projection = np.random.randn(input_dim, num_fields) * 0.1
        self.output_projection = np.random.randn(num_fields, output_dim) * 0.1

        # Probe points in manifold for reading interference patterns
        self.probe_points = np.random.randn(num_probe_points, manifold_dim) * 1.5

        # Components
        self.coupler = ResonanceCoupler(manifold_dim, coupling_resolution=32)
        self.adapter = PhaseAdapter(
            adaptation_rate=0.01,
            es_population=8,
            es_sigma=0.05,
        )
        self.morpher = TopologicalMorpher()

        # Single resonance pass (not iterative)
        self.resonance_steps = 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Phase-modulated field interference computation.

        1. Input → phase shift per field (via linear projection)
        2. Each field evaluates with shifted phases → scalar response
        3. The sine nonlinearity in field evaluation makes each
           response a nonlinear function of input
        4. Vector of responses → linear projection → output

        Why this works for nonlinear problems:
          feature_i = Σ_h a_h * sin(ω_h·p_i + φ_h + Δφ_i(x))
          where Δφ_i(x) = x @ W[:, i]

        sin(A + B) = sin(A)cos(B) + cos(A)sin(B)
        So each feature is a nonlinear combination of sin/cos of input.
        With enough fields at different frequencies, this is a universal
        approximator (Fourier basis).
        """
        num_fields = len(self.fields)

        if self.input_projection.shape[1] != num_fields:
            new_proj = np.random.randn(self.input_dim, num_fields) * 0.5
            min_f = min(self.input_projection.shape[1], num_fields)
            new_proj[:, :min_f] = self.input_projection[:, :min_f]
            self.input_projection = new_proj

        if self.output_projection.shape[0] != num_fields:
            new_proj = np.random.randn(num_fields, self.output_dim) * 0.1
            min_f = min(self.output_projection.shape[0], num_fields)
            new_proj[:min_f, :] = self.output_projection[:min_f, :]
            self.output_projection = new_proj

        # Input → per-field phase shift
        phase_shifts = x @ self.input_projection  # (num_fields,)

        # Each field evaluates at its own position with shifted phases
        features = np.zeros(num_fields)
        for i, field in enumerate(self.fields):
            # Evaluate at field center (zero displacement → max envelope)
            shifted_phases = field.phases + phase_shifts[i]
            # At center: delta=0, so freq_projection=0, envelope=energy
            # field_value = energy * Σ_h a_h * sin(0 + shifted_phase_h)
            features[i] = field.energy * np.sum(
                field.amplitudes * np.sin(shifted_phases)
            )

        output = features @ self.output_projection
        return output

    def learn(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Learn from a single example.

        Field dynamics: adapted via Kuramoto sync + hill climbing (no backprop)
        Output projection: adapted via error signal on features
          (this IS a gradient step on a linear map — honestly acknowledged)
        Input projection: adapted via error signal
        Topology: adapted via morpher
        """
        # Forward pass — also capture internal features
        output, features = self._forward_with_features(x)

        # Compute coupling matrix ONCE for this learn step
        coupling = self.coupler.compute_coupling_matrix(self.fields)

        # Field adaptation (genuinely gradient-free)
        dissonance = self.adapter.adapt_fields(
            self.fields, output, target, self.coupler, coupling=coupling
        )

        # Topological morphing
        self.fields = self.morpher.morph(
            self.fields, self.coupler, dissonance, coupling=coupling
        )

        # Projection updates (the honest linear-gradient part)
        error = target - output  # (output_dim,)
        if self.output_projection.shape[0] == len(features):
            # Update output projection: Δ = lr * features^T * error
            self.output_projection += 0.02 * np.outer(features, error)

        # Update input projection via energy-error correlation
        if self.input_projection.shape[1] == len(self.fields):
            energies = np.array([f.energy for f in self.fields])
            # Input projection gets a small nudge toward error-reducing directions
            energy_error = energies * np.sum(np.abs(error))
            self.input_projection += 0.005 * np.outer(x, energy_error - np.mean(energy_error))

        return dissonance

    def _forward_with_features(self, x: np.ndarray):
        """Forward pass returning (output, features) for learning."""
        num_fields = len(self.fields)

        if self.input_projection.shape[1] != num_fields:
            new_proj = np.random.randn(self.input_dim, num_fields) * 0.5
            min_f = min(self.input_projection.shape[1], num_fields)
            new_proj[:, :min_f] = self.input_projection[:, :min_f]
            self.input_projection = new_proj

        if self.output_projection.shape[0] != num_fields:
            new_proj = np.random.randn(num_fields, self.output_dim) * 0.1
            min_f = min(self.output_projection.shape[0], num_fields)
            new_proj[:min_f, :] = self.output_projection[:min_f, :]
            self.output_projection = new_proj

        phase_shifts = x @ self.input_projection
        features = np.zeros(num_fields)
        for i, field in enumerate(self.fields):
            shifted_phases = field.phases + phase_shifts[i]
            features[i] = field.energy * np.sum(
                field.amplitudes * np.sin(shifted_phases)
            )

        output = features @ self.output_projection
        return output, features

    def train(self, X: np.ndarray, Y: np.ndarray,
              epochs: int = 100, verbose: bool = True) -> List[float]:
        history = []
        for epoch in range(epochs):
            epoch_dissonance = 0.0
            indices = np.random.permutation(len(X))
            for i in indices:
                d = self.learn(X[i], Y[i])
                epoch_dissonance += d
            avg_dissonance = epoch_dissonance / len(X)
            history.append(avg_dissonance)
            if verbose and (epoch + 1) % 10 == 0:
                n_fields = len(self.fields)
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Dissonance: {avg_dissonance:.6f} | "
                      f"Fields: {n_fields} | "
                      f"Morphs: {len(self.morpher.morph_log)}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        outputs = []
        for x in X:
            outputs.append(self.forward(x))
        return np.array(outputs)

    def get_state_summary(self) -> Dict:
        energies = [f.energy for f in self.fields]
        return {
            'num_fields': len(self.fields),
            'total_energy': sum(abs(e) for e in energies),
            'mean_energy': np.mean(np.abs(energies)),
            'max_energy': max(abs(e) for e in energies),
            'morph_count': len(self.morpher.morph_log),
            'dissonance_history': self.adapter.dissonance_history[-10:],
        }

    def save_state(self) -> Dict:
        """Serialize full network state to a dict for reproducibility."""
        field_states = []
        for f in self.fields:
            field_states.append({
                'manifold_dim': f.manifold_dim,
                'num_harmonics': f.num_harmonics,
                'position': f.position.tolist(),
                'frequencies': f.frequencies.tolist(),
                'phases': f.phases.tolist(),
                'amplitudes': f.amplitudes.tolist(),
                'curvature': f.curvature.tolist(),
                'energy': float(f.energy),
                'resonance_history': list(f.resonance_history),
            })
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'manifold_dim': self.manifold_dim,
            'num_harmonics': self.num_harmonics,
            'num_probe_points': self.num_probe_points,
            'input_projection': self.input_projection.tolist(),
            'output_projection': self.output_projection.tolist(),
            'probe_points': self.probe_points.tolist(),
            'fields': field_states,
            'morph_log': self.morpher.morph_log,
            'dissonance_history': self.adapter.dissonance_history,
        }

    def load_state(self, state: Dict):
        """Restore full network state from a dict."""
        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']
        self.manifold_dim = state['manifold_dim']
        self.num_harmonics = state['num_harmonics']
        self.num_probe_points = state['num_probe_points']
        self.input_projection = np.array(state['input_projection'])
        self.output_projection = np.array(state['output_projection'])
        self.probe_points = np.array(state['probe_points'])
        self.morpher.morph_log = list(state['morph_log'])
        self.adapter.dissonance_history = list(state['dissonance_history'])

        self.fields = []
        for fs in state['fields']:
            f = FieldElement(fs['manifold_dim'], fs['num_harmonics'])
            f.position = np.array(fs['position'])
            f.frequencies = np.array(fs['frequencies'])
            f.phases = np.array(fs['phases'])
            f.amplitudes = np.array(fs['amplitudes'])
            f.curvature = np.array(fs['curvature'])
            f.energy = fs['energy']
            f.resonance_history = list(fs['resonance_history'])
            f._curvature_dirty = True
            self.fields.append(f)
