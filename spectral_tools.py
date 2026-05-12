"""
spectral_tools.py
-----------------
Outils numériques pour l'analyse spectrale de la Hessienne d'un réseau de neurones.

Implémente :
  - HessianVectorProduct        : H @ v sans construire H  (Pearlmutter, 1994)
  - LanczosAlgorithm            : tridiagonalisation dans le sous-espace de Krylov
  - StochasticLanczosQuadrature : Algorithme 1 de Ghorbani et al. (2019)
  - AnisotropicPerturbation     : chocs directionnels guidés par le spectre

Références :
  - Pearlmutter (1994)   : Fast Exact Multiplication by the Hessian
  - Ghorbani et al. (2019) : An Investigation into Neural Net Optimization
    via Hessian Eigenvalue Density  (arXiv:1901.10159)
  - Golub & Welsch (1969) : Calculation of Gauss quadrature rules
"""

import numpy as np
import tensorflow as tf
from scipy.linalg import eigh_tridiagonal
from typing import Callable, List, Literal, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# 1. Hessian-Vector Product (HVP)
# ════════════════════════════════════════════════════════════════════════════

class HessianVectorProduct:
    """
    Calcule H @ v sans construire la Hessienne H explicitement.

    Principe (opérateur R de Pearlmutter) :
    ─────────────────────────────────────────
    On veut   H @ v = ∂²L/∂θ² · v.

    Astuce : si on définit  gv(θ) = ∇L(θ) · v  (scalaire différentiable en θ),
    alors   ∂(gv)/∂θ = H @ v  (dérivée d'un produit scalaire).

    En TF, deux GradientTape imbriqués suffisent :
      tape_inner : calcule g  = ∇L(θ)          (gradient standard)
      tape_outer : calcule ∂(g·v)/∂θ = H @ v   (dérivée du produit scalaire)

    Coût : O(p) mémoire  (pas de stockage de H),  O(p·n) calcul (≈ 2 backprop).

    Paramètres
    ----------
    model      : modèle Keras entraîné
    loss_fn    : (y_true, y_pred) → scalaire
    data_x     : images, shape (N, H, W, C)
    data_y     : étiquettes, shape (N,)
    batch_size : None = full-batch sur data_x (plus précis).
                 Entier = average sur mini-batches (plus rapide mais bruité).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        loss_fn: Callable,
        data_x: np.ndarray,
        data_y: np.ndarray,
        batch_size: Optional[int] = None,
    ):
        self.model      = model
        self.loss_fn    = loss_fn
        # Conversion en tf.constant une seule fois pour éviter la recopie à chaque HVP
        self.data_x     = tf.constant(data_x, dtype=tf.float32)
        self.data_y     = tf.constant(data_y, dtype=tf.int32)
        self.batch_size = batch_size

        # Mémorisation des formes et tailles pour la mise à plat / reconstruction
        self._trainable_vars: List[tf.Variable] = model.trainable_variables
        self._shapes = [v.shape for v in self._trainable_vars]
        self._sizes  = [int(tf.size(v)) for v in self._trainable_vars]
        self.n_params: int = sum(self._sizes)

    # ── Utilitaires de mise en forme ─────────────────────────────────────────

    def _flat_to_vars(self, flat_vec: tf.Tensor) -> List[tf.Tensor]:
        """
        Découpe un vecteur plat 1-D → liste de tenseurs de même forme
        que trainable_variables.

        Nécessaire car TF travaille variable par variable, mais l'algorithme
        de Lanczos manipule un seul vecteur de dimension n_params.
        """
        pieces, offset = [], 0
        for shape, size in zip(self._shapes, self._sizes):
            pieces.append(tf.reshape(flat_vec[offset: offset + size], shape))
            offset += size
        return pieces

    @staticmethod
    def _vars_to_flat(var_list: List[tf.Tensor]) -> tf.Tensor:
        """Inverse de _flat_to_vars : concatène la liste → vecteur 1-D."""
        return tf.concat([tf.reshape(v, [-1]) for v in var_list], axis=0)

    # ── Calcul du HVP (compilé par tf.function) ──────────────────────────────

    @tf.function(reduce_retracing=True)
    def _hvp_batch(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        v_flat: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calcule H_batch @ v sur un seul batch.

        Note sur les deux GradientTapes :
          tape_inner  →  g   = ∂L/∂θ      (gradient de la loss)
          tape_outer  →  ∂(g·v)/∂θ = H@v  (dérivée du produit scalaire g·v)

        On passe v_flat (1-D) plutôt qu'une liste Python pour que tf.function
        ne retrace pas à chaque changement de structure.

        Retourne H_batch @ v, shape (n_params,).
        """
        # Reconstruction de v sous la forme de variables (pour le produit scalaire)
        v_vars = self._flat_to_vars(v_flat)

        with tf.GradientTape() as tape_outer:
            # ── Tape interne : calcule le gradient ∇L(θ) ──────────────────────
            with tf.GradientTape() as tape_inner:
                # training=False : BatchNorm utilise ses statistiques figées
                # (mode inférence), ce qui donne une Hessienne "propre" au
                # minimum courant, indépendante du batch de normalisation.
                preds = self.model(x, training=False)
                loss  = self.loss_fn(y, preds)
            grads = tape_inner.gradient(loss, self._trainable_vars)

            # ── Produit scalaire g · v, différentiable par rapport à θ ────────
            # C'est la quantité clé : sa dérivée en θ donne exactement H @ v.
            # On filtre les None (couches sans gradient, ex. BN en mode eval).
            gv = tf.add_n([
                tf.reduce_sum(g * vv)
                for g, vv in zip(grads, v_vars)
                if g is not None
            ])

        # ── Tape externe : d(g·v)/dθ = H @ v ──────────────────────────────────
        hvp = tape_outer.gradient(gv, self._trainable_vars)

        # Remplace les None éventuels par des zéros (couches gelées ou sans grad)
        hvp = [
            h if h is not None else tf.zeros_like(var)
            for h, var in zip(hvp, self._trainable_vars)
        ]
        return self._vars_to_flat(hvp)

    # ── Interface publique ────────────────────────────────────────────────────

    def compute(self, v: np.ndarray) -> np.ndarray:
        """
        Calcule H @ v.

        Si batch_size=None : HVP exact sur tout data_x.
        Sinon              : moyenne non-biaisée sur les mini-batches
                             (E[H_batch] = H car chaque batch est i.i.d.).

        Paramètres
        ----------
        v : np.ndarray, shape (n_params,)

        Retourne
        --------
        hvp : np.ndarray, shape (n_params,)
        """
        v_tensor = tf.constant(v, dtype=tf.float32)
        n        = len(self.data_x)

        if self.batch_size is None or self.batch_size >= n:
            # Full-batch : une seule passe, résultat exact
            return self._hvp_batch(self.data_x, self.data_y, v_tensor).numpy()
        else:
            # Mini-batch séquentiel : @tf.function n'est pas réentrant sur Metal GPU,
            # les appels parallèles provoqueraient des race conditions.
            n_batches = int(np.ceil(n / self.batch_size))
            hvp_accum = tf.zeros(self.n_params, dtype=tf.float32)
            for i in range(n_batches):
                x_b = self.data_x[i * self.batch_size : (i + 1) * self.batch_size]
                y_b = self.data_y[i * self.batch_size : (i + 1) * self.batch_size]
                hvp_accum = hvp_accum + self._hvp_batch(x_b, y_b, v_tensor)
            # Chaque _hvp_batch normalise déjà par la taille du batch → moyenne simple.
            return (hvp_accum / n_batches).numpy()


# ════════════════════════════════════════════════════════════════════════════
# 2. Algorithme de Lanczos
# ════════════════════════════════════════════════════════════════════════════

class LanczosAlgorithm:
    """
    Tridiagonalisation de H dans le sous-espace de Krylov K_m(H, v).

    Idée géométrique :
    ──────────────────
    À chaque pas j, on construit un vecteur orthogonal à tous les précédents
    en appliquant H une fois de plus. L'ensemble forme une base orthonormale Q
    du sous-espace K_m = span{v, Hv, H²v, ..., H^{m-1}v}.

    Dans cette base, H se représente comme une matrice tridiagonale T :
        T = Q^T H Q  ∈ R^{m×m}  (m ≪ n_params)

    Les valeurs propres de T (valeurs de Ritz) sont d'excellentes approximations
    des valeurs propres extrêmes de H après seulement m pas.

    Récurrence à 3 termes (Lanczos symétrique) :
    ──────────────────────────────────────────────
      z        = H q_j - α_j q_j - β_{j-1} q_{j-1}
      α_j      = q_j^T (H q_j)      ← coefficient diagonal de T
      β_j      = ‖z‖                ← coefficient sous-diagonal de T
      q_{j+1}  = z / β_j            ← prochain vecteur de base

    Si β_j = 0 : le sous-espace de Krylov est invariant (H a déjà été
    complètement capturé) → terminaison anticipée exacte.

    Ré-orthogonalisation :
    ──────────────────────
    En arithmétique finie, les vecteurs q_j perdent leur orthogonalité
    (phénomène de "perte d'orthogonalité" de Lanczos). On peut observer des
    "fausses valeurs propres" (ghost eigenvalues). Deux stratégies :
      - store_vectors=True  : ré-orthogonalisation complète (Gram-Schmidt
        contre tous les q précédents). Stable mais nécessite O(n·m) RAM.
      - store_vectors=False : ré-orthogonalisation locale (3 termes seulement).
        Légère mais moins stable. Suffisant pour la SLQ (densité lissée).
    """

    def run(
        self,
        matvec       : Callable[[np.ndarray], np.ndarray],
        v0           : np.ndarray,
        m            : int,
        store_vectors: bool = False,
        verbose      : bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Paramètres
        ----------
        matvec        : y = H @ x  (callable, R^n → R^n)
        v0            : vecteur de départ, shape (n,)
        m             : nombre de pas (ordre de la tridiagonale)
        store_vectors : stocke Q pour la ré-ortho complète et l'extraction
                        des vecteurs propres de Ritz
        verbose       : affiche la progression tous les 10 pas

        Retourne
        --------
        alpha : diagonale de T,       shape (m_eff,)
        beta  : sous-diagonale de T,  shape (m_eff - 1,)
        Q     : base de Krylov,       shape (n, m_eff)  ou None
        """
        n     = len(v0)
        # float32 pour économiser RAM : Q ∈ R^{n×m} en float64 coûte 2× plus.
        # Précision suffisante pour m ≤ 20 (reorthogonalisation complète).
        alpha = np.zeros(m,     dtype=np.float32)   # diagonale de T
        beta  = np.zeros(m - 1, dtype=np.float32)   # sous-diagonale de T

        # Allocation optionnelle de la base Q
        Q = np.zeros((n, m), dtype=np.float32) if store_vectors else None

        # ── Initialisation : q_1 = v0 / ‖v0‖ ─────────────────────────────────
        q_prev = np.zeros(n, dtype=np.float32)              # q_0 = 0 (fictif)
        q_curr = v0.astype(np.float32) / np.linalg.norm(v0) # q_1 normalisé

        if store_vectors:
            Q[:, 0] = q_curr

        # ── Première application : z = H q_1 - α_1 q_1 ───────────────────────
        z        = matvec(q_curr).astype(np.float32)
        alpha[0] = float(np.dot(q_curr, z))   # α_1 = q_1^T H q_1
        z        = z - alpha[0] * q_curr      # résidu après projection

        # ── Itérations j = 1, ..., m-1 ───────────────────────────────────────
        for j in range(1, m):

            if verbose and j % 10 == 0:
                print(f"    Lanczos step {j}/{m}  (β_{j-1}={beta[j-2]:.3e})")

            # β_j = ‖z‖ : norme du résidu = coefficient hors-diagonale de T
            beta[j - 1] = np.linalg.norm(z)

            # Terminaison anticipée si le sous-espace de Krylov est épuisé
            # (z = 0 ⟹ H q_j ∈ span{q_1,...,q_j} ⟹ T représente H exactement)
            if beta[j - 1] < 1e-12:
                if verbose:
                    print(f"    Terminaison anticipée à j={j} (β={beta[j-1]:.2e})")
                alpha = alpha[:j]
                beta  = beta[:j - 1]
                if store_vectors:
                    Q = Q[:, :j]
                return alpha, beta, Q

            # q_{j+1} = z / β_j  (avant ré-orthogonalisation)
            q_next = z / beta[j - 1]

            if store_vectors:
                # ── Ré-orthogonalisation complète (Gram-Schmidt) ─────────────
                # Soustrait les projections sur tous les vecteurs déjà calculés.
                # Garantit que q_{j+1} ⊥ {q_1, ..., q_j} même en flottant.
                # Coût : O(n·j) par pas, soit O(n·m²) au total.
                q_next = q_next - Q[:, :j] @ (Q[:, :j].T @ q_next)
                norm   = np.linalg.norm(q_next)
                if norm < 1e-12:
                    # Effondrement numérique : la base est déjà complète
                    if verbose:
                        print(f"    Effondrement numérique à j={j}")
                    alpha = alpha[:j]; beta = beta[:j - 1]; Q = Q[:, :j]
                    return alpha, beta, Q
                q_next /= norm
                Q[:, j] = q_next
            else:
                # ── Ré-orthogonalisation locale (3 termes) ───────────────────
                # Soustrait uniquement la composante sur q_{j-1} (déjà soustraite
                # implicitement par la récurrence pour q_j).
                # Moins stable mais économique en mémoire : bon pour la SLQ
                # où seule la tridiagonale T est exploitée (pas les vecteurs Q).
                q_next = q_next - np.dot(q_prev, q_next) * q_prev
                norm   = np.linalg.norm(q_next)
                if norm < 1e-12:
                    alpha = alpha[:j]; beta = beta[:j - 1]
                    return alpha, beta, None
                q_next /= norm

            # ── Avance d'un pas : mise à jour des vecteurs courant/précédent ─
            q_prev = q_curr
            q_curr = q_next

            # ── Application de H et calcul de α_j ────────────────────────────
            z        = matvec(q_curr).astype(np.float32)
            alpha[j] = float(np.dot(q_curr, z))   # α_j = q_j^T H q_j

            # Résidu pour le prochain pas (récurrence à 3 termes complète)
            # z ← H q_j - α_j q_j - β_{j-1} q_{j-1}
            z = z - alpha[j] * q_curr - beta[j - 1] * q_prev

        return alpha, beta, Q


# ════════════════════════════════════════════════════════════════════════════
# 3. Stochastic Lanczos Quadrature (SLQ)
# ════════════════════════════════════════════════════════════════════════════

class StochasticLanczosQuadrature:
    """
    Estimation de la densité spectrale de H — Algorithme 1 de Ghorbani (2019).

    Objectif :
    ──────────
    Estimer φ_σ(t) = (1/n) tr(f(H; t, σ²)) où
    f(λ; t, σ²) = (1/(σ√2π)) exp(-(t-λ)²/(2σ²)) est un noyau gaussien.

    Pour σ assez petit, φ_σ ≈ densité spectrale (histogramme lissé des λᵢ).

    Deux approximations successives :
    ───────────────────────────────────
    1. Estimateur de Hutchinson :
         φ_σ(t) = E_v[ v^T f(H) v ]   avec v ~ N(0, (1/n) I)
       En effet : E[v^T A v] = (1/n) tr(A) pour ce choix de variance.
       → Remplace le trace (coût O(n)) par une moyenne sur k tirages.

    2. Quadrature gaussienne de Gauss-Ritz :
         v^T f(H) v ≈ Σ_{i=1}^m ω_i f(ℓ_i; t, σ²)
       où (ℓ_i, ω_i) sont construits à partir de la tridiagonale T de Lanczos.
       → Remplace le produit matrice-vecteur f(H)v (coût O(n²)) par une somme
         sur m termes (coût O(m)).
       Exact pour tout polynôme de degré ≤ 2m-1 (Théorème 2.1).

    Paramètres
    ----------
    hvp    : instance de HessianVectorProduct
    n_params : nombre de paramètres entraînables
    m      : pas de Lanczos = ordre de la quadrature. Défaut 90.
             La précision croît exponentiellement avec m (éq. 7 de l'article).
    k      : nombre de vecteurs sondes. Défaut 10.
             L'erreur décroît en 1/√k (concentration gaussienne).
    sigma2 : variance du noyau gaussien. Défaut 1e-5.
             Plus petit = densité plus fine, mais m plus grand nécessaire.
    """

    def __init__(
        self,
        hvp     : HessianVectorProduct,
        n_params: int,
        m       : int   = 90,
        k       : int   = 10,
        sigma2  : float = 1e-5,
    ):
        self.hvp      = hvp
        self.n        = n_params
        self.m        = m
        self.k        = k
        self.sigma2   = sigma2
        self._lanczos = LanczosAlgorithm()

    # ── Méthodes internes ─────────────────────────────────────────────────────

    def _sample_probe(self) -> np.ndarray:
        """
        Tire v ~ N(0, (1/n) I_n).

        Ce choix de variance est crucial pour l'estimateur de Hutchinson :
            E[v^T A v] = (1/n) Σ_i λ_i = (1/n) tr(A) = φ_σ(t)
        (chaque composante contribue en moyenne 1/n à la trace normalisée).
        """
        return np.random.randn(self.n) / np.sqrt(self.n)

    def _gaussian_kernel(self, lam: np.ndarray, t: float) -> np.ndarray:
        """
        Noyau gaussien f(λ; t, σ²) = (1/(σ√2π)) exp(-(t-λ)²/(2σ²)).
        Vectorisé sur un tableau de valeurs propres λ.
        """
        sigma = np.sqrt(self.sigma2)
        return np.exp(-0.5 * ((t - lam) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def _estimate_single_draw(
        self, v: np.ndarray, t_values: np.ndarray
    ) -> np.ndarray:
        """
        Étape I de l'Algorithme 1 : estime φ_σ^(v)(t) = v^T f(H) v pour un v fixé.

        Connexion quadrature ↔ Lanczos (Théorème 2.2 de l'article) :
        ──────────────────────────────────────────────────────────────
        Après m pas de Lanczos partant de v/‖v‖, la tridiagonale T satisfait
        T = Q^T H Q. Ses valeurs propres ℓ_i sont les nœuds de quadrature,
        et les poids ω_i = (U_{1,i})² proviennent de la première ligne U_{1,:}
        de la matrice propre U (T = U L U^T).

        Intuition : ω_i mesure la "projection" de v sur le sous-espace propre ℓ_i
        → les directions où v a le plus de masse contribuent le plus.

        Retourne array shape (len(t_values),).
        """
        # Lanczos sans stockage de Q (seule T est nécessaire pour la quadrature)
        alpha, beta, _ = self._lanczos.run(
            self.hvp.compute, v, self.m, store_vectors=False
        )
        m_eff = len(alpha)   # peut être < m si terminaison anticipée

        if m_eff == 1:
            # Cas dégénéré : une seule valeur propre
            nodes   = alpha.copy()
            weights = np.array([1.0])
        else:
            # Décomposition spectrale de la tridiagonale T = U L U^T
            nodes, U = eigh_tridiagonal(alpha, beta)
            # Poids de quadrature gaussienne : ω_i = (U_{1,i})²
            # La première ligne de U encode la projection de v sur chaque
            # vecteur propre de T (et par extension, de H via le sous-espace de Krylov).
            weights = U[0, :] ** 2

        # Évaluation du noyau gaussien aux nœuds, pondérée par les poids
        # φ̂^(v)(t) = Σ_i ω_i f(ℓ_i; t, σ²)
        density = np.array([
            float(np.dot(weights, self._gaussian_kernel(nodes, t)))
            for t in t_values
        ])
        return density

    # ── Interface publique ────────────────────────────────────────────────────

    def estimate_density(
        self,
        t_values: np.ndarray,
        verbose : bool = True,
    ) -> np.ndarray:
        """
        Étape II de l'Algorithme 1 : moyenne sur k vecteurs sondes.

            φ̂_σ(t) = (1/k) Σ_{i=1}^k φ̂^(v_i)(t)

        Par la loi des grands nombres, φ̂_σ → φ_σ quand k → ∞.
        La concentration est exponentielle (Claim 2.3 de l'article) :
        avec n = 11M paramètres, k=10 suffit pour une erreur négligeable.

        Paramètres
        ----------
        t_values : grille d'évaluation, shape (T,)
        verbose  : affiche la progression

        Retourne
        --------
        density : densité estimée, shape (T,)
        """
        density_sum = np.zeros(len(t_values))

        for i in range(self.k):
            if verbose:
                print(f"  SLQ tirage {i + 1}/{self.k} ...")
            v = self._sample_probe()
            density_sum += self._estimate_single_draw(v, t_values)

        return density_sum / self.k

    def estimate_top_eigenvalues(
        self,
        m_lanczos: int = 15,
        verbose  : bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estime les valeurs propres dominantes et leurs vecteurs propres de Ritz
        par un seul run de Lanczos avec ré-orthogonalisation complète.

        Pourquoi un seul run suffit ?
        Les valeurs propres extrêmes de H convergent très vite dans le sous-espace
        de Krylov : après ~30 pas, λ_max est estimé à < 1 % d'erreur.

        Note RAM : Q ∈ R^{n × m_lanczos}, float32 (moitié RAM vs float64).
        Pour n=11M et m=30 : ~2.6 Go. Réduire m_lanczos si nécessaire.

        Retourne
        --------
        ritz_vals : valeurs propres approx., triées décroissantes, shape (m,)
        ritz_vecs : vecteurs propres approx., shape (n_params, m)
        """
        # Vecteur de départ aléatoire (pas de biais sur la direction initiale)
        v0  = np.random.randn(self.n)
        v0 /= np.linalg.norm(v0)

        if verbose:
            mem_gb = self.n * m_lanczos * 8 / 1e9
            print(f"  Lanczos (m={m_lanczos}, store_vectors=True, RAM≈{mem_gb:.1f} Go) ...")

        alpha, beta, Q = self._lanczos.run(
            self.hvp.compute, v0, m_lanczos,
            store_vectors=True, verbose=verbose
        )

        if len(alpha) == 1:
            return alpha, Q

        # Décomposition de la tridiagonale de taille m (rapide : O(m³) avec m≪n)
        ritz_vals, U = eigh_tridiagonal(alpha, beta)

        # Tri décroissant (convention : λ_1 ≥ λ_2 ≥ ...)
        idx       = np.argsort(ritz_vals)[::-1]
        ritz_vals = ritz_vals[idx]
        U         = U[:, idx]

        # Vecteurs de Ritz dans l'espace des paramètres : y_i = Q u_i
        # (changement de base : sous-espace de Krylov → espace original R^n)
        ritz_vecs = Q @ U

        return ritz_vals, ritz_vecs

    def compute_flatness_ratio(
        self,
        lambda_max: float,
        density   : np.ndarray,
        t_values  : np.ndarray,
    ) -> dict:
        """
        Critère spectral de flatness : ζ = λ_max / λ_bulk.

        λ_bulk est la médiane spectrale (médiane de la distribution pondérée
        par la densité), qui approche le centre du bulk de Marchenko-Pastur.
        Un ζ élevé indique un spike très isolé du bulk → minimum sharp.

        Retourne dict : lambda_max, lambda_bulk, ratio, is_sharp
        """
        # CDF approchée par intégration numérique de la densité
        dt         = t_values[1] - t_values[0]
        cdf        = np.cumsum(density) * dt
        cdf       /= cdf[-1]   # normalisation → CDF entre 0 et 1
        # λ_bulk = médiane : premier t tel que CDF(t) ≥ 0.5
        lambda_bulk = float(t_values[np.searchsorted(cdf, 0.5)])

        ratio = lambda_max / max(abs(lambda_bulk), 1e-8)

        return {
            "lambda_max" : lambda_max,
            "lambda_bulk": lambda_bulk,
            "ratio"      : ratio,
            "is_sharp"   : ratio > 10.0,   # seuil indicatif, à calibrer
        }


# ════════════════════════════════════════════════════════════════════════════
# 4. Chocs anisotropes
# ════════════════════════════════════════════════════════════════════════════

class AnisotropicPerturbation:
    """
    Perturbations anisotropes des paramètres guidées par le spectre de H.

    Principe
    ────────
    Quand ζ > seuil, le minimum est jugé trop sharp. On applique un choc
    δθ aligné sur les k vecteurs propres dominants (les spikes), puis on
    ré-optimise pour chercher un minimum plus plat.

    Pourquoi les directions des spikes ?
    Les top-k vecteurs propres q₁,...,q_k correspondent aux directions de
    courbure maximale — les "murs" du bassin sharp. Les perturber permet de
    franchir ces murs. Les directions du bulk (λ ≈ 0) correspondent surtout
    à des symétries de jauge (l'espace des paramètres redondants) et sont
    peu informatives pour l'exploration.

    Modes de pondération (sᵢ ∈ {-1,+1} aléatoire pour chaque direction)
    ──────────────────────────────────────────────────────────────────────
    'uniform'   δθ = ε · (1/√k) Σ sᵢ qᵢ
                Poids égal. Simple, ignore la courbure locale.

    'inv_sqrt'  δθ ∝ Σ (sᵢ/√λᵢ) qᵢ  [RECOMMANDÉ]
                Raisonnement : dans L ≈ ½(θ-θ*)ᵀH(θ-θ*), un pas δᵢ = c/√λᵢ
                dans la direction qᵢ provoque ΔL = ½λᵢ(c/√λᵢ)² = c²/2.
                → Chaque direction spike reçoit la même "énergie" de choc,
                  exploration équilibrée sans être dominée par λ_max.

    'sqrt'      δθ ∝ Σ (sᵢ√λᵢ) qᵢ
                Plus agressif dans les directions les plus raides.
                Utile pour franchir en priorité le mur de λ_max.

    'bulk'      δθ ⊥ {q₁,...,qₖ}  [EXPLORATION VALLÉE]
                Vecteur gaussien aléatoire projeté orthogonalement aux k
                vecteurs propres dominants (Gram-Schmidt).
                → Vit entièrement dans le bulk (λ ≈ 0) : la perte varie peu
                  après le choc. Permet de "glisser le long du fond de vallée"
                  vers des minima connectés sans franchir de mur.
                Contraste avec les modes spike : loss_choc ≈ loss_pre (pas de
                montée), mais le bassin d'attraction final peut différer.

    Paramètres
    ----------
    k              : nb de vecteurs propres utilisés. Pour MNIST (K=10 classes),
                     Sagun et al. prédit ~K-1=9 spikes → k=9 recommandé.
    epsilon        : norme L2 du choc. Calibrer dans [0.01, 0.1].
    mode           : 'inv_sqrt' | 'uniform' | 'sqrt' | 'bulk'
    zeta_threshold : seuil de ζ. Défaut 10.
    seed           : graine RNG pour la reproductibilité.
    """

    def __init__(
        self,
        k             : int   = 9,
        epsilon       : float = 0.05,
        mode          : Literal["inv_sqrt", "uniform", "sqrt", "bulk"] = "inv_sqrt",
        zeta_threshold: float = 10.0,
        seed          : Optional[int] = None,
    ):
        self.k              = k
        self.epsilon        = epsilon
        self.mode           = mode
        self.zeta_threshold = zeta_threshold
        self._rng           = np.random.default_rng(seed)

    def should_perturb(self, zeta: float) -> bool:
        """True si ζ > seuil → choc recommandé."""
        return zeta > self.zeta_threshold

    def compute_perturbation(
        self,
        ritz_vals: np.ndarray,
        ritz_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Calcule δθ avec ‖δθ‖ = epsilon exactement.

        Paramètres
        ----------
        ritz_vals : valeurs de Ritz décroissantes, shape (m,)
        ritz_vecs : vecteurs de Ritz, shape (n_params, m)

        Retourne
        --------
        delta_theta : shape (n_params,)
        """
        k_eff = min(self.k, ritz_vals.shape[0])
        vals  = ritz_vals[:k_eff]      # λ₁ ≥ ... ≥ λ_k
        vecs  = ritz_vecs[:, :k_eff]   # colonnes q₁, ..., q_k

        # Signes aléatoires indépendants : chaque direction est explorée
        # dans un sens ou l'autre, ce qui rend le choc symétrique en moyenne.
        signs = self._rng.choice([-1.0, 1.0], size=k_eff)

        # ── Coefficients selon le mode ────────────────────────────────────────
        if self.mode == "uniform":
            # Contribution égale de chaque direction (norme unitaire)
            coeffs = signs / np.sqrt(k_eff)

        elif self.mode == "inv_sqrt":
            # Clip à 1e-8 pour éviter la division par zéro sur les λ ≈ 0
            lam_pos = np.maximum(vals, 1e-8)
            raw     = signs / np.sqrt(lam_pos)
            # Normalisation intermédiaire : on fixera ‖δθ‖=ε à la fin
            coeffs  = raw / (np.linalg.norm(raw) + 1e-12)

        elif self.mode == "sqrt":
            lam_pos = np.maximum(vals, 1e-8)
            raw     = signs * np.sqrt(lam_pos)
            coeffs  = raw / (np.linalg.norm(raw) + 1e-12)

        elif self.mode == "bulk":
            # ── Mode bulk : exploration perpendiculaire aux spikes ────────────
            # Vecteur gaussien aléatoire dans R^n_params, puis on retire
            # ses composantes sur chaque vecteur spike via Gram-Schmidt.
            # → δθ vit entièrement dans le sous-espace bulk (λ ≈ 0).
            n_params = ritz_vecs.shape[0]
            v = self._rng.standard_normal(n_params).astype(np.float32)
            # Projection orthogonale : v ← v - Σᵢ (v·qᵢ) qᵢ
            for i in range(k_eff):
                qi = vecs[:, i]
                v -= np.dot(v, qi) * qi
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                raise RuntimeError(
                    "Vecteur bulk nul après projection — "
                    "vérifier que ritz_vecs ne couvre pas tout R^n."
                )
            return v * (self.epsilon / norm_v)

        else:
            raise ValueError(
                f"Mode inconnu '{self.mode}'. Choisir : inv_sqrt, uniform, sqrt, bulk."
            )

        # ── Combinaison linéaire des vecteurs propres ─────────────────────────
        # δθ = vecs @ coeffs  est dans l'espace des paramètres R^n
        delta_flat = vecs @ coeffs    # shape (n_params,)

        # ── Mise à l'échelle finale : garantit ‖δθ‖ = epsilon ────────────────
        # Nécessaire pour les modes uniform/inv_sqrt/sqrt car la normalisation
        # intermédiaire et la combinaison linéaire peuvent modifier la norme.
        norm = np.linalg.norm(delta_flat)
        if norm < 1e-12:
            raise RuntimeError("Vecteur de perturbation nul — vérifier ritz_vecs.")
        delta_flat = delta_flat * (self.epsilon / norm)

        return delta_flat

    def apply_to_model(
        self,
        model     : tf.keras.Model,
        delta_flat: np.ndarray,
    ) -> None:
        """
        Ajoute δθ aux poids entraînables du modèle (in-place : θ ← θ + δθ).

        Parcourt les variables dans le même ordre que _flat_to_vars pour
        garantir la correspondance entre le vecteur plat et les tenseurs.
        """
        offset = 0
        for var in model.trainable_variables:
            size      = int(tf.size(var))
            delta_var = delta_flat[offset: offset + size].reshape(var.shape)
            # assign_add : opération TF in-place, préserve le graphe de calcul
            var.assign_add(tf.constant(delta_var, dtype=var.dtype))
            offset += size

    def perturb(
        self,
        model    : tf.keras.Model,
        ritz_vals: np.ndarray,
        ritz_vecs: np.ndarray,
        zeta     : float,
        verbose  : bool = True,
    ) -> Optional[np.ndarray]:
        """
        Interface tout-en-un : vérifie ζ, calcule et applique le choc.
        Retourne delta_flat si appliqué, None sinon.
        """
        if not self.should_perturb(zeta):
            if verbose:
                print(f"  Choc non appliqué : ζ={zeta:.2f} ≤ seuil={self.zeta_threshold}")
            return None

        delta = self.compute_perturbation(ritz_vals, ritz_vecs)
        self.apply_to_model(model, delta)

        if verbose:
            k_eff = min(self.k, len(ritz_vals))
            print(f"  Choc appliqué : mode='{self.mode}', k={k_eff}, ε={self.epsilon:.4f}")
            print(f"  ‖δθ‖ = {np.linalg.norm(delta):.6f}")
            print(f"  λ utilisées : {ritz_vals[:k_eff].round(4).tolist()}")

        return delta
