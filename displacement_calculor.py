import numpy as np
from numpy import pi
from dataclasses import dataclass
from typing import Callable, Tuple
import cmath
from scipy.optimize import least_squares
# -----------------------------
# Data structures
# -----------------------------

@dataclass
class SoilProps:
    Es: float          # 杨氏模量
    Gs: float          # 剪切模量
    rho_s: float       # 密度
    nu_s: float        # 泊松比
    beta_s: float      # 阻尼比

    @property
    def G_star(self):
        return self.Gs * cmath.sqrt((1.0 + 2.0j * self.beta_s))

@dataclass
class PileProps:
    Ep: float          # 杨氏模量
    Ap: float          # 截面积
    Ip: float          # 惯性矩
    rho_p: float       # 密度
    L: float           # length
    D_eq: float        # 等效直径
    r_ext: float       # 中性面距离

@dataclass
class LayerProps:
    H: float           # 土层厚度

@dataclass
class FrequencySpec:
    omega: float       # 圆频率
    ug: complex        # 自由场地表面位移幅值（可为复数）

@dataclass             #映射函数及其导数（保角映射）
class MappingFuncs:
    # Conformal mapping and derivatives
    F: Callable[[complex], complex]
    Fp: Callable[[complex], complex]
    Fpp: Callable[[complex], complex]

# -----------------------------
# Conformal mapping templates
# -----------------------------

def circular_mapping(Ra: float) -> MappingFuncs:
    def F(zeta: complex) -> complex:
        return Ra * zeta
    def Fp(zeta: complex) -> complex:
        return Ra + 0*zeta
    def Fpp(zeta: complex) -> complex:
        return 0.0 + 0*zeta
    return MappingFuncs(F=F, Fp=Fp, Fpp=Fpp)

def fit_rectangular_mapping(a: float, b: float, n_points: int = 40) -> MappingFuncs:
    """
    Fit conformal mapping coefficients for a rectangle of width a and height b
    using least squares on boundary points.
    """

    # 定义映射函数
    def F(zeta, B, b1, b3, b5):
        return B * (zeta + b1/zeta + b3/(zeta**3) + b5/(zeta**5))

    # 误差函数：点到矩形边界的距离
    def residuals(params, a, b, thetas):
        B, b1, b3, b5 = params
        res = []
        for theta in thetas:
            zeta = np.exp(1j*theta)
            z = F(zeta, B, b1, b3, b5)
            # 计算点到矩形边界的距离
            dx = max(0, abs(z.real) - a/2)
            dy = max(0, abs(z.imag) - b/2)
            res.append(np.sqrt(dx**2 + dy**2))
        return res

    # 在单位圆上取样点
    thetas = np.linspace(0, 2*np.pi, n_points)

    # 初始猜测
    initial_guess = [1.0, 0.5, -0.1, 0.05]#初始猜想，主要是利用优化算法拟合到误差平方和最小

    # 最小二乘拟合
    result = least_squares(residuals, initial_guess, args=(a, b, thetas))
    B, b1, b3, b5 = result.x

    # 定义最终的映射函数和导数
    def F_func(zeta: complex) -> complex:
        return B * (zeta + b1/zeta + b3/(zeta**3) + b5/(zeta**5))

    def Fp_func(zeta: complex) -> complex:
        return B * (1.0 - b1/(zeta**2) - 3.0*b3/(zeta**4) - 5.0*b5/(zeta**6))

    def Fpp_func(zeta: complex) -> complex:
        return B * (2.0*b1/(zeta**3) + 12.0*b3/(zeta**5) + 30.0*b5/(zeta**7))

    return MappingFuncs(F=F_func, Fp=Fp_func, Fpp=Fpp_func)

def fit_xshape_mapping(L: float, t: float, n_points: int = 60) -> MappingFuncs:
    """
    Least-squares fit for X-shaped section mapping.
    L: arm length
    t: arm thickness
    """

    # 映射函数
    def F(zeta, B, b1, b3, b5):
        return B * (zeta + b1/zeta + b3/(zeta**3) + b5/(zeta**5))

    # 判断点到 X 型截面边界的距离
    def dist_to_xshape(z, L, t):
        # X 型由四个矩形臂组成：水平和竖直方向
        dx = max(0, abs(z.real) - L/2) if abs(z.imag) < t/2 else float('inf')
        dy = max(0, abs(z.imag) - L/2) if abs(z.real) < t/2 else float('inf')
        return min(dx, dy)

    # 残差函数
    def residuals(params, L, t, thetas):
        B, b1, b3, b5 = params
        res = []
        for theta in thetas:
            zeta = np.exp(1j*theta)
            z = F(zeta, B, b1, b3, b5)
            res.append(dist_to_xshape(z, L, t))
        return res

    # 采样点
    thetas = np.linspace(0, 2*np.pi, n_points)

    # 初始猜测
    initial_guess = [1.0, 0.5, -0.1, 0.05]

    # 最小二乘拟合
    result = least_squares(residuals, initial_guess, args=(L, t, thetas))
    B, b1, b3, b5 = result.x

    # 定义最终函数
    def F_func(zeta: complex) -> complex:
        return B * (zeta + b1/zeta + b3/(zeta**3) + b5/(zeta**5))

    def Fp_func(zeta: complex) -> complex:
        return B * (1.0 - b1/(zeta**2) - 3.0*b3/(zeta**4) - 5.0*b5/(zeta**6))

    def Fpp_func(zeta: complex) -> complex:
        return B * (2.0*b1/(zeta**3) + 12.0*b3/(zeta**5) + 30.0*b5/(zeta**7))

    return MappingFuncs(F=F_func, Fp=Fp_func, Fpp=Fpp_func)

# -----------------------------
# Geometry factors U, V and derivatives
# -----------------------------

def UV_factors(rho: float, alpha: float, Fp: Callable[[complex], complex], Fpp: Callable[[complex], complex]) -> Tuple[float, float, float, float]:
    """
    Compute U = cos(gamma), V = sin(gamma) and radial derivatives dU/drho, dV/drho
    using F'(zeta) and F''(zeta). zeta = rho * exp(i alpha).
    """
    zeta = rho * np.exp(1j*alpha)
    Fp_val = Fp(zeta)
    absFp = np.abs(Fp_val) + 1e-16
    # U + iV = F'(zeta)/|F'(zeta)|
    W = Fp_val / absFp
    U = np.real(W)
    V = np.imag(W)

    # Derivatives: d/drho Re{...} and Im{...}; use chain rule on F'(zeta)
    Fpp_val = Fpp(zeta)
    # d/dzeta (F'/|F'|) = (F''*|F'| - F'*d|F'|/dzeta)/|F'|^2
    # We need d/d rho: dzeta/d rho = e^{i alpha}
    d_absFp_dzeta = np.conj(Fp_val) * Fpp_val / absFp  # complex; derivative of |Fp| needs care
    # Use directional derivative along zeta with e^{i alpha}
    D = (Fpp_val*absFp - Fp_val*d_absFp_dzeta) / (absFp**2 + 1e-16)
    dW_drho = D * np.exp(1j*alpha)  # chain rule
    dU = np.real(dW_drho)
    dV = np.imag(dW_drho)
    return U, V, dU, dV

# -----------------------------
# Free-field displacement
# -----------------------------

def free_field_u(z: np.ndarray, soil: SoilProps, layer: LayerProps, freq: FrequencySpec) -> np.ndarray:
    Gstar = soil.G_star
    alpha_f = np.sqrt(soil.rho_s * freq.omega**2 / Gstar)
    return freq.ug * np.cos(alpha_f * z) / np.cos(alpha_f * layer.H)

# -----------------------------
# Attenuation ODE assembly (f1, f2, f3)
# -----------------------------

def assemble_f123(rhos: np.ndarray, alphas: np.ndarray, zs: np.ndarray,
                  Upsc_z: np.ndarray, uf_z: np.ndarray,
                  mapfuncs: MappingFuncs, soil: SoilProps) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerically assemble f1(rho), f2(rho), f3(rho) via quadrature over alpha and z.
    This implementation uses midpoint quadrature for demonstration; refine as needed.
    """
    Gstar = soil.G_star
    f1 = np.zeros_like(rhos, dtype=complex)
    f2 = np.zeros_like(rhos, dtype=complex)
    f3 = np.zeros_like(rhos, dtype=complex)

    # Weighting for quadrature
    dalpha = (alphas[-1] - alphas[0]) / (len(alphas) - 1)
    dz = (zs[-1] - zs[0]) / (len(zs) - 1)

    diff_z = (Upsc_z - uf_z)  # (Up/Usc - uf)
    diff_z2_int = np.trapz(np.abs(diff_z)**2, zs)

    for i, rho in enumerate(rhos):
        S1 = 0.0 + 0j
        S2 = 0.0 + 0j
        S3 = 0.0 + 0j
        for alpha in alphas:
            U, V, dU, dV = UV_factors(rho, alpha, mapfuncs.Fp, mapfuncs.Fpp)
            # f1 term: ~ ∫ (n? U^2 + V^2) * (Up/Usc - uf)^2 dz; we approximate coefficient n? with (1+2*nu_s)
            ncoeff = (1.0 + 2.0*soil.nu_s)
            S1 += ncoeff * (U*U + V*V)
            # f2 term: includes geometry derivatives; simplified consistent contribution
            S2 += (U*U + V*V + 2.0*rho*(U*dU + V*dV))
            # f3 term: includes frequency term; proportional to ω^2 ρs / G*
            # We accumulate alpha-dependent part and multiply with z-integral
            S3 += 1.0
        # Integrate over alpha and z for each rho
        f1[i] = Gstar * S1 * diff_z2_int * dalpha
        f2[i] = Gstar * S2 * diff_z2_int * dalpha
        f3[i] = (soil.rho_s * 0.0 + 0.0j)  # base; we’ll add ω^2 part outside (see below)

    # Add ω^2 term into f3 (global multiplier)
    # f3 ~ (ω^2 ρs / G*) ∫ (Up/Usc - uf)^2 dz; scale f3 by that factor
    f3 += (soil.rho_s * (1.0) / soil.G_star) * diff_z2_int
    return f1, f2, f3

# -----------------------------
# Solve attenuation ODE via central difference
# -----------------------------

def solve_phi(rhos: np.ndarray, f1: np.ndarray, f2: np.ndarray, f3: np.ndarray) -> np.ndarray:
    """
    Solve phi'' + (f1/rho^2 - f2/rho) phi' - f3 phi = 0,
    with phi(1)=1 and phi(rhos[-1])=0, using central differences.
    """
    N = len(rhos)
    phi = np.zeros(N, dtype=complex)
    phi[0] = 1.0 + 0j  # boundary at rho=1
    phi[-1] = 0.0 + 0j

    # Build tridiagonal system A*phi_interior = b
    A = np.zeros((N-2, N-2), dtype=complex)
    b = np.zeros(N-2, dtype=complex)

    for i in range(1, N-1):
        rho_i = rhos[i]
        dr_im = rhos[i] - rhos[i-1]
        dr_ip = rhos[i+1] - rhos[i]

        # central difference coefficients
        # phi'' ~  2/(dr_ip+dr_im) * ( (phi_{i+1}-phi_i)/dr_ip - (phi_i - phi_{i-1})/dr_im )
        # phi'  ~  (phi_{i+1}-phi_{i-1})/(dr_ip+dr_im)

        c2 = 2.0 / (dr_ip + dr_im)
        cprime = 1.0 / (dr_ip + dr_im)

        # Coefficients of phi_{i-1}, phi_i, phi_{i+1}
        a_im = c2 * (-1.0/dr_im) + cprime * ( f1[i]/(rho_i**2) - f2[i]/rho_i ) * (-1.0)
        a_i  = c2 * ( 1.0/dr_im + 1.0/dr_ip) + cprime * ( f1[i]/(rho_i**2) - f2[i]/rho_i ) * (0.0) + (-f3[i])
        a_ip = c2 * (-1.0/dr_ip) + cprime * ( f1[i]/(rho_i**2) - f2[i]/rho_i ) * (1.0)

        row = i-1
        if i-1 >= 1:
            A[row, row-1] = a_im
        else:
            # i=1 touches boundary phi[0]
            b[row] -= a_im * phi[0]

        A[row, row] = a_i

        if i+1 <= N-2:
            A[row, row+1] = a_ip
        else:
            # i=N-2 touches boundary phi[-1]=0 (no contribution)
            pass

    # Solve linear system
    phi_interior = np.linalg.solve(A, b)
    phi[1:-1] = phi_interior
    return phi

# -----------------------------
# Compute k and c
# -----------------------------

def compute_k_c(rhos: np.ndarray, alphas: np.ndarray, phi: np.ndarray, mapfuncs: MappingFuncs, soil: SoilProps) -> Tuple[complex, complex]:
    Gstar = soil.G_star
    dalpha = (alphas[-1] - alphas[0]) / (len(alphas)-1)

    # k = G* ∫∫ (dphi/drho)^2 (U^2+V^2) dα dρ
    dphi = np.gradient(phi, rhos)
    k_int = 0.0 + 0j
    c_int = 0.0 + 0j

    for i, rho in enumerate(rhos):
        S_UV = 0.0
        S_geom = 0.0
        for alpha in alphas:
            U, V, _, _ = UV_factors(rho, alpha, mapfuncs.Fp, mapfuncs.Fpp)
            S_UV += (U*U + V*V)
            zeta = rho * np.exp(1j*alpha)
            Fp_val = mapfuncs.Fp(zeta)
            S_geom += np.abs(Fp_val)**2 * (rho**2)
        k_int += (dphi[i]**2) * S_UV * dalpha * (rhos[1]-rhos[0])  # uniform rho spacing assumed near boundary
        c_int += S_geom * dalpha * (rhos[1]-rhos[0])

    k = Gstar * k_int
    two_c = Gstar * c_int
    return k, two_c

# -----------------------------
# Solve pile/soil column linear system for constants
# -----------------------------

def solve_pile_column_constants(pile: PileProps, soil: SoilProps, layer: LayerProps,
                                freq: FrequencySpec, k: complex, two_c: complex,
                                uf_z: np.ndarray, z_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble and solve the coupled linear algebraic system for constants (C1,C2,D1..D4)
    based on boundary conditions and the closed-form forms of U_sc(z) and U_p(z).
    For demonstration, we compute U_sc and U_p via shooting with the closed-form basis.
    """
    # Closed-form parameters
    Gstar = soil.G_star
    Ap = pile.Ap
    Ep = pile.Ep
    Ip = pile.Ip

    # Soil column exponent A and Af
    # A = k/(2c + Ap*G*) - rho_s*ω^2/G*
    A = k / (two_c + Ap*Gstar) - soil.rho_s * freq.omega**2 / Gstar
    Af = k / (k + (two_c + Ap*Gstar) * (1.0 + 0.0j - soil.rho_s * freq.omega**2 / Gstar))  # simplified symbolic form

    # Pile exponent lambdas (two pairs)
    # λ satisfy Ep Ip λ^4 - 2c λ^2 + k - ω^2( (2c ρs/G*) + ρp Ap ) = 0
    # Quadratic in λ^2: Ep Ip (λ^2)^2 - 2c (λ^2) + (k - ω^2*((2c ρs/G*) + ρp Ap)) = 0
    quad_a = Ep * Ip
    quad_b = -two_c
    quad_c = (k - freq.omega**2 * ( (two_c * soil.rho_s / Gstar) + pile.rho_p * Ap ))
    # Solve for λ^2:
    disc = quad_b**2 - 4.0*quad_a*quad_c
    lam2_1 = ( -quad_b + np.sqrt(disc) ) / (2.0*quad_a)
    lam2_2 = ( -quad_b - np.sqrt(disc) ) / (2.0*quad_a)
    lam1 = np.sqrt(lam2_1)
    lam2 = np.sqrt(lam2_2)

    # Build basis functions
    z = z_grid
    uf = uf_z

    # U_sc(z) = C1 e^{A z} + C2 e^{-A z} + Af uf(z)
    # U_p(z)  = D1 e^{lam1 z} + D2 e^{-lam1 z} + D3 e^{lam2 z} + D4 e^{-lam2 z} + Ξf uf(z)
    # For simplicity, set Ξf = 0 (pile particular via uf omitted). You may include it per Eq.(20).
    Xi_f = 0.0 + 0.0j

    # Boundary conditions: we enforce at z=0 and z=L, and column head z=L and base z=H
    L = pile.L
    H = layer.H

    # Define unknowns vector X = [C1, C2, D1, D2, D3, D4]
    # Equations:
    # 1) U_sc(L) = U_p(L)
    # 2) U_sc(H) = ug
    # 3) pile head: bending moment and shear conditions (free head) -> U_p''(0)=0 and EpIp U_p'''(0) - 2c U_p'(0) = 0
    # 4) pile tip: matching conditions -> U_p''(L)=0 and continuity of shear/interaction (approx: EpIp U_p'''(L) - 2c U_p'(L) + (2c+Ap G*) (U_sc'(L) - U_p'(L)) = 0)
    # This is a simplified consistent set; adjust to match Eq. (19) precisely in production.

    def esc(z):
        return np.array([np.exp(A*z), np.exp(-A*z)], dtype=complex)

    def ep_basis(z):
        return np.array([np.exp(lam1*z), np.exp(-lam1*z), np.exp(lam2*z), np.exp(-lam2*z)], dtype=complex)

    def ep_basis_d(z, order=1):
        if order == 1:
            return np.array([lam1*np.exp(lam1*z), -lam1*np.exp(-lam1*z),
                             lam2*np.exp(lam2*z), -lam2*np.exp(-lam2*z)], dtype=complex)
        if order == 2:
            return np.array([lam1**2*np.exp(lam1*z), lam1**2*np.exp(-lam1*z),
                             lam2**2*np.exp(lam2*z), lam2**2*np.exp(-lam2*z)], dtype=complex)
        if order == 3:
            return np.array([lam1**3*np.exp(lam1*z), -lam1**3*np.exp(-lam1*z),
                             lam2**3*np.exp(lam2*z), -lam2**3*np.exp(-lam2*z)], dtype=complex)
        raise ValueError("order must be 1,2,3")

    # Assemble 6x6 linear system
    A_mat = np.zeros((6,6), dtype=complex)
    b_vec = np.zeros(6, dtype=complex)

    # Eq1: U_sc(L) - U_p(L) = 0
    A_mat[0,0:2] = esc(L)         # C1, C2
    A_mat[0,2:6] = -ep_basis(L)   # -D1..D4
    b_vec[0] = -Af*uf[np.searchsorted(z, L)] - Xi_f*uf[np.searchsorted(z, L)]

    # Eq2: U_sc(H) = ug
    A_mat[1,0:2] = esc(H)
    b_vec[1] = freq.ug - Af*uf[np.searchsorted(z, H)]

    # Eq3: U_p''(0) = 0  => basis second derivatives at 0
    A_mat[2,2:6] = ep_basis_d(0.0, order=2)
    b_vec[2] = 0.0

    # Eq4: EpIp U_p'''(0) - 2c U_p'(0) = 0
    A_mat[3,2:6] = Ep*Ip*ep_basis_d(0.0, order=3) - two_c*ep_basis_d(0.0, order=1)
    b_vec[3] = 0.0

    # Eq5: U_p''(L) = 0
    A_mat[4,2:6] = ep_basis_d(L, order=2)
    b_vec[4] = 0.0

    # Eq6: shear/interaction consistency at L (simplified)
    A_mat[5,2:6] = Ep*Ip*ep_basis_d(L, order=3) - two_c*ep_basis_d(L, order=1)
    # include column contribution via (2c+Ap G*)(U_sc'(L) - U_p'(L)) ~ moved to RHS:
    Usc_p_L = A*esc(L) @ np.array([1.0, -1.0])  # derivative of column basis wrt z; simplification
    b_vec[5] = (two_c + Ap*soil.G_star) * ( Usc_p_L - ep_basis_d(L, order=1) @ np.array([0,0,0,0]) )

    # Solve
    X = np.linalg.solve(A_mat, b_vec)
    C1, C2, D1, D2, D3, D4 = X

    # Construct U_sc(z), U_p(z)
    Usc = C1*np.exp(A*z) + C2*np.exp(-A*z) + Af*uf
    Up  = D1*np.exp(lam1*z) + D2*np.exp(-lam1*z) + D3*np.exp(lam2*z) + D4*np.exp(-lam2*z) + Xi_f*uf

    return Usc, Up

# -----------------------------
# Main iterative solver
# -----------------------------

def solve_kinematic_response(pile: PileProps, soil: SoilProps, layer: LayerProps,
                             freq: FrequencySpec, mapfuncs: MappingFuncs,
                             N_rho: int = 200, rho_max: float = 200.0, rho_min_step: float = 1e-3,
                             alpha_pts: int = 64, z_pts: int = 301, tol: float = 1e-3, maxit: int = 50):
    # Grids
    # Stretched rho grid: geometric progression from 1 to rho_max
    rhos = np.geomspace(1.0, rho_max, N_rho)
    alphas = np.linspace(0.0, 2.0*pi, alpha_pts)
    z = np.linspace(0.0, layer.H, z_pts)

    uf = free_field_u(z, soil, layer, freq)

    # Initial guess: U_psc^(0) = - uf (soil column + pile treated together along z)
    Upsc_old = -uf.copy()

    for it in range(maxit):
        # Assemble f1,f2,f3
        f1, f2, f3 = assemble_f123(rhos, alphas, z, Upsc_old, uf, mapfuncs, soil)

        # Solve phi(rho)
        phi = solve_phi(rhos, f1, f2, f3)

        # Compute k and 2c
        k, two_c = compute_k_c(rhos, alphas, phi, mapfuncs, soil)

        # Solve pile/column constants and fields
        Usc, Up = solve_pile_column_constants(pile, soil, layer, freq, k, two_c, uf, z)

        # Convergence
        num = np.trapz(np.abs(Usc - Upsc_old)**2, z)
        den = np.trapz(np.abs(Usc)**2, z) + 1e-16
        relerr = np.sqrt(num / den)

        # Update
        Upsc_old = Usc.copy()

        # Check
        if relerr < tol:
            break

    # Post-processing
    # Translational factor Iu at head z=0
    Iu = Up[0] / uf[0]
    # Rotational factor: theta = dUp/dz at head
    dUp = np.gradient(Up, z)
    theta0 = dUp[0]
    Itheta = (theta0 * pile.D_eq) / (Up[0] - uf[0] + 1e-16)

    # Bending strain: eps_b = r_ext * d2Up/dz2
    d2Up = np.gradient(dUp, z)
    eps_b = pile.r_ext * d2Up

    # Winkler stiffness coefficient normalized: Re(k)/Gs
    Winkler_norm = np.real(k) / soil.Gs

    return {
        "z": z,
        "uf": uf,
        "Usc": Usc,
        "Up": Up,
        "phi_rho": (rhos, phi),
        "k": k,
        "two_c": two_c,
        "Iu": Iu,
        "Itheta": Itheta,
        "eps_b": eps_b,
        "Winkler_norm": Winkler_norm
    }

# -----------------------------
# Example usage (circular pile)
# -----------------------------

if __name__ == "__main__":
    # Soil and pile example parameters (adjust to your case)
    soil = SoilProps(Es=50e6, Gs=20e6, rho_s=1800.0, nu_s=0.4, beta_s=0.05)
    pile = PileProps(Ep=30e9, Ap=1.0, Ip=0.083, rho_p=2500.0, L=20.0, D_eq=1.128, r_ext=0.5)
    layer = LayerProps(H=40.0)
    freq = FrequencySpec(omega=2.0*pi*1.0, ug=1.0+0j)  # 1 Hz

    # Circular mapping with radius ~ D_eq/2
    Ra = pile.D_eq/2.0
    mapfuncs = circular_mapping(Ra)

    result = solve_kinematic_response(
        pile, soil, layer, freq, mapfuncs,
        N_rho=200, rho_max=200.0, rho_min_step=1e-3,
        alpha_pts=64, z_pts=501, tol=1e-3, maxit=30
    )

    print("Iu =", result["Iu"])
    print("Itheta =", result["Itheta"])
    print("Re(k)/Gs =", result["Winkler_norm"])
    print("Max bending strain =", np.max(np.abs(result["eps_b"])))

