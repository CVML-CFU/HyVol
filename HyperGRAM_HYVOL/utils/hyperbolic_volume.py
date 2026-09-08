import torch
import numpy as np
from torch.func import vmap

# HyperGRAM: Hyperbolic Gramian Volume Computation
# Combines GRAM's multimodal volume with MERU's hyperbolic geometry

def project_to_hyperboloid(x, curv, eps=1e-8):
    """Project points to hyperboloid to ensure they satisfy the constraint.

    The hyperboloid constraint is: -x_0^2 + ||x_space||^2 = -1/c
    This function recomputes the time component to ensure constraint is satisfied.

    Args:
        x: Tensor of shape (..., dim) - space components
        curv: Positive scalar or tensor - hyperboloid curvature
        eps: Small value for numerical stability

    Returns:
        Tensor of same shape as x - projected space components that satisfy constraint
    """
    # Ensure curvature is positive and bounded
    if isinstance(curv, torch.Tensor):
        curv = torch.clamp(curv, min=eps, max=100.0)
    else:
        curv = max(eps, min(curv, 100.0))
        curv = torch.tensor(curv, dtype=x.dtype, device=x.device)

    # The points are already space components, just ensure they're valid
    # Compute time component that would satisfy the constraint
    x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
    x_norm_sq = torch.clamp(x_norm_sq, min=0, max=1e10)

    # Time component: x_0 = sqrt(||x_space||^2 + 1/c)
    # We don't actually return the time component, just ensure the space components are valid
    # This projection ensures numerical stability

    # Optional: Scale down if norm is too large to prevent numerical issues
    max_norm = 100.0 / torch.sqrt(curv)
    current_norm = torch.sqrt(x_norm_sq)
    scale = torch.where(current_norm > max_norm,
                       max_norm / (current_norm + eps),
                       torch.ones_like(current_norm))

    # Apply scaling - scale already has keepdim=True, so shape is (..., 1)
    return x * scale


def exp_map0(v, curv, eps=1e-8):
    """Exponential map at origin to project to hyperboloid with enhanced stability.

    Args:
        v: Tensor of shape (batch, dim) - tangent vectors at origin
        curv: Positive scalar or tensor - hyperboloid curvature
        eps: Small value for numerical stability

    Returns:
        Tensor of same shape as v - points on hyperboloid
    """
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=eps, max=100.0)  # Also clamp max to prevent overflow

    # Ensure curvature is positive and bounded
    if isinstance(curv, torch.Tensor):
        curv = torch.clamp(curv, min=eps, max=100.0)
    else:
        curv = max(eps, min(curv, 100.0))
        # Convert to tensor for sqrt operation
        curv = torch.tensor(curv, dtype=v.dtype, device=v.device)

    sqrt_curv = torch.sqrt(curv)
    scale = sqrt_curv * v_norm

    # Use Taylor expansion for small values to avoid numerical instability
    # sinh(x)/x ≈ 1 + x²/6 + x⁴/120 + x⁶/5040 for small x
    small_scale = scale < 0.5  # Use Taylor for smaller values
    large_scale = ~small_scale

    coeff = torch.zeros_like(scale)

    # Taylor expansion up to 6th order for small values
    if small_scale.any():
        scale_small = scale[small_scale]
        # More accurate Taylor series: sinh(x)/x = 1 + x²/6 + x⁴/120 + x⁶/5040
        coeff[small_scale] = 1.0 + scale_small**2 / 6.0 + scale_small**4 / 120.0 + scale_small**6 / 5040.0

    # Standard computation for larger values
    if large_scale.any():
        scale_large = scale[large_scale]
        # Safe division with clamping
        scale_large_safe = torch.clamp(scale_large, min=eps)
        coeff[large_scale] = torch.sinh(scale_large) / scale_large_safe

    # Apply projection to ensure result is on hyperboloid
    return project_to_hyperboloid(coeff * v, curv, eps)


def lorentz_inner(x, y, curv, eps=1e-8):
    """Compute Lorentzian inner product with enhanced stability.

    Args:
        x: Tensor of shape (batch1, dim) - space components
        y: Tensor of shape (batch2, dim) - space components
        curv: Positive scalar or tensor - hyperboloid curvature
        eps: Small value for numerical stability

    Returns:
        Tensor of shape (batch1, batch2) - pairwise Lorentzian inner products
    """
    # Project inputs to ensure they're on the hyperboloid
    x = project_to_hyperboloid(x, curv, eps)
    y = project_to_hyperboloid(y, curv, eps)

    # Ensure curvature is positive and bounded
    if isinstance(curv, torch.Tensor):
        curv = torch.clamp(curv, min=eps, max=100.0)
    else:
        curv = max(eps, min(curv, 100.0))

    # Compute squared norms with clamping for stability
    x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y**2, dim=-1, keepdim=True)

    # Clamp to prevent overflow in time component
    x_norm_sq = torch.clamp(x_norm_sq, min=0, max=1e10)
    y_norm_sq = torch.clamp(y_norm_sq, min=0, max=1e10)

    # Compute time components with numerical protection
    x_time = torch.sqrt(torch.clamp(1.0/curv + x_norm_sq, min=eps))
    y_time = torch.sqrt(torch.clamp(1.0/curv + y_norm_sq, min=eps))

    # Compute spatial and temporal products with clamping
    spatial_dot = torch.clamp(x @ y.T, min=-1e10, max=1e10)
    temporal_dot = torch.clamp(x_time @ y_time.T, min=-1e10, max=1e10)

    # Lorentzian inner product: <x,y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n
    return -temporal_dot + spatial_dot


def compute_gram_element(xi, xj, curvi, curvj, eps=1e-8):
    """Compute single element of hyperbolic Gram matrix with enhanced stability.

    Args:
        xi, xj: Tensors of shape (dim,) - modality features
        curvi, curvj: Scalars - curvatures for each modality
        eps: Small value for numerical stability

    Returns:
        Scalar - Gram matrix element G_ij
    """
    # Project inputs to ensure they're on the hyperboloid
    xi = project_to_hyperboloid(xi, curvi, eps)
    xj = project_to_hyperboloid(xj, curvj, eps)

    # Ensure curvatures are positive and bounded
    curvi_safe = max(min(curvi, 100.0), eps)
    curvj_safe = max(min(curvj, 100.0), eps)

    # Compute squared norms with protection
    xi_norm_sq = torch.sum(xi**2)
    xj_norm_sq = torch.sum(xj**2)

    # Clamp norms to prevent overflow
    xi_norm_sq = torch.clamp(xi_norm_sq, min=0, max=1e10)
    xj_norm_sq = torch.clamp(xj_norm_sq, min=0, max=1e10)

    # Compute time components with numerical protection
    # Ensure we're working with tensors, not Python scalars
    xi_time_sq = torch.tensor(1.0/curvi_safe, device=xi.device, dtype=xi.dtype) + xi_norm_sq
    xj_time_sq = torch.tensor(1.0/curvj_safe, device=xj.device, dtype=xj.dtype) + xj_norm_sq

    # Use stable computation for time component
    xi_time = torch.sqrt(torch.clamp(xi_time_sq, min=eps))
    xj_time = torch.sqrt(torch.clamp(xj_time_sq, min=eps))

    # Compute spatial product with clamping
    spatial = torch.clamp(torch.dot(xi, xj), min=-1e10, max=1e10)

    # Compute temporal product with clamping
    temporal = torch.clamp(xi_time * xj_time, min=-1e10, max=1e10)

    # Lorentzian inner product: <x,y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n
    result = -temporal + spatial

    # Final clamping to prevent extreme values
    return torch.clamp(result, min=-1e10, max=1e10)


def hyperbolic_volume2(modality1, modality2, curvatures):
    """Compute 2-modal hyperbolic volume using vmap.

    Args:
        modality1: Tensor of shape (batch1, dim)
        modality2: Tensor of shape (batch2, dim)
        curvatures: Scalar, list of 2 values, or tensor

    Returns:
        Tensor of shape (batch1, batch2) - hyperbolic volumes (areas)
    """
    batch_size1 = modality1.shape[0]
    batch_size2 = modality2.shape[0]

    # Handle curvature inputs
    if isinstance(curvatures, (int, float)):
        curvatures = [curvatures] * 2
    elif isinstance(curvatures, torch.Tensor):
        if curvatures.numel() == 1:
            curvatures = [curvatures.item()] * 2
        else:
            curvatures = curvatures.tolist()[:2]

    curv_1, curv_2 = curvatures[0], curvatures[1]

    # Define function to compute single 2x2 Gram matrix and its volume (area)
    def compute_single_volume(m1, m2):
        # Build 2x2 Gram matrix using stack to avoid inplace operations
        g00 = compute_gram_element(m1, m1, curv_1, curv_1)
        g11 = compute_gram_element(m2, m2, curv_2, curv_2)
        g01 = compute_gram_element(m1, m2, curv_1, curv_2)

        # Construct 2x2 matrix using stack operations
        row0 = torch.stack([g00, g01])
        row1 = torch.stack([g01, g11])
        G = torch.stack([row0, row1])

        # Add adaptive regularization for numerical stability
        G_norm = torch.norm(G, p='fro')
        # Use torch.where instead of if for vmap compatibility
        reg_strength = torch.where(G_norm > 10.0,
                                  torch.tensor(1e-4, device=G.device, dtype=G.dtype),
                                  torch.tensor(1e-6, device=G.device, dtype=G.dtype))

        eye = torch.eye(2, device=G.device, dtype=G.dtype)
        G_reg = G + reg_strength * eye

        # Compute area as sqrt(|det(G)|) with stability checks
        det = torch.det(G_reg)

        # Handle numerical issues using torch.where
        det = torch.where(torch.isnan(det) | torch.isinf(det),
                          torch.tensor(1e-8, device=det.device, dtype=det.dtype),
                          det)

        # Ensure positive and bounded
        area = torch.sqrt(torch.clamp(torch.abs(det), min=1e-8, max=1e10))

        return area

    # Use nested vmap to compute all batch1 x batch2 areas
    compute_all_volumes = vmap(vmap(compute_single_volume, in_dims=(None, 0)), in_dims=(0, None))

    volumes = compute_all_volumes(modality1, modality2)
    return volumes


def hyperbolic_volume3(language, video, audio, curvatures):
    """Compute 3-modal hyperbolic volume using vmap.

    Args:
        language: Tensor of shape (batch1, dim)
        video: Tensor of shape (batch2, dim)
        audio: Tensor of shape (batch2, dim)
        curvatures: Scalar, list of 3 values, or tensor

    Returns:
        Tensor of shape (batch1, batch2) - hyperbolic volumes
    """
    batch_size1 = language.shape[0]
    batch_size2 = video.shape[0]

    # Handle curvature inputs
    if isinstance(curvatures, (int, float)):
        curvatures = [curvatures] * 3
    elif isinstance(curvatures, torch.Tensor):
        if curvatures.numel() == 1:
            curvatures = [curvatures.item()] * 3
        else:
            curvatures = curvatures.tolist()[:3]

    curv_l, curv_v, curv_a = curvatures[0], curvatures[1], curvatures[2]

    # Define function to compute single 3x3 Gram matrix and its volume
    def compute_single_volume(l, v, a):
        # Build 3x3 Gram matrix using stack to avoid inplace operations
        g00 = compute_gram_element(l, l, curv_l, curv_l)
        g11 = compute_gram_element(v, v, curv_v, curv_v)
        g22 = compute_gram_element(a, a, curv_a, curv_a)

        g01 = compute_gram_element(l, v, curv_l, curv_v)
        g02 = compute_gram_element(l, a, curv_l, curv_a)
        g12 = compute_gram_element(v, a, curv_v, curv_a)

        # Construct matrix using stack operations
        row0 = torch.stack([g00, g01, g02])
        row1 = torch.stack([g01, g11, g12])
        row2 = torch.stack([g02, g12, g22])
        G = torch.stack([row0, row1, row2])

        # Add adaptive regularization for numerical stability
        G_norm = torch.norm(G, p='fro')
        # Use torch.where instead of if for vmap compatibility
        reg_strength = torch.where(G_norm > 10.0,
                                  torch.tensor(1e-4, device=G.device, dtype=G.dtype),
                                  torch.tensor(1e-6, device=G.device, dtype=G.dtype))

        eye = torch.eye(3, device=G.device, dtype=G.dtype)
        G_reg = G + reg_strength * eye

        # Compute volume as sqrt(|det(G)|) with stability checks
        det = torch.det(G_reg)

        # Handle numerical issues using torch.where
        det = torch.where(torch.isnan(det) | torch.isinf(det),
                          torch.tensor(1e-8, device=det.device, dtype=det.dtype),
                          det)

        # Ensure positive and bounded
        volume = torch.sqrt(torch.clamp(torch.abs(det), min=1e-8, max=1e10))

        return volume

    # Use nested vmap to compute all batch1 x batch2 volumes
    compute_all_volumes = vmap(vmap(compute_single_volume, in_dims=(None, 0, 0)), in_dims=(0, None, None))

    volumes = compute_all_volumes(language, video, audio)
    return volumes


def hyperbolic_volume4(language, video, audio, subtitles, curvatures):
    """Compute 4-modal hyperbolic volume using vmap.

    Args:
        language: Tensor of shape (batch1, dim)
        video: Tensor of shape (batch2, dim)
        audio: Tensor of shape (batch2, dim)
        subtitles: Tensor of shape (batch2, dim)
        curvatures: Scalar, list of 4 values, or tensor

    Returns:
        Tensor of shape (batch1, batch2) - hyperbolic volumes
    """
    batch_size1 = language.shape[0]
    batch_size2 = video.shape[0]

    # Handle curvature inputs
    if isinstance(curvatures, (int, float)):
        curvatures = [curvatures] * 4
    elif isinstance(curvatures, torch.Tensor):
        if curvatures.numel() == 1:
            curvatures = [curvatures.item()] * 4
        else:
            curvatures = curvatures.tolist()[:4]

    curv_l, curv_v, curv_a, curv_s = curvatures[0], curvatures[1], curvatures[2], curvatures[3]

    # Define function to compute single 4x4 Gram matrix and its volume
    def compute_single_volume(l, v, a, s):
        # Build 4x4 Gram matrix using stack to avoid inplace operations
        g00 = compute_gram_element(l, l, curv_l, curv_l)
        g11 = compute_gram_element(v, v, curv_v, curv_v)
        g22 = compute_gram_element(a, a, curv_a, curv_a)
        g33 = compute_gram_element(s, s, curv_s, curv_s)

        g01 = compute_gram_element(l, v, curv_l, curv_v)
        g02 = compute_gram_element(l, a, curv_l, curv_a)
        g03 = compute_gram_element(l, s, curv_l, curv_s)
        g12 = compute_gram_element(v, a, curv_v, curv_a)
        g13 = compute_gram_element(v, s, curv_v, curv_s)
        g23 = compute_gram_element(a, s, curv_a, curv_s)

        # Construct matrix using stack operations
        row0 = torch.stack([g00, g01, g02, g03])
        row1 = torch.stack([g01, g11, g12, g13])
        row2 = torch.stack([g02, g12, g22, g23])
        row3 = torch.stack([g03, g13, g23, g33])
        G = torch.stack([row0, row1, row2, row3])

        # Add adaptive regularization for numerical stability
        G_norm = torch.norm(G, p='fro')
        # Use torch.where instead of if for vmap compatibility
        reg_strength = torch.where(G_norm > 10.0,
                                  torch.tensor(1e-4, device=G.device, dtype=G.dtype),
                                  torch.tensor(1e-6, device=G.device, dtype=G.dtype))

        eye = torch.eye(4, device=G.device, dtype=G.dtype)
        G_reg = G + reg_strength * eye

        # Compute volume as sqrt(|det(G)|) with stability checks
        det = torch.det(G_reg.float())  # Use float32 for stability

        # Handle numerical issues using torch.where
        det = torch.where(torch.isnan(det) | torch.isinf(det),
                          torch.tensor(1e-8, device=det.device, dtype=det.dtype),
                          det)

        # Ensure positive and bounded
        volume = torch.sqrt(torch.clamp(torch.abs(det), min=1e-8, max=1e10))

        return volume

    # Use nested vmap to compute all batch1 x batch2 volumes
    compute_all_volumes = vmap(vmap(compute_single_volume, in_dims=(None, 0, 0, 0)), in_dims=(0, None, None, None))

    volumes = compute_all_volumes(language, video, audio, subtitles)
    return volumes


def hyperbolic_volume5(language, video, audio, subtitles, depth, curvatures):
    """Compute 5-modal hyperbolic volume using vmap.

    Args:
        language: Tensor of shape (batch1, dim)
        video: Tensor of shape (batch2, dim)
        audio: Tensor of shape (batch2, dim)
        subtitles: Tensor of shape (batch2, dim)
        depth: Tensor of shape (batch2, dim)
        curvatures: Scalar, list of 5 values, or tensor

    Returns:
        Tensor of shape (batch1, batch2) - hyperbolic volumes
    """
    batch_size1 = language.shape[0]
    batch_size2 = video.shape[0]

    # Handle curvature inputs
    if isinstance(curvatures, (int, float)):
        curvatures = [curvatures] * 5
    elif isinstance(curvatures, torch.Tensor):
        if curvatures.numel() == 1:
            curvatures = [curvatures.item()] * 5
        else:
            curvatures = curvatures.tolist()[:5]

    curv_l, curv_v, curv_a, curv_s, curv_d = curvatures[0], curvatures[1], curvatures[2], curvatures[3], curvatures[4]

    # Define function to compute single 5x5 Gram matrix and its volume
    def compute_single_volume(l, v, a, s, d):
        # Build 5x5 Gram matrix using stack to avoid inplace operations
        g00 = compute_gram_element(l, l, curv_l, curv_l)
        g11 = compute_gram_element(v, v, curv_v, curv_v)
        g22 = compute_gram_element(a, a, curv_a, curv_a)
        g33 = compute_gram_element(s, s, curv_s, curv_s)
        g44 = compute_gram_element(d, d, curv_d, curv_d)

        g01 = compute_gram_element(l, v, curv_l, curv_v)
        g02 = compute_gram_element(l, a, curv_l, curv_a)
        g03 = compute_gram_element(l, s, curv_l, curv_s)
        g04 = compute_gram_element(l, d, curv_l, curv_d)
        g12 = compute_gram_element(v, a, curv_v, curv_a)
        g13 = compute_gram_element(v, s, curv_v, curv_s)
        g14 = compute_gram_element(v, d, curv_v, curv_d)
        g23 = compute_gram_element(a, s, curv_a, curv_s)
        g24 = compute_gram_element(a, d, curv_a, curv_d)
        g34 = compute_gram_element(s, d, curv_s, curv_d)

        # Construct matrix using stack operations
        row0 = torch.stack([g00, g01, g02, g03, g04])
        row1 = torch.stack([g01, g11, g12, g13, g14])
        row2 = torch.stack([g02, g12, g22, g23, g24])
        row3 = torch.stack([g03, g13, g23, g33, g34])
        row4 = torch.stack([g04, g14, g24, g34, g44])
        G = torch.stack([row0, row1, row2, row3, row4])

        # Add adaptive regularization for numerical stability
        G_norm = torch.norm(G, p='fro')
        # Use torch.where instead of if for vmap compatibility
        reg_strength = torch.where(G_norm > 10.0,
                                  torch.tensor(1e-4, device=G.device, dtype=G.dtype),
                                  torch.tensor(1e-6, device=G.device, dtype=G.dtype))

        eye = torch.eye(5, device=G.device, dtype=G.dtype)
        G_reg = G + reg_strength * eye

        # Compute volume as sqrt(|det(G)|) with stability checks
        det = torch.det(G_reg.float())  # Use float32 for stability

        # Handle numerical issues using torch.where
        det = torch.where(torch.isnan(det) | torch.isinf(det),
                          torch.tensor(1e-8, device=det.device, dtype=det.dtype),
                          det)

        # Ensure positive and bounded
        volume = torch.sqrt(torch.clamp(torch.abs(det), min=1e-8, max=1e10))

        return volume

    # Use nested vmap to compute all batch1 x batch2 volumes
    compute_all_volumes = vmap(vmap(compute_single_volume, in_dims=(None, 0, 0, 0, 0)), in_dims=(0, None, None, None, None))

    volumes = compute_all_volumes(language, video, audio, subtitles, depth)
    return volumes


def hyperbolic_volume(language, *inputs, curvatures=1.0):
    """
    General function to compute hyperbolic volume for multimodal data.
    Automatically selects the appropriate function based on number of modalities.

    Args:
        language: Tensor of shape (batch1, dim) - anchor modality (typically text)
        *inputs: Variable number of tensors of shape (batch2, dim) - other modalities
        curvatures: Scalar, list, or tensor of curvature values

    Returns:
        Tensor of shape (batch1, batch2) - hyperbolic volumes
    """
    num_modalities = len(inputs)

    if num_modalities == 1:
        return hyperbolic_volume2(language, inputs[0], curvatures)
    elif num_modalities == 2:
        return hyperbolic_volume3(language, inputs[0], inputs[1], curvatures)
    elif num_modalities == 3:
        return hyperbolic_volume4(language, inputs[0], inputs[1], inputs[2], curvatures)
    elif num_modalities == 4:
        return hyperbolic_volume5(language, inputs[0], inputs[1], inputs[2], inputs[3], curvatures)
    else:
        raise ValueError(f"Unsupported number of modalities: {num_modalities + 1}. Only 2-5 modalities are supported.")


# Hybrid Euclidean-Hyperbolic volume

def hybrid_volume2(modality1, modality2, euclidean_weight=0.5, hyperbolic_weight=0.5, curvatures=1.0):
    """
    Compute hybrid Euclidean-Hyperbolic volume for 2 modalities.

    Args:
        modality1, modality2: Tensors of shape (batch, dim)
        euclidean_weight: Weight for Euclidean volume
        hyperbolic_weight: Weight for Hyperbolic volume
        curvatures: Curvature parameter(s) for hyperbolic space

    Returns:
        Tensor of shape (batch1, batch2) - hybrid volume
    """
    from .volume import volume_computation2

    # Compute Euclidean volume
    euclidean_volume = volume_computation2(modality1, modality2)

    # Compute Hyperbolic volume
    hyperbolic_vol = hyperbolic_volume2(modality1, modality2, curvatures)

    # Normalize volumes to similar scales
    euclidean_norm = euclidean_volume.mean().detach()
    hyperbolic_norm = hyperbolic_vol.mean().detach()

    if hyperbolic_norm > 0:
        scale_factor = euclidean_norm / (hyperbolic_norm + 1e-8)
        scale_factor = torch.clamp(scale_factor, min=0.001, max=10.0)
        hyperbolic_vol = hyperbolic_vol * scale_factor

    # Compute weighted combination
    hybrid_vol = euclidean_weight * euclidean_volume + hyperbolic_weight * hyperbolic_vol

    return hybrid_vol


def hybrid_volume3(language, video, audio, euclidean_weight=0.5, hyperbolic_weight=0.5, curvatures=1.0):
    """
    Compute hybrid Euclidean-Hyperbolic volume for 3 modalities.

    Args:
        language, video, audio: Tensors of shape (batch, dim)
        euclidean_weight: Weight for Euclidean volume
        hyperbolic_weight: Weight for Hyperbolic volume
        curvatures: Curvature parameter(s) for hyperbolic space

    Returns:
        Tensor of shape (batch1, batch2) - hybrid volume
    """
    from .volume import volume_computation3

    # Compute Euclidean volume
    euclidean_volume = volume_computation3(language, video, audio)

    # Compute Hyperbolic volume
    hyperbolic_vol = hyperbolic_volume3(language, video, audio, curvatures)

    # Normalize volumes to similar scales
    euclidean_norm = euclidean_volume.mean().detach()
    hyperbolic_norm = hyperbolic_vol.mean().detach()

    if hyperbolic_norm > 0:
        scale_factor = euclidean_norm / (hyperbolic_norm + 1e-8)
        scale_factor = torch.clamp(scale_factor, min=0.001, max=10.0)
        hyperbolic_vol = hyperbolic_vol * scale_factor

    # Compute weighted combination
    hybrid_vol = euclidean_weight * euclidean_volume + hyperbolic_weight * hyperbolic_vol

    return hybrid_vol


def hybrid_volume4(language, video, audio, subtitles, euclidean_weight=0.5, hyperbolic_weight=0.5, curvatures=1.0):
    """
    Compute hybrid Euclidean-Hyperbolic volume for 4 modalities.

    Args:
        language, video, audio, subtitles: Tensors of shape (batch, dim)
        euclidean_weight: Weight for Euclidean volume
        hyperbolic_weight: Weight for Hyperbolic volume
        curvatures: Curvature parameter(s) for hyperbolic space

    Returns:
        Tensor of shape (batch1, batch2) - hybrid volume
    """
    from .volume import volume_computation4

    # Compute Euclidean volume
    euclidean_volume = volume_computation4(language, video, audio, subtitles)

    # Compute Hyperbolic volume
    hyperbolic_vol = hyperbolic_volume4(language, video, audio, subtitles, curvatures)

    # Normalize volumes to similar scales
    euclidean_norm = euclidean_volume.mean().detach()
    hyperbolic_norm = hyperbolic_vol.mean().detach()

    if hyperbolic_norm > 0:
        scale_factor = euclidean_norm / (hyperbolic_norm + 1e-8)
        scale_factor = torch.clamp(scale_factor, min=0.001, max=10.0)
        hyperbolic_vol = hyperbolic_vol * scale_factor

    # Compute weighted combination
    hybrid_vol = euclidean_weight * euclidean_volume + hyperbolic_weight * hyperbolic_vol

    return hybrid_vol


def hybrid_volume5(language, video, audio, subtitles, depth, euclidean_weight=0.5, hyperbolic_weight=0.5, curvatures=1.0):
    """
    Compute hybrid Euclidean-Hyperbolic volume for 5 modalities.

    Args:
        language, video, audio, subtitles, depth: Tensors of shape (batch, dim)
        euclidean_weight: Weight for Euclidean volume
        hyperbolic_weight: Weight for Hyperbolic volume
        curvatures: Curvature parameter(s) for hyperbolic space

    Returns:
        Tensor of shape (batch1, batch2) - hybrid volume
    """
    from .volume import volume_computation5

    # Compute Euclidean volume
    euclidean_volume = volume_computation5(language, video, audio, subtitles, depth)

    # Compute Hyperbolic volume
    hyperbolic_vol = hyperbolic_volume5(language, video, audio, subtitles, depth, curvatures)

    # Normalize volumes to similar scales
    euclidean_norm = euclidean_volume.mean().detach()
    hyperbolic_norm = hyperbolic_vol.mean().detach()

    if hyperbolic_norm > 0:
        scale_factor = euclidean_norm / (hyperbolic_norm + 1e-8)
        scale_factor = torch.clamp(scale_factor, min=0.001, max=10.0)
        hyperbolic_vol = hyperbolic_vol * scale_factor

    # Compute weighted combination
    hybrid_vol = euclidean_weight * euclidean_volume + hyperbolic_weight * hyperbolic_vol

    return hybrid_vol


def hybrid_volume(language, *inputs, euclidean_weight=0.5, hyperbolic_weight=0.5, curvatures=1.0):
    """
    Unified hybrid volume computation for variable number of modalities.

    Args:
        language: Tensor of shape (batch1, dim) - anchor modality
        *inputs: Variable number of tensors of shape (batch2, dim)
        euclidean_weight: Weight for Euclidean volume
        hyperbolic_weight: Weight for Hyperbolic volume
        curvatures: Curvature parameter(s) for hyperbolic space

    Returns:
        Tensor of shape (batch1, batch2) - hybrid volume
    """
    num_modalities = len(inputs)

    if num_modalities == 1:
        return hybrid_volume2(language, inputs[0], euclidean_weight, hyperbolic_weight, curvatures)
    elif num_modalities == 2:
        return hybrid_volume3(language, inputs[0], inputs[1], euclidean_weight, hyperbolic_weight, curvatures)
    elif num_modalities == 3:
        return hybrid_volume4(language, inputs[0], inputs[1], inputs[2], euclidean_weight, hyperbolic_weight, curvatures)
    elif num_modalities == 4:
        return hybrid_volume5(language, inputs[0], inputs[1], inputs[2], inputs[3], euclidean_weight, hyperbolic_weight, curvatures)
    else:
        raise ValueError(f"Unsupported number of modalities: {num_modalities + 1}. Only 2-5 modalities are supported.")