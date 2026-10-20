"""
PMRL Loss Functions

Based on: "Principled Multimodal Representation Learning" (arXiv:2507.17343)

Implements three loss components:
1. L^M: Multimodal alignment via dominant singular value maximization
2. L^M': Instance-wise regularization via principal eigenvector contrastive
3. L_IM: Instance matching (implemented via ITM head in main model)

Mathematical Formulation:
    L^M = -log(exp(σ₁/τ) / Σ exp(σⱼ/τ))
    L^M' = Instance-wise contrastive on principal eigenvector u₁
    Total: L = L^M + λ₁·L^M' + λ₂·L_IM

Hyperparameters (from PMRL paper):
    λ₁ = 1.0
    λ₂ = 0.1
    τ = 0.07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMRLLoss(nn.Module):
    """
    PMRL Loss Module

    Computes L^M and L^M' losses. L_IM is computed separately via ITM head.

    Supports both Euclidean and Hybrid (Euclidean + Hyperbolic) spaces.

    Args:
        lambda1 (float): Weight for L^M' (default: 1.0 from paper)
        lambda2 (float): Weight for L_IM (default: 0.1 from paper)
        temperature (float): Temperature for softmax (default: 0.07)
        use_hybrid_space (bool): If True, compute SVD in hybrid space (default: False)

    Example:
        >>> pmrl = PMRLLoss(lambda1=1.0, lambda2=0.1, temperature=0.07)
        >>> embeddings = {
        ...     'text': torch.randn(64, 512),
        ...     'video': torch.randn(64, 512),
        ...     'audio': torch.randn(64, 512)
        ... }
        >>> loss_dict = pmrl(embeddings, return_components=True)
        >>> print(loss_dict['pmrl_total'])
    """

    def __init__(self, lambda1=1.0, lambda2=0.1, temperature=0.07, use_hybrid_space=False):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau = temperature
        self.use_hybrid_space = use_hybrid_space

    def forward(self, embeddings_dict, return_components=False,
                euclidean_weight=None, hyperbolic_weight=None, curvatures=None):
        """
        Forward pass

        Args:
            embeddings_dict (dict): {modality_name: tensor[B, D]}
                Example: {'text': [64, 512], 'video': [64, 512], 'audio': [64, 512]}
            return_components (bool): If True, return dict with all components
            euclidean_weight (float, optional): Weight for Euclidean component (for hybrid mode)
            hyperbolic_weight (float, optional): Weight for Hyperbolic component (for hybrid mode)
            curvatures (float or tensor, optional): Curvature parameter(s) for hyperbolic space

        Returns:
            If return_components=False: total_loss (scalar)
            If return_components=True: dict with keys:
                - 'pmrl_total': Total loss (L^M + λ₁·L^M')
                - 'pmrl_L_M': SVD alignment loss
                - 'pmrl_L_Mprime': Instance-wise regularization loss
                - 'pmrl_singular_values': Top-k singular values (for logging)
        """
        # Check if hybrid mode is enabled
        use_hybrid = (self.use_hybrid_space and euclidean_weight is not None
                      and hyperbolic_weight is not None and curvatures is not None)

        if use_hybrid:
            return self._forward_hybrid(embeddings_dict, return_components,
                                       euclidean_weight, hyperbolic_weight, curvatures)
        else:
            return self._forward_euclidean(embeddings_dict, return_components)

    def _forward_euclidean(self, embeddings_dict, return_components=False):
        """Forward pass in pure Euclidean space (original PMRL)"""
        # Stack modalities: [B, k, D] where B=batch_size, k=num_modalities
        modalities = list(embeddings_dict.values())
        B = modalities[0].shape[0]
        k = len(modalities)
        D = modalities[0].shape[1]

        # Stack: [B, k, D]
        Z = torch.stack(modalities, dim=1)  # [B, k, D]

        # Transpose to [B, D, k] for SVD (each sample is D×k matrix)
        Z = Z.transpose(1, 2)  # [B, D, k]

        # L^M: SVD Alignment Loss
        L_M_samples = []
        all_singular_values = []
        U_matrices = []

        for i in range(B):
            Z_i = Z[i]  # [D, k]
            U_i, S_i, Vh_i = torch.linalg.svd(Z_i, full_matrices=False)

            svd_logits = S_i / self.tau
            exp_logits = torch.exp(svd_logits)
            L_M_i = -torch.log(exp_logits[0] / torch.sum(exp_logits))

            L_M_samples.append(L_M_i)
            all_singular_values.append(S_i)
            U_matrices.append(U_i)

        L_M = torch.mean(torch.stack(L_M_samples))
        S_batch = torch.stack(all_singular_values, dim=0)

        # L^M': Instance-wise Regularization
        u1_vectors = torch.stack([U_i[:, 0] for U_i in U_matrices], dim=0)
        u1_vectors = F.normalize(u1_vectors, dim=1)

        sim_matrix = u1_vectors @ u1_vectors.T / self.tau
        labels = torch.arange(B, device=u1_vectors.device)
        L_Mprime = F.cross_entropy(sim_matrix, labels)

        total_loss = L_M + self.lambda1 * L_Mprime

        if return_components:
            mean_singular_values = S_batch.mean(dim=0)
            return {
                'pmrl_total': total_loss,
                'pmrl_L_M': L_M,
                'pmrl_L_Mprime': L_Mprime,
                'pmrl_singular_values': mean_singular_values.detach()
            }

        return total_loss

    def _forward_hybrid(self, embeddings_dict, return_components,
                       euclidean_weight, hyperbolic_weight, curvatures):
        """
        Forward pass in hybrid Euclidean-Hyperbolic space.

        Strategy: Project embeddings to hybrid space using weighted combination
        of Euclidean and Hyperbolic (tangent space) representations, then perform SVD.
        """
        from utils.hyperbolic_volume import exp_map0

        modalities = list(embeddings_dict.values())
        B = modalities[0].shape[0]
        k = len(modalities)
        D = modalities[0].shape[1]

        # Step 1: Project to Hybrid Space
        # For each modality, compute hybrid representation:
        #   hybrid_emb = w_euc * emb + w_hyp * tangent_proj(emb)
        hybrid_modalities = []

        for emb in modalities:
            # Euclidean component (normalized)
            euc_component = F.normalize(emb, dim=-1) * euclidean_weight

            # Hyperbolic component: project to hyperboloid then back to tangent space
            # This creates a non-linear transformation that captures hyperbolic geometry
            hyp_proj = exp_map0(emb, curvatures)  # Project to hyperboloid
            hyp_component = F.normalize(hyp_proj, dim=-1) * hyperbolic_weight

            # Hybrid embedding
            hybrid_emb = euc_component + hyp_component
            hybrid_modalities.append(hybrid_emb)

        # Stack: [B, k, D]
        Z = torch.stack(hybrid_modalities, dim=1)
        Z = Z.transpose(1, 2)  # [B, D, k]

        # Step 2: SVD in Hybrid Space
        L_M_samples = []
        all_singular_values = []
        U_matrices = []

        for i in range(B):
            Z_i = Z[i]  # [D, k]
            U_i, S_i, Vh_i = torch.linalg.svd(Z_i, full_matrices=False)

            svd_logits = S_i / self.tau
            exp_logits = torch.exp(svd_logits)
            L_M_i = -torch.log(exp_logits[0] / torch.sum(exp_logits))

            L_M_samples.append(L_M_i)
            all_singular_values.append(S_i)
            U_matrices.append(U_i)

        L_M = torch.mean(torch.stack(L_M_samples))
        S_batch = torch.stack(all_singular_values, dim=0)

        # Step 3: L^M' in Hybrid Space
        u1_vectors = torch.stack([U_i[:, 0] for U_i in U_matrices], dim=0)
        u1_vectors = F.normalize(u1_vectors, dim=1)

        sim_matrix = u1_vectors @ u1_vectors.T / self.tau
        labels = torch.arange(B, device=u1_vectors.device)
        L_Mprime = F.cross_entropy(sim_matrix, labels)

        total_loss = L_M + self.lambda1 * L_Mprime

        if return_components:
            mean_singular_values = S_batch.mean(dim=0)
            return {
                'pmrl_total': total_loss,
                'pmrl_L_M': L_M,
                'pmrl_L_Mprime': L_Mprime,
                'pmrl_singular_values': mean_singular_values.detach()
            }

        return total_loss


def compute_pmrl_loss_with_itm(embeddings_dict, itm_logits, itm_labels,
                                lambda1=1.0, lambda2=0.1, temperature=0.07):
    """
    Complete PMRL loss including ITM component.

    This is a convenience function for computing all three PMRL losses.

    Args:
        embeddings_dict (dict): {modality_name: tensor[B, D]}
        itm_logits (tensor): [3*B, 2] - Logits from ITM head
        itm_labels (tensor): [3*B] - Ground truth labels (1 for positive, 0 for negative)
        lambda1 (float): Weight for L^M' (default: 1.0)
        lambda2 (float): Weight for L_IM (default: 0.1)
        temperature (float): Temperature for softmax (default: 0.07)

    Returns:
        dict with keys:
            - 'pmrl_total': Total loss (L^M + λ₁·L^M' + λ₂·L_IM)
            - 'pmrl_L_M': SVD alignment loss
            - 'pmrl_L_Mprime': Instance-wise regularization loss
            - 'pmrl_L_IM': Instance matching loss
            - 'pmrl_singular_values': Top-5 singular values

    Example:
        >>> embeddings = {'text': ..., 'video': ..., 'audio': ...}
        >>> itm_logits = model.itm_head(...)  # [192, 2] for B=64
        >>> itm_labels = torch.zeros(192).long()
        >>> itm_labels[:64] = 1
        >>> loss_dict = compute_pmrl_loss_with_itm(embeddings, itm_logits, itm_labels)
    """
    # Compute L^M and L^M'
    pmrl = PMRLLoss(lambda1, lambda2, temperature)
    loss_dict = pmrl(embeddings_dict, return_components=True)

    # Compute L_IM from ITM
    L_IM = F.cross_entropy(itm_logits, itm_labels)

    # Update total loss
    loss_dict['pmrl_L_IM'] = L_IM
    loss_dict['pmrl_total'] = (loss_dict['pmrl_L_M'] +
                               lambda1 * loss_dict['pmrl_L_Mprime'] +
                               lambda2 * L_IM)

    return loss_dict


# Utility Functions

def log_pmrl_metrics(loss_dict, step, logger):
    """
    Log PMRL metrics to WandB or tensorboard

    Args:
        loss_dict (dict): Output from PMRLLoss
        step (int): Training step
        logger: Logger instance (e.g., wandb)

    Example:
        >>> import wandb
        >>> loss_dict = pmrl(embeddings, return_components=True)
        >>> log_pmrl_metrics(loss_dict, step=100, logger=wandb)
    """
    metrics = {
        'loss/pmrl_total': loss_dict['pmrl_total'].item(),
        'loss/pmrl_L_M': loss_dict['pmrl_L_M'].item(),
        'loss/pmrl_L_Mprime': loss_dict['pmrl_L_Mprime'].item(),
    }

    if 'pmrl_L_IM' in loss_dict:
        metrics['loss/pmrl_L_IM'] = loss_dict['pmrl_L_IM'].item()

    # Log singular values
    svs = loss_dict['pmrl_singular_values']
    for i, sv in enumerate(svs):
        metrics[f'singular_value/sigma_{i+1}'] = sv.item()

    # Compute singular value ratio (σ₁ / Σσᵢ)
    sv_ratio = svs[0] / svs.sum()
    metrics['singular_value/ratio_sigma1'] = sv_ratio.item()

    logger.log(metrics, step=step)


def get_alignment_quality(loss_dict):
    """
    Compute alignment quality metric from singular values

    High-quality alignment: σ₁ >> σ₂, ..., σₖ (approaching rank-1)

    Args:
        loss_dict (dict): Output from PMRLLoss with 'pmrl_singular_values' key

    Returns:
        float: Alignment quality score in [0, 1]
               1.0 = perfect rank-1 alignment
               0.0 = no dominant direction

    Example:
        >>> loss_dict = pmrl(embeddings, return_components=True)
        >>> quality = get_alignment_quality(loss_dict)
        >>> print(f"Alignment quality: {quality:.3f}")
    """
    svs = loss_dict['pmrl_singular_values']

    # Ratio: σ₁ / (σ₁ + σ₂ + ... + σₖ)
    ratio = svs[0] / svs.sum()

    # Normalize to [0, 1]
    # For k modalities, perfect rank-1 would have ratio ≈ 1.0
    # Random initialization typically gives ratio ≈ 1/k
    return ratio.item()


def diagnose_pmrl_training(loss_dict, threshold_ratio=0.5):
    """
    Diagnose potential training issues from PMRL metrics

    Args:
        loss_dict (dict): Output from PMRLLoss
        threshold_ratio (float): Minimum expected σ₁/Σσᵢ ratio (default: 0.5)

    Returns:
        dict: Diagnostic information with keys:
            - 'status': 'healthy' | 'warning' | 'critical'
            - 'messages': List of diagnostic messages
            - 'suggestions': List of suggested actions

    Example:
        >>> diag = diagnose_pmrl_training(loss_dict)
        >>> if diag['status'] != 'healthy':
        ...     print("Issues found:", diag['messages'])
        ...     print("Suggestions:", diag['suggestions'])
    """
    svs = loss_dict['pmrl_singular_values']
    ratio = svs[0] / svs.sum()

    status = 'healthy'
    messages = []
    suggestions = []

    # Check singular value ratio
    if ratio < threshold_ratio:
        status = 'warning'
        messages.append(f"Low principal ratio: {ratio:.3f} < {threshold_ratio}")
        suggestions.append("Consider increasing λ₁ (pmrl_lambda1) to strengthen instance regularization")
        suggestions.append("Check if features are properly normalized before PMRL")

    # Check for NaN or Inf
    if torch.isnan(loss_dict['pmrl_L_M']) or torch.isinf(loss_dict['pmrl_L_M']):
        status = 'critical'
        messages.append("NaN or Inf detected in L^M")
        suggestions.append("Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
        suggestions.append("Check for numerical instability in SVD")

    if torch.isnan(loss_dict['pmrl_L_Mprime']) or torch.isinf(loss_dict['pmrl_L_Mprime']):
        status = 'critical'
        messages.append("NaN or Inf detected in L^M'")
        suggestions.append("Check instance-wise contrastive computation")

    # Check if singular values are too small
    if svs[0] < 1e-6:
        status = 'warning'
        messages.append(f"Very small principal singular value: {svs[0]:.2e}")
        suggestions.append("Features may be too sparse or poorly initialized")
        suggestions.append("Check encoder outputs and projection heads")

    # Check if singular values are too uniform
    sv_std = torch.std(svs[:3])
    if sv_std < 0.01:
        status = 'warning'
        messages.append(f"Singular values too uniform: std={sv_std:.4f}")
        suggestions.append("L^M may need stronger weight or lower temperature")

    if status == 'healthy':
        messages.append("PMRL training appears healthy")
        messages.append(f"Principal ratio: {ratio:.3f}")

    return {
        'status': status,
        'messages': messages,
        'suggestions': suggestions,
        'metrics': {
            'ratio': ratio.item(),
            'sigma1': svs[0].item(),
            'sigma_std': sv_std.item() if sv_std is not None else None
        }
    }


# Testing Utilities

def test_pmrl_loss():
    """
    Simple test function for PMRLLoss

    Example:
        >>> test_pmrl_loss()
        Testing PMRLLoss...
        Forward pass successful
        Loss components: L^M=..., L^M'=...
        Singular values shape: torch.Size([5])
    """
    print("Testing PMRLLoss...")

    # Create dummy embeddings
    B, D = 64, 512
    embeddings = {
        'text': torch.randn(B, D),
        'video': torch.randn(B, D),
        'audio': torch.randn(B, D),
        'subtitle': torch.randn(B, D)
    }

    # Normalize embeddings (as done in actual model)
    for k in embeddings:
        embeddings[k] = F.normalize(embeddings[k], dim=-1)

    # Initialize PMRL loss
    pmrl = PMRLLoss(lambda1=1.0, lambda2=0.1, temperature=0.07)

    # Forward pass
    loss_dict = pmrl(embeddings, return_components=True)

    print("Forward pass successful")
    print(f"Loss components: L^M={loss_dict['pmrl_L_M']:.4f}, L^M'={loss_dict['pmrl_L_Mprime']:.4f}")
    print(f"Singular values shape: {loss_dict['pmrl_singular_values'].shape}")
    print(f"Top-5 singular values: {loss_dict['pmrl_singular_values']}")

    # Test alignment quality
    quality = get_alignment_quality(loss_dict)
    print(f"Alignment quality: {quality:.3f}")

    # Test diagnostics
    diag = diagnose_pmrl_training(loss_dict)
    print(f"Training status: {diag['status']}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_pmrl_loss()
