"""COD sampling logic for P-EAGLE parallel group generation."""

import torch


def _spans_one_document(
    document_ids: torch.Tensor, target_idx: torch.Tensor, depth: int
) -> torch.Tensor:
    """Mask of chains whose anchor and target sit in the same real document.

    A chain supervised at ``target_idx`` is anchored ``depth`` positions back, so
    it conditions on everything in between. Documents are packed contiguously and
    padding is marked ``-1``, so matching endpoints means the whole chain stays
    inside one real document; a chain that straddles a boundary would condition
    on another document's tokens.
    """
    doc_ids = document_ids.reshape(-1)
    target_doc = doc_ids[target_idx]
    anchor_doc = doc_ids[target_idx - depth]
    return (target_doc == anchor_doc) & (target_doc != -1)


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.7,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
    max_anchors: int | None = None,
    document_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sampling indices for parallel sequences using COD sampling.

    Conditional-On-Distribution (COD) sampling reduces memory by using geometric
    decay: depth 0 retains all n positions, depth 1 retains n*r positions,
    depth 2 retains n*r^2 positions, etc.

    Args:
        seq_length: Length of the sequence
        loss_mask: Binary mask indicating valid training positions [batch, seq_len]
        num_depths: Number of parallel prediction groups (K)
        down_sample_ratio: Geometric decay ratio r in (0,1)
        down_sample_ratio_min: Minimum retention ratio floor
        filter_position_zero: Whether to filter out position 0 from candidates
        max_anchors: Cap on the chains carried past depth 0
        document_ids: Maps each position to its document index, -1 for padding
            [seq_len] or [1, seq_len]. When given, a chain is kept only if its
            anchor and its supervised position lie in the same real document.

    Returns:
        Tuple of:
            anchor_pos: The starting position in the original sequence the current
                sampling chain started from.
            depth: Which COD sampling round each element belongs to
    """
    loss_mask = loss_mask.squeeze(0)
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]

    if max_anchors is not None and all_valid_indices.shape[0] > max_anchors:
        perm = torch.randperm(all_valid_indices.shape[0], device=device)[:max_anchors]
        seed_indices = all_valid_indices[perm].sort()[0]
    else:
        seed_indices = all_valid_indices

    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]
    prev_indices = seed_indices

    for d in range(1, num_depths):
        valid_length = max(0, seed_indices.shape[0] - d)
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)

        if sample_size <= 0:
            break

        # Subsample from candidate pool, or keep all if pool is too small
        if prev_indices.shape[0] >= sample_size:
            random_selection = torch.randperm(prev_indices.shape[0], device=device)[
                :sample_size
            ]
            sampled_idx = prev_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]  # restore causal order
        else:
            sampled_idx = prev_indices

        # A depth-d chain is anchored d positions back, so a supervised index
        # below d would store a negative anchor. The attention mask indexes
        # document_ids with it, and a negative index silently wraps to the end
        # of the sequence -- i.e. to another document.
        sampled_idx = sampled_idx[sampled_idx >= d]
        if document_ids is not None:
            sampled_idx = sampled_idx[_spans_one_document(document_ids, sampled_idx, d)]
        if sampled_idx.numel() == 0:
            break

        # Build candidate pool for next depth: shift by +1 (next-token targets).
        # No modulo: wrapping past the end would restart the chain at position 0,
        # which in a packed batch belongs to a different document.
        next_candidates = sampled_idx + 1
        next_candidates = next_candidates[next_candidates < seq_length]
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        sample_indices.append(sampled_idx - d)
        n_per_depth.append(sampled_idx.shape[0])

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat(
        [
            torch.full((n,), i, device=device, dtype=torch.long)
            for i, n in enumerate(n_per_depth)
        ]
    )

    return anchor_pos, depth
