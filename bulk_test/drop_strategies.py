"""
Drop strategies for MemoryLLM bulk testing.

Each strategy determines which memory tokens to DROP when new memories come in.
The function returns indices to KEEP (not drop).
"""

import torch
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class MemoryTracker:
    """Tracks metadata for smart dropping strategies."""
    num_layers: int
    num_tokens: int
    device: str = "cuda"

    # Tracking tensors (initialized lazily)
    access_counts: Optional[torch.Tensor] = field(default=None, repr=False)
    last_access: Optional[torch.Tensor] = field(default=None, repr=False)
    insertion_order: Optional[torch.Tensor] = field(default=None, repr=False)
    step: int = 0

    def __post_init__(self):
        self.access_counts = torch.zeros(self.num_layers, self.num_tokens, device=self.device)
        self.last_access = torch.zeros(self.num_layers, self.num_tokens, device=self.device)
        self.insertion_order = torch.arange(self.num_tokens, device=self.device).unsqueeze(0).expand(self.num_layers, -1).clone()

    def record_access(self, layer_idx: int, token_indices: torch.Tensor):
        """Record that tokens were accessed during attention."""
        self.access_counts[layer_idx, token_indices] += 1
        self.last_access[layer_idx, token_indices] = self.step

    def record_insertion(self, layer_idx: int, new_indices: torch.Tensor, kept_indices: torch.Tensor):
        """Update tracking after memory update."""
        # Shift insertion order for kept tokens
        self.insertion_order[layer_idx, :len(kept_indices)] = self.insertion_order[layer_idx, kept_indices]
        # New tokens get current step as insertion order
        self.insertion_order[layer_idx, len(kept_indices):] = self.step

        # Reset stats for new tokens
        new_start = len(kept_indices)
        self.access_counts[layer_idx, new_start:] = 0
        self.last_access[layer_idx, new_start:] = self.step

    def tick(self):
        """Advance time step."""
        self.step += 1

    def to(self, device: str):
        """Move tracking tensors to device."""
        self.device = device
        if self.access_counts is not None:
            self.access_counts = self.access_counts.to(device)
            self.last_access = self.last_access.to(device)
            self.insertion_order = self.insertion_order.to(device)
        return self


# ============================================================================
# TIER 1: Basic Strategies
# ============================================================================

def random_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Baseline: Random drop (current MemoryLLM behavior)."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    indices = torch.randperm(num_tokens, device=memory.device)[:keep_length]
    return indices.sort()[0]


def fifo_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """First-In-First-Out: Drop oldest memories (keep newest)."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    drop_length = num_tokens - keep_length
    # Keep the last keep_length tokens (newest)
    indices = torch.arange(drop_length, num_tokens, device=memory.device)
    return indices


def lifo_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Last-In-First-Out: Drop newest memories (keep oldest) - control."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    # Keep the first keep_length tokens (oldest)
    indices = torch.arange(0, keep_length, device=memory.device)
    return indices


def lru_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
             layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """Least Recently Used: Drop tokens not accessed recently."""
    if tracker is None:
        return random_drop(memory, keep_length)

    # Get indices of most recently accessed tokens
    _, indices = tracker.last_access[layer_idx].topk(keep_length)
    return indices.sort()[0]


def lfu_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
             layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """Least Frequently Used: Drop tokens accessed least often."""
    if tracker is None:
        return random_drop(memory, keep_length)

    # Get indices of most frequently accessed tokens
    _, indices = tracker.access_counts[layer_idx].topk(keep_length)
    return indices.sort()[0]


def mru_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
             layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """Most Recently Used: Drop recently accessed (control experiment)."""
    if tracker is None:
        return random_drop(memory, keep_length)

    # Get indices of LEAST recently accessed (opposite of LRU)
    _, indices = (-tracker.last_access[layer_idx]).topk(keep_length)
    return indices.sort()[0]


def round_robin_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                     layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """Round Robin: Drop in rotating fashion based on step."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    drop_length = num_tokens - keep_length

    step = tracker.step if tracker else 0
    start = (step * drop_length) % num_tokens

    # Create mask of indices to keep
    all_indices = torch.arange(num_tokens, device=memory.device)
    drop_indices = (all_indices >= start) & (all_indices < start + drop_length)
    keep_indices = all_indices[~drop_indices]

    if len(keep_indices) < keep_length:
        # Wrap around
        extra = keep_length - len(keep_indices)
        keep_indices = torch.cat([keep_indices, all_indices[:extra]])

    return keep_indices[:keep_length].sort()[0]


# ============================================================================
# TIER 2: Importance-Based Strategies
# ============================================================================

def norm_low_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Drop low-norm embeddings (keep high-norm as more important)."""
    if memory.dim() == 2:
        norms = memory.norm(dim=-1)
    else:
        norms = memory.norm(dim=-1).mean(dim=0)  # Average across batch

    _, indices = norms.topk(keep_length)
    return indices.sort()[0]


def norm_high_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Drop high-norm embeddings (control experiment)."""
    if memory.dim() == 2:
        norms = memory.norm(dim=-1)
    else:
        norms = memory.norm(dim=-1).mean(dim=0)

    _, indices = (-norms).topk(keep_length)
    return indices.sort()[0]


def variance_low_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Drop low-variance embeddings (keep distinctive ones)."""
    if memory.dim() == 2:
        var = memory.var(dim=-1)
    else:
        var = memory.var(dim=-1).mean(dim=0)

    _, indices = var.topk(keep_length)
    return indices.sort()[0]


def entropy_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Drop low-entropy embeddings (keep uncertain/diverse ones)."""
    if memory.dim() == 2:
        # Approximate entropy via softmax entropy
        probs = torch.softmax(memory, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    else:
        probs = torch.softmax(memory, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean(dim=0)

    _, indices = entropy.topk(keep_length)
    return indices.sort()[0]


def cosine_similar_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Drop most similar to others (remove redundancy)."""
    if memory.dim() == 3:
        mem = memory.mean(dim=0)  # [num_tokens, hidden]
    else:
        mem = memory

    # Normalize for cosine similarity
    mem_norm = torch.nn.functional.normalize(mem, dim=-1)

    # Compute pairwise similarity
    sim = torch.mm(mem_norm, mem_norm.t())
    sim.fill_diagonal_(0)  # Don't count self-similarity

    # Total similarity score (how redundant is each token)
    redundancy = sim.sum(dim=1)

    # Keep LEAST redundant (most unique)
    _, indices = (-redundancy).topk(keep_length)
    return indices.sort()[0]


def cosine_dissimilar_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Drop most dissimilar (control - keep redundant)."""
    if memory.dim() == 3:
        mem = memory.mean(dim=0)
    else:
        mem = memory

    mem_norm = torch.nn.functional.normalize(mem, dim=-1)
    sim = torch.mm(mem_norm, mem_norm.t())
    sim.fill_diagonal_(0)
    redundancy = sim.sum(dim=1)

    # Keep MOST redundant
    _, indices = redundancy.topk(keep_length)
    return indices.sort()[0]


# ============================================================================
# TIER 3: Hybrid Strategies
# ============================================================================

def hybrid_lru_random(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                      layer_idx: int = 0, ratio: float = 0.7, **kwargs) -> torch.Tensor:
    """70% LRU + 30% random."""
    lru_keep = int(keep_length * ratio)
    random_keep = keep_length - lru_keep

    if tracker is None:
        return random_drop(memory, keep_length)

    # Get LRU indices
    _, lru_indices = tracker.last_access[layer_idx].topk(lru_keep)

    # Get random from remaining
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    mask = torch.ones(num_tokens, dtype=torch.bool, device=memory.device)
    mask[lru_indices] = False
    remaining = torch.where(mask)[0]

    if len(remaining) >= random_keep:
        random_indices = remaining[torch.randperm(len(remaining), device=memory.device)[:random_keep]]
    else:
        random_indices = remaining

    indices = torch.cat([lru_indices, random_indices])
    return indices.sort()[0]


def hybrid_fifo_importance(memory: torch.Tensor, keep_length: int, ratio: float = 0.5, **kwargs) -> torch.Tensor:
    """50% FIFO + 50% importance (norm-based)."""
    fifo_keep = int(keep_length * ratio)
    importance_keep = keep_length - fifo_keep

    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]

    # FIFO: keep newest
    fifo_indices = torch.arange(num_tokens - fifo_keep, num_tokens, device=memory.device)

    # Importance from older tokens
    older_memory = memory[:num_tokens - fifo_keep] if memory.dim() == 2 else memory[:, :num_tokens - fifo_keep]
    if older_memory.dim() == 2:
        norms = older_memory.norm(dim=-1)
    else:
        norms = older_memory.norm(dim=-1).mean(dim=0)

    _, importance_indices = norms.topk(min(importance_keep, len(norms)))

    indices = torch.cat([importance_indices, fifo_indices])
    return indices.sort()[0]


def tiered_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """Tiered: Recent=random, Old=LRU."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    mid = num_tokens // 2

    # Keep 60% from recent half (random), 40% from old half (LRU)
    recent_keep = int(keep_length * 0.6)
    old_keep = keep_length - recent_keep

    # Recent half: random
    recent_indices = torch.randperm(mid, device=memory.device)[:recent_keep] + mid

    # Old half: LRU if tracker available, else random
    if tracker is not None:
        old_access = tracker.last_access[layer_idx, :mid]
        _, old_indices = old_access.topk(min(old_keep, mid))
    else:
        old_indices = torch.randperm(mid, device=memory.device)[:old_keep]

    indices = torch.cat([old_indices, recent_indices])
    return indices.sort()[0]


def diversity_preserving_drop(memory: torch.Tensor, keep_length: int, **kwargs) -> torch.Tensor:
    """Keep diverse set using greedy farthest-point sampling."""
    if memory.dim() == 3:
        mem = memory.mean(dim=0)
    else:
        mem = memory

    num_tokens = mem.shape[0]
    device = mem.device

    # Normalize
    mem_norm = torch.nn.functional.normalize(mem, dim=-1)

    # Greedy farthest point sampling
    selected = [torch.randint(num_tokens, (1,), device=device).item()]

    for _ in range(keep_length - 1):
        # Compute min distance to selected set for each point
        selected_mem = mem_norm[selected]
        similarities = torch.mm(mem_norm, selected_mem.t())  # [num_tokens, num_selected]
        max_sim_to_selected = similarities.max(dim=1)[0]  # Most similar to any selected

        # Mask already selected
        max_sim_to_selected[selected] = float('inf')

        # Select point with lowest max similarity (most different from all selected)
        next_idx = max_sim_to_selected.argmin().item()
        selected.append(next_idx)

    indices = torch.tensor(selected, device=device)
    return indices.sort()[0]


def probabilistic_lru(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                      layer_idx: int = 0, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """Soft LRU: Sample proportional to recency."""
    if tracker is None:
        return random_drop(memory, keep_length)

    # Convert access times to probabilities
    recency = tracker.last_access[layer_idx]
    probs = torch.softmax(recency / temperature, dim=0)

    # Sample without replacement
    indices = torch.multinomial(probs, keep_length, replacement=False)
    return indices.sort()[0]


def adaptive_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                  layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """Adaptive: Switch strategy based on memory state."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]

    if tracker is None or tracker.step < 10:
        # Early: use FIFO
        return fifo_drop(memory, keep_length)

    # Check if access patterns are uniform
    access_var = tracker.access_counts[layer_idx].var()

    if access_var < 1.0:
        # Uniform access: use FIFO
        return fifo_drop(memory, keep_length)
    else:
        # Non-uniform: use LFU
        return lfu_drop(memory, keep_length, tracker, layer_idx)


# ============================================================================
# TIER 4: Experimental Strategies
# ============================================================================

def sliding_window_landmarks(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                             layer_idx: int = 0, landmark_ratio: float = 0.2, **kwargs) -> torch.Tensor:
    """Keep sliding window of recent + landmark old memories."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]

    # Keep landmarks (important old ones)
    num_landmarks = int(keep_length * landmark_ratio)
    window_size = keep_length - num_landmarks

    # Landmarks: highest norm from first half
    old_half = num_tokens // 2
    if memory.dim() == 2:
        old_norms = memory[:old_half].norm(dim=-1)
    else:
        old_norms = memory[:, :old_half].norm(dim=-1).mean(dim=0)

    _, landmark_indices = old_norms.topk(min(num_landmarks, old_half))

    # Window: most recent
    window_start = max(old_half, num_tokens - window_size)
    window_indices = torch.arange(window_start, num_tokens, device=memory.device)

    indices = torch.cat([landmark_indices, window_indices])

    # If not enough, fill with random from middle
    if len(indices) < keep_length:
        remaining = keep_length - len(indices)
        mask = torch.ones(num_tokens, dtype=torch.bool, device=memory.device)
        mask[indices] = False
        available = torch.where(mask)[0]
        extra = available[torch.randperm(len(available), device=memory.device)[:remaining]]
        indices = torch.cat([indices, extra])

    return indices[:keep_length].sort()[0]


def exponential_decay_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                           layer_idx: int = 0, decay_rate: float = 0.95, **kwargs) -> torch.Tensor:
    """Exponential decay: Older memories have exponentially lower keep probability."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    device = memory.device

    # Create decay weights (newer = higher weight)
    positions = torch.arange(num_tokens, device=device, dtype=torch.float)
    weights = decay_rate ** (num_tokens - 1 - positions)

    # Sample based on weights
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, keep_length, replacement=False)
    return indices.sort()[0]


def temperature_based_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                           layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """Temperature: Hot (recently/frequently accessed) vs Cold (stale)."""
    if tracker is None:
        return random_drop(memory, keep_length)

    # Compute "temperature" = recency * frequency
    recency = tracker.last_access[layer_idx]
    frequency = tracker.access_counts[layer_idx]

    # Normalize both
    recency_norm = (recency - recency.min()) / (recency.max() - recency.min() + 1e-8)
    freq_norm = (frequency - frequency.min()) / (frequency.max() - frequency.min() + 1e-8)

    temperature = 0.5 * recency_norm + 0.5 * freq_norm

    # Keep hottest
    _, indices = temperature.topk(keep_length)
    return indices.sort()[0]


def two_tier_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                  layer_idx: int = 0, hot_ratio: float = 0.3, **kwargs) -> torch.Tensor:
    """Two-tier: Fast drop tier (recent) + slow archive tier (important old)."""
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]

    hot_size = int(keep_length * hot_ratio)
    archive_size = keep_length - hot_size

    # Hot tier: most recent
    hot_indices = torch.arange(num_tokens - hot_size, num_tokens, device=memory.device)

    # Archive tier: highest importance from older memories
    older = num_tokens - hot_size
    if memory.dim() == 2:
        old_importance = memory[:older].norm(dim=-1)
    else:
        old_importance = memory[:, :older].norm(dim=-1).mean(dim=0)

    if tracker is not None:
        # Boost importance by access frequency
        old_importance = old_importance * (1 + tracker.access_counts[layer_idx, :older])

    _, archive_indices = old_importance.topk(min(archive_size, older))

    indices = torch.cat([archive_indices, hot_indices])
    return indices[:keep_length].sort()[0]


def layer_aware_drop(memory: torch.Tensor, keep_length: int, tracker: MemoryTracker = None,
                     layer_idx: int = 0, num_layers: int = 32, **kwargs) -> torch.Tensor:
    """Different strategy for different layers."""
    # Early layers: FIFO (syntactic)
    # Middle layers: LRU (semantic)
    # Late layers: Importance (task-specific)

    if layer_idx < num_layers // 3:
        return fifo_drop(memory, keep_length)
    elif layer_idx < 2 * num_layers // 3:
        return lru_drop(memory, keep_length, tracker, layer_idx)
    else:
        return norm_low_drop(memory, keep_length)


def attention_score_drop(memory: torch.Tensor, keep_length: int, attention_scores: torch.Tensor = None,
                         tracker: MemoryTracker = None, layer_idx: int = 0, **kwargs) -> torch.Tensor:
    """
    Smart importance-based drop using available signals.

    Since capturing raw attention requires ~13GB of memory (too expensive),
    we use a hybrid approach combining:
    1. Embedding norms (higher norm = more information content)
    2. Recency from tracker (recently used = more relevant)
    3. Access frequency from tracker (frequently accessed = important)

    This approximates attention-based importance without the memory cost.
    """
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]

    # If we have actual attention scores, use them
    if attention_scores is not None:
        try:
            if attention_scores.dim() == 1:
                avg_attention = attention_scores
            elif attention_scores.dim() == 3:
                avg_attention = attention_scores.mean(dim=(0, 1))
            elif attention_scores.dim() == 4:
                avg_attention = attention_scores.mean(dim=(0, 1, 2))
            else:
                avg_attention = None

            if avg_attention is not None and avg_attention.shape[0] >= keep_length:
                if avg_attention.shape[0] > num_tokens:
                    avg_attention = avg_attention[:num_tokens]
                _, indices = avg_attention.topk(keep_length)
                return indices.sort()[0]
        except Exception:
            pass

    # Hybrid importance scoring (memory-efficient alternative)
    mem = memory if memory.dim() == 2 else memory.squeeze(0)
    device = mem.device

    # Component 1: Norm-based importance (higher norm = more content)
    norms = torch.norm(mem, dim=-1)
    norm_scores = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)

    # Component 2: Recency score (newer = higher score)
    if tracker is not None:
        timestamps = tracker.get_timestamps(layer_idx)
        if timestamps is not None and len(timestamps) == num_tokens:
            timestamps = timestamps.to(device).float()
            recency_scores = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        else:
            recency_scores = torch.zeros(num_tokens, device=device)
    else:
        # Without tracker, assume uniform recency
        recency_scores = torch.zeros(num_tokens, device=device)

    # Component 3: Access frequency (more accesses = more important)
    if tracker is not None:
        access_counts = tracker.get_access_counts(layer_idx)
        if access_counts is not None and len(access_counts) == num_tokens:
            access_counts = access_counts.to(device).float()
            access_scores = (access_counts - access_counts.min()) / (access_counts.max() - access_counts.min() + 1e-8)
        else:
            access_scores = torch.zeros(num_tokens, device=device)
    else:
        access_scores = torch.zeros(num_tokens, device=device)

    # Weighted combination: norm matters most, then recency, then access
    importance = 0.5 * norm_scores + 0.3 * recency_scores + 0.2 * access_scores

    # Keep highest importance tokens
    _, indices = importance.topk(keep_length)
    return indices.sort()[0]


# ============================================================================
# Strategy Registry
# ============================================================================

STRATEGIES = {
    # Tier 1: Basic
    "random": random_drop,
    "fifo": fifo_drop,
    "lifo": lifo_drop,
    "lru": lru_drop,
    "lfu": lfu_drop,
    "mru": mru_drop,
    "round_robin": round_robin_drop,

    # Tier 2: Importance-based
    "norm_low": norm_low_drop,
    "norm_high": norm_high_drop,
    "variance_low": variance_low_drop,
    "entropy": entropy_drop,
    "cosine_similar": cosine_similar_drop,
    "cosine_dissimilar": cosine_dissimilar_drop,

    # Tier 3: Hybrid
    "hybrid_lru_random": hybrid_lru_random,
    "hybrid_fifo_importance": hybrid_fifo_importance,
    "tiered": tiered_drop,
    "diversity": diversity_preserving_drop,
    "probabilistic_lru": probabilistic_lru,
    "adaptive": adaptive_drop,

    # Tier 4: Experimental
    "sliding_landmarks": sliding_window_landmarks,
    "exponential_decay": exponential_decay_drop,
    "temperature": temperature_based_drop,
    "two_tier": two_tier_drop,
    "layer_aware": layer_aware_drop,
    "attention_score": attention_score_drop,
}


def get_drop_indices(
    memory: torch.Tensor,
    drop_length: int,
    strategy: str = "random",
    tracker: MemoryTracker = None,
    layer_idx: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    Main entry point for drop strategies.

    Args:
        memory: Memory tensor [num_tokens, hidden] or [batch, num_tokens, hidden]
        drop_length: Number of tokens to drop
        strategy: Name of drop strategy
        tracker: MemoryTracker for stateful strategies
        layer_idx: Current layer index
        **kwargs: Additional strategy-specific arguments

    Returns:
        Indices to KEEP (sorted)
    """
    num_tokens = memory.shape[0] if memory.dim() == 2 else memory.shape[1]
    keep_length = num_tokens - drop_length

    if strategy not in STRATEGIES:
        print(f"Warning: Unknown strategy '{strategy}', falling back to random")
        strategy = "random"

    strategy_fn = STRATEGIES[strategy]

    return strategy_fn(
        memory=memory,
        keep_length=keep_length,
        tracker=tracker,
        layer_idx=layer_idx,
        **kwargs
    )


def list_strategies() -> List[str]:
    """Return list of available strategy names."""
    return list(STRATEGIES.keys())
