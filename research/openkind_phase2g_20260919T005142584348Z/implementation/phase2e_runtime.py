"""Index-safe audits and fatal CUDA handling; no downloads or GPU work on import."""
from __future__ import annotations
from collections.abc import Mapping
from numbers import Integral


class FatalCUDAError(RuntimeError):
    """The CUDA context is unusable; only CPU-side reporting may continue."""


def is_fatal_cuda(error):
    """Follow exception chains, but do not confuse ordinary CUDA OOM with a poisoned context."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, FatalCUDAError):
            return True
        message = str(error).lower()
        fatal = ('device-side assert', 'device side assert', 'cudaerrorassert',
                 'cuda_error_assert', 'illegal memory access', 'misaligned address',
                 'unspecified launch failure', 'cudaerrorlaunchfailure')
        if any(s in message for s in fatal):
            return True
        error = error.__cause__ or error.__context__
    return False


def raise_if_fatal_cuda(error):
    if isinstance(error, FatalCUDAError):
        raise error
    if is_fatal_cuda(error):
        raise FatalCUDAError(
            'Fatal CUDA context error. GPU work is stopped; restart the Colab session '
            'before running again. empty_cache() cannot repair this context. '
            'The chained exception preserves the original failure.'
        ) from error


def integer_sample_indices(numel, samples=17):
    """Inclusive spaced indices computed with Python integers, never floating point.

    float32 linspace rounds 635699199 to 635699200 for Qwen's embedding;
    casting that value to long does NOT repair the out-of-bounds index.
    """
    if isinstance(numel, bool) or not isinstance(numel, Integral) or numel < 0:
        raise ValueError('numel must be a nonnegative integer')
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 1:
        raise ValueError('samples must be a positive integer')
    n, k = int(numel), min(int(numel), int(samples))
    if k == 0:
        return []
    if k == 1:
        return [0]
    indices = [(i * (n - 1)) // (k - 1) for i in range(k)]
    if indices[0] != 0 or indices[-1] != n - 1 or len(set(indices)) != k:
        raise AssertionError('Invalid parameter-sampling index construction')
    return indices


def indexing_selftest():
    sizes = (0, 1, 16, 17, 18, 2**24 + 1, 2**24 + 2,
             248320 * 2560, 2**53 + 123, 2**63 - 1)
    for n in sizes:
        ix = integer_sample_indices(n)
        assert len(ix) == min(n, 17) and all(0 <= i < n for i in ix)
        if n > 1:
            assert ix[0] == 0 and ix[-1] == n - 1
    return {'status': 'passed', 'sizes_checked': len(sizes),
            'qwen_embedding_numel': 248320 * 2560,
            'qwen_last_valid_sample_index': integer_sample_indices(248320 * 2560)[-1],
            'method': 'Python integer arithmetic -> torch.long; no floating-point indices',
            'scope': 'CPU index contract; does not validate Qwen/CUDA inference'}


def aggregate_outcomes(outcomes):
    """Resolve stage status from actual child outcomes, not from absence of an exception."""
    items = list(outcomes.values()) if isinstance(outcomes, Mapping) else list(outcomes)
    statuses = [x.get('status', 'failed') for x in items if isinstance(x, Mapping)]
    if not statuses:
        return 'skipped'
    good = sum(s in ('completed', 'passed') for s in statuses)
    partial = any(s == 'partial' for s in statuses)
    failed = any(s in ('failed', 'aborted', 'running', 'blocked_after_fatal_cuda') for s in statuses)
    if good == len(statuses):
        return 'completed'
    if good or partial:
        return 'partial'
    return 'failed' if failed else 'skipped'


def validate_token_rows(rows, vocab_size, pad_id=None):
    """CPU-only bounds check before an embedding kernel. Never clips or rewrites tokens."""
    if not isinstance(vocab_size, Integral) or vocab_size <= 0:
        raise ValueError('Invalid embedding vocabulary size')
    if pad_id is not None and (isinstance(pad_id, bool) or not isinstance(pad_id, Integral)
                               or not 0 <= pad_id < vocab_size):
        raise ValueError(f'pad_token_id must be in [0, {vocab_size})')
    count = 0
    for row_index, row in enumerate(rows):
        seq = row['input_ids'] if isinstance(row, Mapping) else row
        if not len(seq):
            raise ValueError(f'Empty input sequence at row {row_index}')
        for j, token_id in enumerate(seq):
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise ValueError(f'Non-integer token at row {row_index}, offset {j}')
            if not 0 <= token_id < vocab_size:
                raise ValueError(f'Token {int(token_id)} outside [0, {vocab_size}) '
                                 f'at row {row_index}, offset {j}')
        count += 1
    return count
