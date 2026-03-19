import torch


class SecureAggregator:
    """Interface-compatible placeholder for secure aggregation.

    The current implementation returns the exact plaintext sum of client
    statistics while keeping the aggregation boundary isolated in one file.
    """

    def __init__(self, num_clusters, feature_dim, mode='plaintext'):
        if mode != 'plaintext':
            raise ValueError(f'Unsupported secure aggregation mode: {mode}')
        self.mode = mode
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self._global_sum = torch.zeros(num_clusters, feature_dim, dtype=torch.float32)
        self._global_count = torch.zeros(num_clusters, dtype=torch.float32)
        self._num_clients = 0

    def collect(self, client_id, packed_stats):
        local_sum = packed_stats['sum'].detach().cpu().float()
        local_count = packed_stats['count'].detach().cpu().float()
        if local_sum.shape != self._global_sum.shape:
            raise ValueError(
                f'Unexpected local sum shape {tuple(local_sum.shape)}, expected {tuple(self._global_sum.shape)}'
            )
        if local_count.shape != self._global_count.shape:
            raise ValueError(
                f'Unexpected local count shape {tuple(local_count.shape)}, expected {tuple(self._global_count.shape)}'
            )
        self._global_sum += local_sum
        self._global_count += local_count
        self._num_clients += 1

    def finalize(self):
        return {
            'sum': self._global_sum.clone(),
            'count': self._global_count.clone(),
            'num_clients': self._num_clients,
        }
