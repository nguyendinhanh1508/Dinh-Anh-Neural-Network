import numpy as np
import os
import json

class ParamStorage:
    def __init__(self):
        self.storage = {
            'biases': {},
            'weights': {},
            'metadata': {}
        }
        self.next_id = 0
    
    def create_entry(self, shape, layer_type, layer_params):
        entry_id = self.next_id
        self.next_id += 1
        self.storage['biases'][entry_id] = np.random.randn(*shape) * 0.01
        self.storage['metadata'][entry_id] = {
            'shape': shape,
            'type': layer_type,
            'params': layer_params
        }
        return entry_id
    
    def get_bias(self, bias_id):
        return self.storage['biases'][bias_id]
    
    def update_bias(self, bias_id, update):
        self.storage['biases'][bias_id] -= update
    
    def save_weights(self, layer_id, weights):
        self.storage['weights'][layer_id] = weights
    
    def get_weights(self, layer_id):
        return self.storage['weights'][layer_id]
    
    def save(self, filename='network_state.npz'):
        serializable_metadata = {}
        for k, v in self.storage['metadata'].items():
            serializable_metadata[k] = {
                'shape': list(v['shape']),
                'type': v['type'],
                'params': v['params']
            }
        np.savez(
            filename,
            biases=self.storage['biases'],
            weights=self.storage['weights'],
            metadata=json.dumps(serializable_metadata)
        )
    
    def load(self, filename='network_state.npz'):
        if os.path.exists(filename):
            try:
                data = np.load(filename, allow_pickle=True)
                self.storage['biases'] = {int(k): v for k, v in data['biases'].item().items()}
                self.storage['weights'] = {int(k): v for k, v in data['weights'].item().items()}
                metadata_str = data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
                self.storage['metadata'] = json.loads(metadata_str)
                for k, v in self.storage['metadata'].items():
                    if 'shape' in v:
                        v['shape'] = tuple(v['shape'])
                if self.storage['biases']:
                    self.next_id = max(map(int, self.storage['biases'].keys())) + 1
                else:
                    self.next_id = 0
                return True
            except Exception as e:
                print(f"Error loading saved state: {e}")
                return False
        return False

param_storage = ParamStorage()