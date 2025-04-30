import numpy as np
import os
import json

class ParamStorage:
    def __init__(self):
        self.storage = {
            'biases': {},
            'weights': {},  # Add weight storage
            'metadata': {}  # Store network architecture info
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
        # Convert numpy arrays to lists for metadata
        metadata = {k: {**v, 'shape': list(v['shape'])} 
                   for k, v in self.storage['metadata'].items()}
        
        np.savez(
            filename,
            biases=self.storage['biases'],
            weights=self.storage['weights'],
            metadata=json.dumps(metadata)  # Save metadata as JSON string
        )
    
    def load(self, filename='network_state.npz'):
        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            self.storage['biases'] = dict(enumerate(data['biases'])) if data['biases'].size > 0 else {}
            self.storage['weights'] = dict(enumerate(data['weights'])) if data['weights'].size > 0 else {}
            self.storage['metadata'] = json.loads(str(data['metadata']))
            if 'metadata' in data:
                metadata_str = str(data['metadata']) if isinstance(data['metadata'], np.ndarray) else data['metadata']
                self.storage['metadata'] = json.loads(metadata_str)
                for k, v in self.storage['metadata'].items():
                    if 'shape' in v:
                        v['shape'] = tuple(v['shape'])
            if self.storage['biases']:
                self.next_id = max(map(int, self.storage['biases'].keys())) + 1
            else:
                self.next_id = 0

            return True
        return False

param_storage = ParamStorage()