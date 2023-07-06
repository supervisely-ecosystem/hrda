from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
import numpy as np


@DATASETS.register_module()
class SuperviselyDataset(CustomDataset):
    CLASSES = ['bg', 'cracks']
    PALETTE = [[0,0,0], [0,255,0]]

    def __init__(self, crop_pseudo_margins=None, valid_mask_size=(512,512), **kwargs):
        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')
        super().__init__(**kwargs)
        
        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = list(valid_mask_size)

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        if self.pseudo_margins is not None:
            results['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.pseudo_margins[0] > 0:
                results['valid_pseudo_mask'][:self.pseudo_margins[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[1] > 0:
                results['valid_pseudo_mask'][-self.pseudo_margins[1]:, :] = 0
            if self.pseudo_margins[2] > 0:
                results['valid_pseudo_mask'][:, :self.pseudo_margins[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[3] > 0:
                results['valid_pseudo_mask'][:, -self.pseudo_margins[3]:] = 0
            results['seg_fields'].append('valid_pseudo_mask')
