import torchvision
import torch
import numpy as np

import logging
from base_component import BaseComponent

logging.basicConfig(level=logging.INFO)


class Component(BaseComponent):

    def __init__(self, config):
        super().__init__(config)

        self.weights = torchvision.models.efficientnet.EfficientNet_B1_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b7(
            weights=self.weights
        ).cuda()
        self.model.eval()
        self.preprocess = self.weights.transforms()
        logging.info('model loaded.')


    def process(self, frame, tracks):
        # tracks is a (n, 5) array containing l, t, r, b and track_id
        chips = []
        frame_h, frame_w, _ = frame.shape
        for (l, t, r, b, _) in tracks:
            l = max(l, 0)
            t = max(t, 0)
            r = min(r, frame_w)
            b = min(b, frame_h)
            chip = frame[t:b, l:r]
            # move axis from (H, W, C) to (C, H, W)
            chip = np.moveaxis(chip, 2, 0)
            chips.append(self.preprocess(torch.from_numpy(chip).cuda()))

        if not chips:
            return np.array([])

        chips = torch.stack(chips)
        with torch.no_grad():
            prediction = self.model(chips).softmax(1)
        class_ids = prediction.argmax(1).tolist()

        result = [[self.weights.meta['categories'][class_id], str(prediction[i, class_id].item())] for i, class_id in enumerate(class_ids)]
        logging.info(f'Classifier results: {result}')
        result = np.array(result, dtype='U32')
        return result
