import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import C2f, Conv
from ultralytics.nn.simam import SimAM  # Import SimAM

class YOLOv8SimAM(DetectionModel):
    def __init__(self, cfg='yolo11m-pose.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # Modify backbone to add SimAM after C2f modules
        for i, layer in enumerate(self.model):
            if isinstance(layer, C2f):
                # Add SimAM after C2f
                self.model[i] = nn.Sequential(layer, SimAM())
        # Optionally, add SimAM in the head (e.g., after Conv layers)
        # Example: Add after the first Conv in the head
        head_layers = self.model[-1].m  # Assuming head is the last module
        for j, head_layer in enumerate(head_layers):
            if isinstance(head_layer, Conv):
                head_layers[j] = nn.Sequential(head_layer, SimAM())
                break  # Add to first Conv only, adjust as needed