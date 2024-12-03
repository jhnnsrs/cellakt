from cellpose import models,core

model_type = "cyto"

model = models.Cellpose(gpu=True, model_type=model_type)
