import numpy as np

from datasets.preprocessing import MRIProcessor, PreprocessingConfig

processor = MRIProcessor(PreprocessingConfig())

volume = np.random.rand(
    240,
    240,
    155,
).astype(np.float32)

processed = processor.preprocess(volume)

processor.summary(processed)

print()

print(processed.shape)
