"""Anomaly Detection application -- paired with
architectures/autoencoder.py's AnomalyAutoencoder. Reconstruction-based
anomaly detection: the standard technique for this model family (train,
or here random-init -- same "traffic over accuracy" policy as every
other architecture, on data resembling "normal"; flag samples the model
reconstructs poorly as anomalous).

preprocess reuses the exact same real-image conversion
applications/image_reconstruction.py already does (families/autoencoder's
scale_to_unit_range) -- same input shape/range the underlying
_AutoencoderModel needs either way.

postprocess receives the per-sample reconstruction-error score
AnomalyAutoencoder.forward() already computed (not raw pixels -- see that
module's own docstring for why the comparison against the original input
happens inside the architecture instead of here) and applies a fixed
threshold to turn it into a human-readable decision. The threshold itself
is arbitrary -- every architecture in this project is random-init, so
"normal" reconstruction error has no calibrated baseline to set it from --
the point is exercising the real request/response shape a genuine
anomaly-detection deployment has (score in, threshold, decision out), not
a tuned detector.
"""
import numpy as np

from applications.base import Application
from core.registry import APPLICATIONS
from families.autoencoder import scale_to_unit_range

ANOMALY_SCORE_THRESHOLD = 0.05


class AnomalyDetection(Application):
    def preprocess(self, raw_sample):
        if isinstance(raw_sample, np.ndarray):
            return scale_to_unit_range(raw_sample)
        return raw_sample  # already a tensor

    def postprocess(self, output_tensor):
        score = float(output_tensor.reshape(-1)[0].item())
        return {
            "anomaly_score": score,
            "is_anomalous": score > ANOMALY_SCORE_THRESHOLD,
            "threshold": ANOMALY_SCORE_THRESHOLD,
        }


APPLICATIONS.register("Anomaly Detection", implemented=True, datasets=["CIFAR10"])(AnomalyDetection)
