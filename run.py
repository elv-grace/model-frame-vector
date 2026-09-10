from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from general_detection.model import FrameVectorModel
from general_detection.config import RuntimeConfig
from config import config

if __name__ == '__main__':
    setproctitle.setproctitle('model-frame-vector')

    catch_errors()

    params = get_params()
    params = from_dict(RuntimeConfig, data=params)

    # One whole-frame vector tag per sampled frame, or -- when the request sets `detect_target` --
    # one per detected crop, plus a vector-less twin of each when it also sets `output_tags`.
    # Detector weights are not passed here. Which detectors run (if any) depends on the request's
    # `detect_target` and `brand_detector`, so the model resolves and lazily loads them itself
    # from the measured defaults in general_detection/detector.py.
    model = FrameVectorModel(
        cfg=params,
        embedder_model_id=config["model"]["embedder"]["model_id"],
        embedder_revision=config["model"]["embedder"].get("revision"),
        cache_dir=config["storage"]["cache_path"],
    )

    run_default(model)
