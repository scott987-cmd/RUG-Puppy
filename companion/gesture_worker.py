"""Private, one-image CPU worker. Input: JPEG stdin. Output: small JSON only."""
import io
import json
import math
import sys
from dataclasses import asdict

if __package__:
    from .gesture import GestureObservation, MAPPING, MAX_JPEG_BYTES
else:
    from gesture import GestureObservation, MAPPING, MAX_JPEG_BYTES


def recognize(jpeg, model_path):
    import numpy as np
    from PIL import Image
    import mediapipe as mp

    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError("invalid image length")
    with Image.open(io.BytesIO(jpeg)) as source:
        if source.format != "JPEG" or source.width * source.height > 2_000_000 or min(source.size) < 64:
            raise ValueError("invalid image dimensions/format")
        rgb = np.asarray(source.convert("RGB"))
    gray = rgb.astype(np.float32).mean(axis=2)
    brightness = float(gray.mean())
    laplacian = (gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1]
                 + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1])
    if not 35 <= brightness <= 220 or float(laplacian.var()) < 20:
        return GestureObservation("uncertain", reason="image_quality")
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path,
                                          delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=.7, min_hand_presence_confidence=.7,
        canned_gesture_classifier_options=mp.tasks.components.processors.ClassifierOptions(max_results=2))
    with mp.tasks.vision.GestureRecognizer.create_from_options(options) as engine:
        result = engine.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks and not result.gestures:
        return GestureObservation("absent")
    if len(result.hand_landmarks) != 1 or len(result.gestures) != 1:
        return GestureObservation("uncertain", reason="not_one_hand")
    points = result.hand_landmarks[0]
    if len(points) != 21 or any(not math.isfinite(v) for p in points for v in (p.x, p.y, p.z)):
        return GestureObservation("uncertain", reason="invalid_landmarks")
    xs, ys = [p.x for p in points], [p.y for p in points]
    if (min(xs + ys) < .015 or max(xs + ys) > .985
            or (max(xs) - min(xs)) * (max(ys) - min(ys)) < .015):
        return GestureObservation("uncertain", reason="hand_small_or_cropped")
    choices = sorted(result.gestures[0], key=lambda item: item.score, reverse=True)
    if not choices:
        return GestureObservation("uncertain", reason="no_category")
    best = choices[0]
    margin = best.score - choices[1].score if len(choices) > 1 else best.score
    if best.category_name not in MAPPING or best.score < .85 or margin < .15:
        return GestureObservation("uncertain", reason="classification")
    return GestureObservation("gesture", best.category_name, float(best.score))


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("model path required")
        output = recognize(sys.stdin.buffer.read(MAX_JPEG_BYTES + 1), sys.argv[1])
        print(json.dumps(asdict(output), allow_nan=False), flush=True)
    except Exception:
        # Never log input bytes, photo paths, or native exception dumps.
        raise SystemExit(1)
