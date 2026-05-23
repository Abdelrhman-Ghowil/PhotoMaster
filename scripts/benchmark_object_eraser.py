import statistics
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.image_service import erase_object, get_object_eraser_model


def _make_sample(size):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    box_pad = size // 3
    draw.rectangle(
        (box_pad, box_pad, size - box_pad, size - box_pad),
        fill="black",
    )

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle(
        (box_pad - 8, box_pad - 8, size - box_pad + 8, size - box_pad + 8),
        fill=255,
    )

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue(), mask


def _time_call(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.2f}s")
    return result, elapsed


def main():
    model_path = Path("assets/big-lama.pt")
    if not model_path.exists():
        raise SystemExit(
            "Missing assets/big-lama.pt. Add the Object Eraser model before running this benchmark."
        )

    model, model_load_seconds = _time_call("model load", get_object_eraser_model)
    if model is None:
        raise SystemExit("Object Eraser model did not load.")

    samples = [256, 512, 1024]
    timings = []

    for size in samples:
        image_content, mask = _make_sample(size)
        output, elapsed = _time_call(
            f"erase {size}x{size}",
            lambda: erase_object(image_content, mask),
        )
        if not output:
            raise SystemExit(f"Object Eraser returned no output for {size}x{size}.")
        timings.append(elapsed)

    image_content, mask = _make_sample(512)
    _, first_cached = _time_call("cache seed 512x512", lambda: erase_object(image_content, mask))
    _, second_cached = _time_call("cache hit 512x512", lambda: erase_object(image_content, mask))

    print("")
    print(f"average erase time: {statistics.mean(timings):.2f}s")
    print(f"median erase time: {statistics.median(timings):.2f}s")
    print(f"cache speedup: {first_cached / max(second_cached, 0.001):.1f}x")


if __name__ == "__main__":
    main()
