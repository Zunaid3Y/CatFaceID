import json
from pathlib import Path

import coremltools as ct


ONNX_MODEL_PATH = Path("models/embedder.onnx")
COREML_MODEL_PATH = Path("models/embedder.mlmodel")
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def main() -> None:
    if not ONNX_MODEL_PATH.exists():
        raise SystemExit(f"ONNX model not found at {ONNX_MODEL_PATH}. Run scripts/export_onnx.py first.")

    image_scale = 1.0 / 255.0
    input_type = ct.ImageType(
        name="input",
        shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        color_layout="RGB",
        scale=image_scale,
        bias=(0.0, 0.0, 0.0),
    )

    mlmodel = ct.converters.onnx.convert(
        model=str(ONNX_MODEL_PATH),
        inputs=[input_type],
        minimum_deployment_target=ct.target.iOS16,
    )

    COREML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(COREML_MODEL_PATH))

    spec = mlmodel.get_spec()
    model_inputs = [
        {
            "name": inp.name,
            "type": inp.type.WhichOneof("Type"),
        }
        for inp in spec.description.input
    ]
    model_outputs = [
        {
            "name": out.name,
            "type": out.type.WhichOneof("Type"),
        }
        for out in spec.description.output
    ]

    print("Saved CoreML model to:", COREML_MODEL_PATH)
    print("Inputs:", json.dumps(model_inputs, indent=2))
    print("Outputs:", json.dumps(model_outputs, indent=2))
    mean_str = ", ".join(f"{m:.3f}" for m in IMAGE_MEAN)
    std_str = ", ".join(f"{s:.3f}" for s in IMAGE_STD)
    print(f"Normalization: subtract mean ({mean_str}) and divide by std ({std_str}) per channel before inference.")
    print(
        "Inputs expect pixels scaled by 1/255 ahead of normalization."
    )
    print(
        "iOS usage tip: let model = try! Embedder(configuration: MLModelConfiguration()); "
        "resize to 224x224 RGB, normalize with ImageNet mean/std, then feed via model.prediction."
    )


if __name__ == "__main__":
    main()
