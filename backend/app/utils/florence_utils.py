from __future__ import annotations

import logging
import os
import importlib
from typing import List, Tuple, Optional

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


_MODEL_CACHE: dict[str, tuple[AutoProcessor, AutoModelForCausalLM, str]] = {}


def _to_model_dtype_on_device(batch, model, device: str):
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                batch[key] = value.to(device=device, dtype=getattr(model, "dtype", value.dtype), non_blocking=True)
            else:
                batch[key] = value.to(device=device, non_blocking=True)
    return batch


def load_florence(model_id: str = "microsoft/Florence-2-large", device: Optional[str] = None, dtype_mode: Optional[str] = None) -> tuple[AutoProcessor, AutoModelForCausalLM, str]:
    global _MODEL_CACHE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = f"{model_id}:{device}:{dtype_mode or 'auto'}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # Map dtype_mode → torch dtype
    if dtype_mode in ("float16", "fp16"):
        dtype = torch.float16
    elif dtype_mode in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    elif dtype_mode in ("float32", "fp32"):
        dtype = torch.float32
    else:  # auto
        dtype = torch.float16 if (device.startswith("cuda")) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device).eval()

    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_id is not None:
        model.generation_config.eos_token_id = eos_id
        model.generation_config.pad_token_id = eos_id
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    model.generation_config.use_cache = False
    # Align defaults with script behavior
    try:
        model.generation_config.early_stopping = True
        model.generation_config.no_repeat_ngram_size = 3
    except Exception:
        pass
    if hasattr(model, "language_model") and hasattr(model.language_model, "generation_config"):
        inner = model.language_model.generation_config
        if eos_id is not None:
            inner.eos_token_id = eos_id
            inner.pad_token_id = eos_id
        inner.num_beams = 1
        inner.do_sample = False
        inner.use_cache = False
        try:
            inner.early_stopping = True
            inner.no_repeat_ngram_size = 3
        except Exception:
            pass

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    _MODEL_CACHE[cache_key] = (processor, model, device)
    return _MODEL_CACHE[cache_key]


@torch.inference_mode()
def run_florence_od_batch(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    images_pil: List[Image.Image],
    device: str,
    max_new_tokens: int = 96,
) -> List[dict]:
    prompts = ["<OD>"] * len(images_pil)
    inputs = processor(text=prompts, images=images_pil, return_tensors="pt", padding=True)
    inputs = _to_model_dtype_on_device(inputs, model, device)
    generated_ids = model.generate(
        **inputs,
        generation_config=model.generation_config,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        use_cache=False,
    )
    texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    outs = []
    for i, gen_text in enumerate(texts):
        out = processor.post_process_generation(
            text=gen_text,
            task="<OD>",
            image_size=(images_pil[i].width, images_pil[i].height),
        )
        outs.append(out)
    return outs


def parse_od_output(od_out: dict) -> tuple[list[Tuple[float, float, float, float]], list[str], list[float]]:
    od = od_out.get("<OD>", {})
    boxes = od.get("bboxes", []) or []
    labels = od.get("labels", []) or ["object"] * len(boxes)
    scores = od.get("scores", []) or [1.0] * len(boxes)
    return boxes, labels, scores


# ---------------- TensorRT / ONNX Runtime EP helpers -----------------

def _export_vision_tower_onnx(model, onnx_path: str, input_shape: tuple[int, int, int, int], opset: int = 17) -> None:
    class VTWrapper(torch.nn.Module):
        def __init__(self, vt, vt_dtype):
            super().__init__()
            self.vt = vt
            self.vt_dtype = vt_dtype

        def forward(self, x):
            x = x.to(dtype=self.vt_dtype)
            return self.vt.forward_features_unpool(x)

    vt = model.vision_tower
    vt_dtype = next(vt.parameters()).dtype
    device = next(model.parameters()).device
    wrapper = VTWrapper(vt, vt_dtype).to(device).eval()
    dummy = torch.randn(*input_shape, device=device, dtype=vt_dtype)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["features"],
        opset_version=opset,
        dynamic_axes=None,
    )


def _monkeypatch_vision_tower_with_ort(model, ort_session):
    vt = model.vision_tower
    vt_dtype = next(vt.parameters()).dtype
    device = next(model.parameters()).device

    def forward_features_unpool_ort(x: torch.Tensor):
        import numpy as np
        x_np = x.detach().to(dtype=vt_dtype).contiguous().cpu().numpy()
        outputs = ort_session.run(None, {"pixel_values": x_np})
        feats = outputs[0]
        feats_t = torch.from_numpy(feats).to(device=device, dtype=vt_dtype)
        return feats_t

    vt.forward_features_unpool = forward_features_unpool_ort  # type: ignore


def _np_from_trt_dtype(dtype):
    import numpy as np
    import tensorrt as trt  # type: ignore
    if dtype == trt.DataType.FLOAT:
        return np.float32
    if dtype == trt.DataType.HALF:
        return np.float16
    if dtype == trt.DataType.INT32:
        return np.int32
    raise ValueError(f"Unsupported TRT dtype: {dtype}")


class _TrtRunner:
    def __init__(self, engine, input_shape):
        import tensorrt as trt  # type: ignore
        import pycuda.driver as cuda  # type: ignore
        import numpy as np
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_idx = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)][0]
        self.output_idx = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)][0]
        self.input_name = engine.get_binding_name(self.input_idx)
        self.output_name = engine.get_binding_name(self.output_idx)
        self.input_shape = input_shape
        self.context.set_binding_shape(self.input_idx, input_shape)
        self.output_shape = self.context.get_binding_shape(self.output_idx)
        self.input_dtype = _np_from_trt_dtype(engine.get_binding_dtype(self.input_idx))
        self.output_dtype = _np_from_trt_dtype(engine.get_binding_dtype(self.output_idx))
        self.h_in = np.empty(self.input_shape, dtype=self.input_dtype)
        self.h_out = np.empty(self.output_shape, dtype=self.output_dtype)
        self.d_in = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.bindings = [int(self.d_in), int(self.d_out)]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import pycuda.driver as cuda  # type: ignore
        import numpy as np
        if x.dim() == 3:
            x = x.unsqueeze(0)
        outputs = []
        for i in range(x.shape[0]):
            xi = x[i].detach().float().contiguous().cpu().numpy().astype(self.input_dtype, copy=False)
            if xi.ndim == 3:
                xi = np.expand_dims(xi, axis=0)
            xi = np.ascontiguousarray(xi, dtype=self.input_dtype)
            cuda.memcpy_htod(self.d_in, xi)
            self.context.execute_v2(self.bindings)
            cuda.memcpy_dtoh(self.h_out, self.d_out)
            out = torch.from_numpy(self.h_out).to(x.device)
            outputs.append(out)
        return torch.stack(outputs, dim=0)


def _monkeypatch_vision_tower_with_trt(model, engine, input_shape):
    runner = _TrtRunner(engine, input_shape)
    vt = model.vision_tower

    def forward_features_unpool_trt(x):
        return runner(x)

    vt.forward_features_unpool = forward_features_unpool_trt  # type: ignore


def _build_trt_engine_for_vision_tower(model, engine_path: str, input_shape, fp16: bool, opset: int):
    import tensorrt as trt  # type: ignore
    logger = trt.Logger(trt.Logger.WARNING)
    onnx_path = engine_path + ".onnx"
    _export_vision_tower_onnx(model, onnx_path, input_shape, opset)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
        config = builder.create_builder_config()
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape("pixel_values", min=input_shape, opt=input_shape, max=input_shape)
        config.add_optimization_profile(profile)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("TRT OnnxParser failed: " + "\n".join(errors))
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TRT engine for vision tower")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())


def _load_trt_engine(engine_path: str):
    import tensorrt as trt  # type: ignore
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        return engine


def enable_trt_vision(
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    vision_h: int,
    vision_w: int,
    *,
    fp16: bool = True,
    opset: int = 17,
    cache_dir: Optional[str] = None,
)-> bool:
    """Enable TensorRT acceleration for Florence-2 vision tower if available.

    Returns True if enabled, False otherwise.
    """
    # Native TensorRT path disabled by request; always use ONNX Runtime providers
    logging.getLogger(__name__).info("Native TensorRT path disabled; attempting ONNX Runtime providers")

    # Fallback: ONNX Runtime with TRT/CUDA EP
    try:
        ort = importlib.import_module("onnxruntime")
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers or "CUDAExecutionProvider" in providers:
            onnx_path = os.path.join(cache_dir or os.getcwd(), f"vision_tower_{vision_h}x{vision_w}.onnx")
            if not os.path.isfile(onnx_path):
                _export_vision_tower_onnx(model, onnx_path, (1, 3, vision_h, vision_w), opset)
            trt_opts = {"trt_fp16_enable": bool(fp16), "trt_engine_cache_enable": True}
            if cache_dir:
                trt_opts["trt_engine_cache_path"] = cache_dir
            use_providers = []
            if "TensorrtExecutionProvider" in providers:
                use_providers.append(("TensorrtExecutionProvider", trt_opts))
            if "CUDAExecutionProvider" in providers:
                use_providers.append("CUDAExecutionProvider")
            use_providers.append("CPUExecutionProvider")
            sess = ort.InferenceSession(onnx_path, providers=use_providers)
            _monkeypatch_vision_tower_with_ort(model, sess)
            try:
                active_providers = sess.get_providers() if hasattr(sess, "get_providers") else getattr(sess, "providers", [])
            except Exception:
                active_providers = []
            logging.getLogger(__name__).info(
                "Enabled Florence vision tower via ONNX Runtime EP; requested=%s active=%s",
                use_providers,
                active_providers,
            )
            return True
    except Exception as e:
        logging.getLogger(__name__).info("ORT/TRT EP not available or failed: %s", e)
    return False



