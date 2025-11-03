def _get_conf_thresh():
    import os
    # Read from env if provided; 0.0 is valid
    for k in ('DERM_CONF','DERM_CONF_THRESH','DERMA_THRESHOLD','CONF_LESION','CONF_ACNE','CONF'):
        v = os.getenv(k)
        if v is not None:
            try:
                val = float(v)
                return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
            except Exception:
                pass
    return 0.15

CONF_THRESH = _get_conf_thresh()

# --- Compat imports (idempotent) ---
import os as _os
import os
import time
import numpy as np
import cv2
_np = np
import queue
import numpy as np
_np = np
import cv2
print("[DERM-HAILO] HOTFIX2 ACTIVE — hailo_dual_hef.py __file__=", __file__)

ACNE_CLASS_NAMES = ["comedone", "papule", "pustule", "acne_scar", "seborrheic_hyperplasia", "other"]
LESION_CLASS_NAMES = ["lesion"]

try:
    from multiprocessing import shared_memory as _shm
except Exception:
    _shm = None
import hailo_platform as hp
from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm
from .hailo_device_manager import device_manager

# --- Robust coercers for Hailo NMS variants ---
import numpy as _np
import numbers as _numbers
import os as _os
import re as _re

# Keep preproc constants; default to 640x640 if not defined elsewhere
try:
    PREPROC_W
except NameError:
    PREPROC_W = 640
try:
    PREPROC_H
except NameError:
    PREPROC_H = 640

CONF_THRESH = _get_conf_thresh()
DERM_BYPASS_ROI = bool(int(_os.getenv("DERM_BYPASS_ROI", "0")))
DERM_FAKE = bool(int(_os.getenv("DERM_FAKE", "0")))

def _is_scalar(x):
    return isinstance(x, (_numbers.Number, _np.generic))

def _safe_asarray(x, dtype=_np.float32, copy=False):
    try:
        a = __safe_asarray(x, dtype=dtype)
        if a.dtype != dtype:
            a = a.astype(dtype, copy=False)
        return a
    except Exception:
        return _np.empty((0,), dtype=dtype)

def _to_listish(x):
    """Return a Python list/ndarray/str suitable for parsing or [] on failure."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, _np.ndarray)):
        return x
    for attr in _iterify(("to_numpy", "numpy", "tolist")):
        try:
            if hasattr(x, attr):
                y = getattr(x, attr)()
                return y if y is not None else []
        except Exception:
            pass
    if isinstance(x, (bytes, bytearray, _np.void)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    return x

def _tokenize_numbers(s: str):
    """Extract floats from arbitrary text like '0.12, 0.5  1.2e-1 [0.7]'."""
    try:
        txt = str(s)
    except Exception:
        return _np.empty((0,), dtype=_np.float32)
    toks = _re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", txt)
    if not toks:
        return _np.empty((0,), dtype=_np.float32)
    try:
        arr = _np.asarray([float(t) for t in toks], dtype=_np.float32)
    except Exception:
        arr = _np.empty((0,), dtype=_np.float32)
    return arr

def _flat_floats(x):
    """Return a 1D float32 array of finite values from arrays/lists/scalars/str/bytes."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return _np.empty((0,), dtype=_np.float32)
    if isinstance(x, (list, tuple, _np.ndarray)):
        try:
            a = _safe_asarray(x).reshape(-1)
            a = a[_np.isfinite(a)]
            return a.astype(_np.float32, copy=False)
        except Exception:
            return _np.empty((0,), dtype=_np.float32)
    if isinstance(x, (str, bytes, bytearray, _np.void)):
        a = _tokenize_numbers(x)
        return a[_np.isfinite(a)].astype(_np.float32, copy=False)
    if _is_scalar(x) and _np.isfinite(x):
        return _np.asarray([float(x)], dtype=_np.float32)
    try:
        a = _safe_asarray(x).reshape(-1)
        a = a[_np.isfinite(a)]
        return a.astype(_np.float32, copy=False)
    except Exception:
        return _np.empty((0,), dtype=_np.float32)

def _solve_kx5_variants(f, W=PREPROC_W, H=PREPROC_H):
    """
    Accept flat floats f of length 5*K or 6*K.
    Try layouts:
      A: [x1,y1,x2,y2,conf] (xyxy last is conf)
      B: [conf,x1,y1,x2,y2] (conf-first)
      C: [x1,y1,x2,y2,conf,cls] (drop cls)
    Pixel or normalized: if any geom >1.5, normalize first by W/H, then clip.
    Choose the variant with the most valid boxes (w>0 & h>0) and reasonable conf [0,1].
    Return (K,5) [cx,cy,w,h,conf] normalized to [0,1].
    """
    def _reshape5(ff):
        K5 = (ff.size // 5) * 5
        if K5 == 0: return _np.empty((0,5), dtype=_np.float32)
        return ff[:K5].reshape(-1,5).astype(_np.float32, copy=False)

    if f.size % 6 == 0 and f.size >= 6:
        ff = f.copy().reshape(-1,6)[:, :5].reshape(-1)
    else:
        ff = f

    A = _reshape5(ff)
    if ff.size >= 5:
        B = ff.copy().reshape(-1,5)
        B = _np.concatenate([B[:,1:], B[:,0:1]], axis=1).astype(_np.float32, copy=False)
    else:
        B = _np.empty((0,5), dtype=_np.float32)

    def _normalize_and_score(kx5):
        if kx5.size == 0:
            return kx5, -1.0
        arr = kx5.copy()
        maxv = float(_np.max(_np.abs(arr[:, :4]))) if arr.size else 0.0
        is_pixels = maxv > 1.5
        if (is_pixels is not None and (getattr(is_pixels,"size",0)>0 if hasattr(is_pixels,"__array__") else bool(is_pixels))):
            arr[:, [0,2]] /= float(W)
            arr[:, [1,3]] /= float(H)
        wxy = arr[:,2] - arr[:,0]; hxy = arr[:,3] - arr[:,1]
        xyxy_score = 0.0
        try:
            xyxy_score = 0.5*(float(_np.mean(wxy > 0)) + float(_np.mean(hxy > 0)))
        except Exception:
            pass
        if xyxy_score >= 0.5:
            cx = (arr[:,0] + arr[:,2]) * 0.5
            cy = (arr[:,1] + arr[:,3]) * 0.5
            w  = wxy; h = hxy
        else:
            cx, cy, w, h = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
        out = _np.stack([cx,cy,w,h,arr[:,4]], axis=1).astype(_np.float32, copy=False)
        out[:, :4] = _np.clip(out[:, :4], 0.0, 1.0)
        valid = (out[:,2] > 0) & (out[:,3] > 0) & (out[:,4] >= 0.0) & (out[:,4] <= 1.0)
        score = float(_np.mean(valid)) if out.shape[0] else -1.0
        return out, score

    A_out, A_score = _normalize_and_score(A)
    B_out, B_score = _normalize_and_score(B)
    best = A_out if A_score >= B_score else B_out
    return best

def _maybe_xyxy_to_cxcywh(kx5, W=PREPROC_W, H=PREPROC_H):
    """
    kx5: (K,5) as [a,b,c,d,conf] where geometry may be either:
      - xyxy+conf (x1,y1,x2,y2,conf) or
      - cxcywh+conf (cx,cy,w,h,conf)
    Accepts normalized or pixel units. If any geom value > 1.5, treat as pixels first.
    """
    if not isinstance(kx5, _np.ndarray) or kx5.size == 0:
        return _np.empty((0,5), dtype=_np.float32)
    a = kx5.astype(_np.float32, copy=False)
    maxv = float(_np.max(_np.abs(a[:, :4]))) if a.size else 0.0
    is_pixels = maxv > 1.5
    if (is_pixels is not None and (getattr(is_pixels,"size",0)>0 if hasattr(is_pixels,"__array__") else bool(is_pixels))):
        a_norm = a.copy()
        a_norm[:, [0, 2]] /= float(W)
        a_norm[:, [1, 3]] /= float(H)
    else:
        a_norm = a
    x1, y1, x2, y2 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    wxy = x2 - x1
    hxy = y2 - y1
    xyxy_score = 0.0
    try:
        xyxy_score = 0.5 * (float(_np.mean(wxy > 0)) + float(_np.mean(hxy > 0)))
    except Exception:
        pass
    if xyxy_score >= 0.5:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = wxy
        h = hxy
        out = _np.stack([cx, cy, w, h, a_norm[:, 4]], axis=1).astype(_np.float32, copy=False)
    else:
        b = a.copy()
        if (is_pixels is not None and (getattr(is_pixels,"size",0)>0 if hasattr(is_pixels,"__array__") else bool(is_pixels))):
            b[:, [0, 2]] /= float(W)
            b[:, [1, 3]] /= float(H)
        out = _np.stack([b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]], axis=1).astype(_np.float32, copy=False)
    out[:, :4] = _np.clip(out[:, :4], 0.0, 1.0)
    return out
import numpy as _np
import numbers as _numbers
import os as _os

# Keep preproc constants; default to 640x640 if not defined elsewhere
try:
    PREPROC_W
except NameError:
    PREPROC_W = 640
try:
    PREPROC_H
except NameError:
    PREPROC_H = 640

CONF_THRESH = _get_conf_thresh()

def _is_scalar(x):
    return isinstance(x, (_numbers.Number, _np.generic))

def _safe_asarray(x, dtype=_np.float32, copy=False):
    try:
        a = __safe_asarray(x, dtype=dtype)
        if a.dtype != dtype:
            a = a.astype(dtype, copy=False)
        return a
    except Exception:
        return _np.empty((0,), dtype=dtype)

def _coerce_kx5(obj, W=PREPROC_W, H=PREPROC_H):
    """
    Accepts: None, scalar, strings/bytes, flat list/array length 5 or 5*K, nested lists/arrays.
    Returns (K,5) float32 [cx,cy,w,h,conf], normalized [0,1] for geometry.
    """
    f = _flat_floats(_to_listish(obj))
    if f.size < 5:
        return _np.empty((0,5), dtype=_np.float32)
    out = _solve_kx5_variants(f, W, H)
    return out if isinstance(out, _np.ndarray) else _np.empty((0,5), dtype=_np.float32)
_coerce_kx5_norm = _coerce_kx5

def _pack_with_cls(kx5, cls_id):
    if not isinstance(kx5, _np.ndarray) or kx5.size == 0:
        return _np.empty((0,6), dtype=_np.float32)
    cls_col = _np.full((kx5.shape[0], 1), float(cls_id), dtype=_np.float32)
    out = _np.concatenate([kx5[:, :5], cls_col], axis=1).astype(_np.float32, copy=False)
    if out.size:
        out = out[out[:,4] >= CONF_THRESH]
    return out

def _stack_or_empty(*arrs):
    good = [a for a in arrs if isinstance(a, _np.ndarray) and a.size and a.shape[1] == 6]
    if not good:
        return _np.empty((0,6), dtype=_np.float32)
    return _np.concatenate(good, axis=0).astype(_np.float32, copy=False)


def _decode_hailo_dual_nms(outputs, conf_thres=0.15, max_keep=200):
    """
    Robust decoder for Hailo NMS outputs.
    Accepts either:
      • nested per-class lists of (K,5) arrays, or
      • 3D arrays like (C,5,K) or (C,K,5), or
      • 2D arrays like (5,K) / (K,5).
    Returns:
      det_all: (M,6) float32 with columns (cx,cy,w,h,conf,cls) in [0..1] space.
    """
    import numpy as np

    def to_list(x):
        if x is None:
            return []
        return list(x) if isinstance(x, (list, tuple)) else [x]

    def norm_kx5(a):
        """Normalize any recognized shape into (K,5) float32, otherwise empty (0,5)."""
        a = np.asarray(a)
        if a.ndim == 3:
            # (C,5,K) or (C,K,5)
            if a.shape[1] == 5:      # (C,5,K) -> (5,K) -> (K,5)
                a = a[0].transpose(1, 0)
            elif a.shape[2] == 5:    # (C,K,5) -> (K,5)
                a = a[0]
            else:
                return np.empty((0, 5), np.float32)
        elif a.ndim == 2:
            if a.shape[1] == 5:      # (K,5) -> ok
                pass
            elif a.shape[0] == 5:    # (5,K) -> (K,5)
                a = a.T
            else:
                return np.empty((0, 5), np.float32)
        elif a.ndim == 1 and a.size == 5:
            a = a[None, :]
        else:
            return np.empty((0, 5), np.float32)
        return a.astype(np.float32, copy=False)

    dets = []

    # Lesion head (treat as class 0)
    for lvl1 in to_list(outputs.get("lesion")):
        for el in to_list(lvl1):
            kx5 = norm_kx5(el)
            if kx5.size:
                kx5 = kx5[kx5[:, 4] >= float(conf_thres)]
                if kx5.size:
                    cls = np.full((kx5.shape[0], 1), 0.0, dtype=np.float32)
                    dets.append(np.concatenate([kx5, cls], axis=1))

    # Acne head (often 6 per-class arrays)
    for lvl1 in to_list(outputs.get("acne")):
        if isinstance(lvl1, (list, tuple)):
            iterable = list(enumerate(lvl1))
        else:
            iterable = [(0, lvl1)]
        for j, el in iterable:
            kx5 = norm_kx5(el)
            if kx5.size:
                kx5 = kx5[kx5[:, 4] >= float(conf_thres)]
                if kx5.size:
                    cls = np.full((kx5.shape[0], 1), float(j), dtype=np.float32)
                    dets.append(np.concatenate([kx5, cls], axis=1))

    if dets:
        det_all = np.concatenate(dets, axis=0)
        if det_all.shape[0] > max_keep:
            det_all = det_all[:max_keep]
    else:
        det_all = np.empty((0, 6), np.float32)

    return det_all

def _to_pixel_xyxy(det_all, W=PREPROC_W, H=PREPROC_H):
    if not isinstance(det_all, _np.ndarray) or det_all.size == 0:
        return _np.empty((0,6), dtype=_np.float32)
    cx,cy,w,h,conf,cls_ = _np.split(det_all, [1,2,3,4,5], axis=1)
    x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
    boxes = _np.concatenate([x1,y1,x2,y2,conf,cls_], axis=1).astype(_np.float32, copy=False)
    boxes[:,0] = _np.clip(boxes[:,0], 0, W-1)
    boxes[:,1] = _np.clip(boxes[:,1], 0, H-1)
    boxes[:,2] = _np.clip(boxes[:,2], 0, W-1)
    boxes[:,3] = _np.clip(boxes[:,3], 0, H-1)
    return boxes


def _iterify(x):
    """Yield x if iterable; otherwise yield nothing. Avoids 'int is not iterable'."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, dict, _np.ndarray)):
        return x
    return []

# --- end helpers ---

# -- resolve classes regardless of wheel layout --
def _get_cls(name):
    for mod in _iterify((hp, getattr(hp, "pyhailort", None), getattr(getattr(hp,"pyhailort",None), "_pyhailort", None))):
        if mod and hasattr(mod, name):
            return getattr(mod, name)
    return None

InferVStreams = _get_cls("InferVStreams")
VStreamParams = _get_cls("VStreamParams")
assert InferVStreams and VStreamParams, "Hailo classes not found (check hailort 4.22 wheel)."

# === CODEX PATCH #2: META trace flag BEGIN ===
_META_TRACE = str(os.getenv('DERM_TRACE_META', '')).lower() in ('1', 'true', 'on')
# === CODEX PATCH #2: META trace flag END ===

def _force_host_user_buffers(params_obj):
    # Best-effort: force host/user buffers, avoid DMA to prevent VDMA mapping errors.
    # Wheel APIs vary; guard each attribute.
    for attr, val in (
        ("dma_buf", False),
        ("use_dma", False),
        ("dma", False),
        ("is_dma", False),
    ):
        if hasattr(params_obj, attr):
            try: setattr(params_obj, attr, val)
            except Exception: pass
    for attr, val in (
        ("is_user_buffer", True),
        ("use_dma_buffer", False),
    ):
        if hasattr(params_obj, attr):
            try: setattr(params_obj, attr, val)
            except Exception: pass
    # Some wheels expose memory_type or buffer_type enums
    for attr, pref in (
        ("memory_type", ("HOST", "USER", "CPU")),
        ("buffer_type", ("USER", "HOST")),
        ("transfer_mode", ("USER", "HOST")),
    ):
        if hasattr(params_obj, attr):
            for v in _iterify(pref):
                try:
                    enum_ns = getattr(hp, "pyhailort", None) or hp
                    enum = getattr(enum_ns, attr.upper(), None)
                    if enum and hasattr(enum, v):
                        setattr(params_obj, attr, getattr(enum, v))
                        break
                except Exception:
                    pass

def _make_params(info):
    f = getattr(VStreamParams, "make_from_vstream_info", None)
    try:
        p = f(info) if f else VStreamParams(info)
    except Exception:
        p = VStreamParams()
    for a in _iterify(("queue_size","max_queue_size","queue_depth","queue_size_frames")):
        if hasattr(p,a):
            try: setattr(p,a, 4)
            except: pass
    for a in _iterify(("timeout_ms","timeout")):
        if hasattr(p,a):
            try: setattr(p,a, 10000)
            except: pass
    _force_host_user_buffers(p)
    return p

def _buf_flags(params_obj):
    is_dma = False
    is_user = False
    for a in _iterify(("dma_buf","use_dma","dma","is_dma")):
        if hasattr(params_obj,a):
            try:
                if bool(getattr(params_obj,a)):
                    is_dma = True
            except Exception:
                pass
    for a in _iterify(("memory_type","buffer_type","transfer_mode")):
        if hasattr(params_obj,a):
            try:
                sval = str(getattr(params_obj,a))
                if 'USER' in sval or 'HOST' in sval or 'CPU' in sval:
                    is_user = True
            except Exception:
                pass
    if not is_dma and not is_user:
        is_user = True
    return is_user, is_dma

def _letterbox(rgb, size=(640,640)):
    ih, iw = size
    h, w = rgb.shape[:2]
    scale = min(iw / float(w), ih / float(h))
    nw, nh = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
    dw = (iw - nw) // 2
    dh = (ih - nh) // 2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    return canvas, scale, dw, dh

# Preproc constants and confidence threshold (env-configured)
PREPROC_W = 640
PREPROC_H = 640
try:
    CONF_THRESH = _get_conf_thresh()
except Exception:
    CONF_THRESH = _get_conf_thresh()

def _nms(dets, iou_th=0.45):
    if not dets:
        return []
    boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2'], d['score']] for d in dets], dtype=np.float32)
    x1, y1, x2, y2, sc = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_th)[0]
        order = order[inds + 1]
    return [dets[k] for k in keep]

def _center_square_rgb(bgr, out_sz=(640,640)):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h,w = rgb.shape[:2]; s = min(h,w)
    y0 = (h-s)//2; x0 = (w-s)//2
    crop = rgb[y0:y0+s, x0:x0+s]
    img  = cv2.resize(crop, out_sz).astype(np.uint8)
    if not img.flags['C_CONTIGUOUS']: img = np.ascontiguousarray(img)
    return img, (x0,y0,s)

def _first_array(x):
    if isinstance(x, np.ndarray): return x
    if isinstance(x, (list,tuple)):
        for it in _iterify(x):
            arr = _first_array(it)
            if isinstance(arr, np.ndarray): return arr
        return np.array([])
    try: return _safe_asarray(x)
    except Exception: return np.array([])

import numpy as np

def _nms_list_to_dets(nms_list, conf_thresh=0.15, cls_id=None):
    """
    Accepts Hailo-style list-of-class NMS outputs: a Python list of arrays with shape (K,5),
    or nested variants like (1,K,5). Returns float32 array (M,6) with columns:
    cx, cy, w, h, conf, cls  (all normalized 0..1). If empty -> shape (0,6).
    """
    rows = []
    if nms_list is None:
        return np.zeros((0,6), dtype=np.float32)
    for arr in _iterify((nms_list if isinstance(nms_list, (list, tuple)) else [nms_list])):
        a = _safe_asarray(arr)
        if a.size == 0:
            continue
        # Accept shapes like (K,5) or (1,K,5); squeeze any leading dims > 2
        if a.ndim > 2:
            a = np.squeeze(a, axis=0)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[-1] != 5:
            # Skip any malformed entries
            continue
        # a is (K,5) -> xyxy+conf normalized (0..1) per Hailo postprocess
        x1, y1, x2, y2, conf = a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]
        # confidence gate
        keep = conf >= float(conf_thresh)
        if not np.any(keep):
            continue
        x1, y1, x2, y2, conf = x1[keep], y1[keep], x2[keep], y2[keep], conf[keep]
        # xyxy -> cxcywh
        w  = (x2 - x1)
        h  = (y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        # clamp to [0,1]
        for v in _iterify((cx, cy, w, h, conf)):
            np.clip(v, 0.0, 1.0, out=v)
        rows.append(np.stack([cx, cy, w, h, conf], axis=1))

    if not rows:
        out = np.zeros((0,6), dtype=np.float32)
    else:
        dets = np.concatenate(rows, axis=0)  # (M,5)
        cls_col = np.full((dets.shape[0], 1), float(cls_id if cls_id is not None else 0.0), dtype=np.float32)
        out = np.concatenate([dets.astype(np.float32, copy=False), cls_col], axis=1)  # (M,6)

    # sanitize
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if out.ndim != 2 or out.shape[1] != 6:
        out = np.zeros((0,6), dtype=np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)

def _coerce_kx5(obj) -> np.ndarray:
    """Compatibility shim: delegate to normalized coercer."""
    try:
        return _coerce_kx5_norm(obj)
    except Exception:
        return np.empty((0,5), np.float32)

def _pack_with_cls(kx5: np.ndarray, cls_id: int) -> np.ndarray:
    """
    Turn (K,5) → (K,6) by appending the class id. (x1,y1,x2,y2,conf,cls)
    """
    if kx5.size == 0:
        return np.empty((0,6), np.float32)
    cls_col = np.full((kx5.shape[0],1), float(cls_id), dtype=np.float32)
    out = np.concatenate([kx5.astype(np.float32, copy=False), cls_col], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)

def _to_pixel_xyxy(norm_xyxy: np.ndarray, w=PREPROC_W, h=PREPROC_H) -> np.ndarray:
    """
    Convert normalized [x1,y1,x2,y2] to pixel xyxy in the 640x640 preproc frame.
    Clamps to image bounds. Returns (K,4) float32.
    """
    if norm_xyxy.size == 0:
        return np.empty((0,4), np.float32)
    a = norm_xyxy.astype(np.float32, copy=False)
    a[:,[0,2]] *= float(w)
    a[:,[1,3]] *= float(h)
    a[:,0::2] = np.clip(a[:,0::2], 0, w-1)
    a[:,1::2] = np.clip(a[:,1::2], 0, h-1)
    return np.ascontiguousarray(a[:, :4], dtype=np.float32)




import numpy as np
import os

# Confidence threshold (adjustable via env)
try:
    CONF_THRESH = _get_conf_thresh()
except Exception:
    CONF_THRESH = _get_conf_thresh()

def _safe_print(*a, **k):
    try:
        print(*a, **k)
    except Exception:
        pass

def _decode_hailo_yolov8_nms_by_class(out_dict, class_names, min_conf=0.25, log_once_key=None):
    dets = []

    # Normalize Hailo outputs to a single numpy array "nms"
    nms = None
    nms_key = None

    if isinstance(out_dict, dict):
        # pick the first float-like output
        for k, v in _iterify(out_dict.items()):
            arr = _safe_asarray(v)
            if arr.dtype in (np.float32, np.float16, np.float64):
                nms = arr
                nms_key = k
                break
        if nms is None:
            return dets
    elif isinstance(out_dict, (list, tuple)):
        # Hailo pyhailort often returns a list; use the first element
        if not out_dict:
            return dets
        nms = _safe_asarray(out_dict[0])
        nms_key = "list[0]"
    elif isinstance(out_dict, np.ndarray):
        nms = out_dict
        nms_key = "array"
    else:
        return dets

    # Logging with robust sample extraction
    try:
        f = _flat_floats(nms)
        name = f"{log_once_key or ''} key={nms_key}"
        sample = " ".join([f"{v:.3f}" for v in f[:15]]) if f.size else ""
        if f.size:
            _safe_print(f"[HAILO OUT] {name} shape={(f.shape,)} dtype=float32")
        else:
            _safe_print(f"[HAILO OUT] {name} shape=<no-shape> dtype=None")
        _safe_print(f"[HAILO OUT SAMPLE] {name}: {sample}")
    except Exception:
        pass

    def _append(cls_id, rec):
        cx, cy, w, h, score = rec[:5]
        if not np.isfinite(score) or score < float(min_conf):
            return
        if not np.all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
            return
        if cls_id is None or cls_id < 0 or cls_id >= len(class_names):
            return
        dets.append({
            "cls_id": int(cls_id),
            "cls_name": str(class_names[cls_id]),
            "conf": float(score),
            "cx": float(np.clip(cx, 0.0, 1.0)),
            "cy": float(np.clip(cy, 0.0, 1.0)),
            "w":  float(np.clip(w,  0.0, 1.0)),
            "h":  float(np.clip(h,  0.0, 1.0)),
        })

    # Decode both layouts:
    if hasattr(nms, "ndim") and nms.ndim == 3:
        # BY CLASS: (C, B, F)
        C, B, F = nms.shape
        for c in _iterify(range(C)):
            for b in _iterify(range(B)):
                _append(c, nms[c, b, :])
    elif hasattr(nms, "ndim") and nms.ndim == 2:
        # Unified: (B, F) possibly with class in last column
        B, F = nms.shape
        cls_col = None
        if F >= 6:
            last = nms[:, -1]
            if np.all((last >= -0.5) & (last < len(class_names) + 0.5)):
                cls_col = -1
        for b in _iterify(range(B)):
            rec = nms[b, :]
            if cls_col is not None:
                c_id = int(round(rec[cls_col]))
                _append(c_id, rec[:5])
            else:
                # No class column; if single-class model, use class 0,
                # otherwise skip (we can't assign class safely).
                if len(class_names) == 1:
                    _append(0, rec[:5])
    else:
        # 1-D row or unknown → try to parse rows of length >= 5
        arr = _safe_asarray(nms)
        if arr.size >= 5:
            row = arr.reshape(-1)[:5]
            if len(class_names) == 1:
                _append(0, row)
    return dets

import numpy as np

def _safe_print(*a, **k):
    try:
        print(*a, **k)
    except Exception:
        pass

def _decode_hailo_yolov8_nms_by_class(out_dict, class_names, min_conf=0.25, log_once_key=None):
    dets = []

    # Normalize Hailo outputs to a single numpy array "nms"
    nms = None
    nms_key = None

    if isinstance(out_dict, dict):
        # pick the first float-like output
        for k, v in _iterify(out_dict.items()):
            arr = _safe_asarray(v)
            if arr.dtype in (np.float32, np.float16, np.float64):
                nms = arr
                nms_key = k
                break
        if nms is None:
            return dets
    elif isinstance(out_dict, (list, tuple)):
        # Hailo pyhailort often returns a list; use the first element
        if not out_dict:
            return dets
        nms = _safe_asarray(out_dict[0])
        nms_key = "list[0]"
    elif isinstance(out_dict, np.ndarray):
        nms = out_dict
        nms_key = "array"
    else:
        return dets

    # Logging (robust to non-ndarrays)
    try:
        shp = tuple(nms.shape) if hasattr(nms, "shape") else "<no-shape>"
        dt = getattr(nms, "dtype", None)
        _safe_print(f"[HAILO OUT] {log_once_key or ''} key={nms_key} shape={shp} dtype={dt}")
        f = _flat_floats(nms)[:12]
        _safe_print("[HAILO OUT SAMPLE]", " ".join(f"{float(x):.3f}" for x in f))
    except Exception:
        pass

    def _append(cls_id, rec):
        cx, cy, w, h, score = rec[:5]
        if not np.isfinite(score) or score < float(min_conf):
            return
        if not np.all(np.isfinite([cx, cy, w, h])) or w <= 0 or h <= 0:
            return
        if cls_id is None or cls_id < 0 or cls_id >= len(class_names):
            return
        dets.append({
            "cls_id": int(cls_id),
            "cls_name": str(class_names[cls_id]),
            "conf": float(score),
            "cx": float(np.clip(cx, 0.0, 1.0)),
            "cy": float(np.clip(cy, 0.0, 1.0)),
            "w":  float(np.clip(w,  0.0, 1.0)),
            "h":  float(np.clip(h,  0.0, 1.0)),
        })

    # Decode both layouts:
    if hasattr(nms, "ndim") and nms.ndim == 3:
        # BY CLASS: (C, B, F)
        C, B, F = nms.shape
        for c in _iterify(range(C)):
            for b in _iterify(range(B)):
                _append(c, nms[c, b, :])
    elif hasattr(nms, "ndim") and nms.ndim == 2:
        # Unified: (B, F) possibly with class in last column
        B, F = nms.shape
        cls_col = None
        if F >= 6:
            last = nms[:, -1]
            if np.all((last >= -0.5) & (last < len(class_names) + 0.5)):
                cls_col = -1
        for b in _iterify(range(B)):
            rec = nms[b, :]
            if cls_col is not None:
                c_id = int(round(rec[cls_col]))
                _append(c_id, rec[:5])
            else:
                # No class column; if single-class model, use class 0,
                # otherwise skip (we can't assign class safely).
                if len(class_names) == 1:
                    _append(0, rec[:5])
    else:
        # 1-D row or unknown → try to parse rows of length >= 5
        arr = _safe_asarray(nms)
        if arr.size >= 5:
            row = arr.reshape(-1)[:5]
            if len(class_names) == 1:
                _append(0, row)
    return dets

class _Model:
    def __init__(self, hef_path, device):
        from hailo_platform import HEF
        self.hef_path = hef_path
        self.device = device
        self.cfg = None
        self.runner = None
        self.in_name = None
        self.out_key = None
        self._act = None
        self.input_shape = None  # raw
        self.hwc = (640, 640, 3)
        self.layout = 'HWC'
        self.frame_size_bytes = 640*640*3
        self.frames_count = 1
        # === CODEX PATCH: UB→PIPE auto-fallback BEGIN ===
        self._force_pipe = False  # set True when UB path mismatches and fallback is enabled
        # === CODEX PATCH: UB→PIPE auto-fallback END ===

    def start(self):
        from hailo_platform import HEF
        hef = HEF(self.hef_path)
        # Configure on provided device
        self.cfg = self.device.configure(hef)[0]
        # vstream infos from configured group if available
        get_in = getattr(self.cfg, "get_input_vstream_infos", None)
        get_out = getattr(self.cfg, "get_output_vstream_infos", None)
        in_infos  = get_in()  if callable(get_in)  else hef.get_input_vstream_infos()
        out_infos = get_out() if callable(get_out) else hef.get_output_vstream_infos()
        in_params  = {ii.name: _make_params(ii) for ii in in_infos}
        out_params = {oi.name: _make_params(oi) for oi in out_infos}
        self._in_params = in_params
        self._out_params = out_params
        self._out_infos = out_infos
        # === CODEX PATCH 3: frames_per_infer=1 (UB) BEGIN ===
        # Enforce single-frame per infer on input vstreams for user-buffer path.
        def _force_single_frame_param_ub(name, param):
            # Try attribute then setter (ignore API differences quietly)
            try:
                setattr(param, "frames_per_infer", 1)
            except Exception:
                try:
                    getattr(param, "set_frames_per_infer")(1)
                except Exception:
                    pass
            # Required debug line
            if os.getenv("DEBUG_DERM", "0") == "1":
                try:
                    print(f"[DERM-HAILO] Input '{name}' frames_per_infer set to 1")
                except Exception:
                    pass
        for _name, _p in _iterify(in_params.items()):
            _force_single_frame_param_ub(_name, _p)
        # === CODEX PATCH 3: frames_per_infer=1 (UB) END ===
        # Keep outputs conservative as single-frame where possible too
        for p in _iterify(out_params.values()):
            for attr in _iterify(("frames_count", "frames_per_infer", "batch_size")):
                if hasattr(p, attr):
                    try: setattr(p, attr, 1)
                    except Exception: pass
        self.frames_count = 1
        self.in_name = next(iter(in_params))
        # Capture input shape if available
        try:
            ii0 = next(iter(in_infos))
            shp = getattr(ii0, 'shape', None)
            if shp is None:
                n = getattr(ii0, 'n', 1)
                h = getattr(ii0, 'height', getattr(ii0, 'rows', 0))
                w = getattr(ii0, 'width', getattr(ii0, 'cols', 0))
                c = getattr(ii0, 'channels', 3)
                shp = (n, h, w, c)
            shp = tuple(int(v) for v in shp)
            self.input_shape = shp
            # Canonicalize
            if len(shp)==4:
                n,a,b,d = shp
                if a==3:
                    self.hwc=(b,d,a); self.layout='CHW'
                elif d==3:
                    self.hwc=(a,b,d); self.layout='HWC'
                else:
                    self.hwc=(a,b,d); self.layout='HWC'
            elif len(shp)==3:
                a,b,c=shp
                if a==3:
                    self.hwc=(b,c,a); self.layout='CHW'
                elif c==3:
                    self.hwc=(a,b,c); self.layout='HWC'
                else:
                    self.hwc=(a,b,c); self.layout='HWC'
            else:
                self.hwc=(640,640,3); self.layout='HWC'
            H,W,C=self.hwc
            self.frame_size_bytes=int(H*W*C)
        except Exception:
            self.input_shape = (1,3,640,640)
            self.hwc=(640,640,3)
            self.layout='HWC'
            self.frame_size_bytes=640*640*3
        # Build and enter once; scheduler handles activation internally
        # Force single-frame per infer on UB path to prevent size mismatch
        try:
            for _k,_v in getattr(in_params, 'items', lambda: [])():
                try:
                    setattr(_v, 'frames_per_infer', 1)
                except Exception:
                    try:
                        _v.set_frames_per_infer(1)
                    except Exception:
                        pass
        except Exception:
            pass
        self.runner = InferVStreams(self.cfg, in_params, out_params)
        self.runner.__enter__()

    def stop(self):
        try:
            if self.runner:
                self.runner.__exit__(None, None, None)
        except Exception:
            pass
        self.runner = None
        self.cfg = None
        # Activation is per-infer; nothing to deactivate here

    def infer(self, inp_nhwc_uint8):

        import numpy as np


        # === auto-resize to model input ===

        try:

            import numpy as _np, cv2 as _cv2

            _H,_W,_C = self.hwc

            _src = inp_nhwc_uint8

            if getattr(_src, 'dtype', None) is not None and _src.dtype is not _np.uint8:

                _src = _src.astype(_np.uint8, copy=False)

            if hasattr(_src, 'shape') and len(_src.shape) >= 2 and (_src.shape[0] != _H or _src.shape[1] != _W):

                _src = _cv2.resize(_src, (_W, _H))

            inp_nhwc_uint8 = _src

        except Exception:

            pass

        # === end auto-resize ===

        # ==== Normalize to N=1, uint8, and target (H,W) (3D/4D-safe) ====

        try:

            import os as _os, numpy as _np, cv2 as _cv2

            _H,_W,_C = self.hwc

            _arr = inp_nhwc_uint8

            _dbg = _os.getenv('DEBUG_DERM','0') == '1'

            # ensure uint8

            if getattr(_arr,'dtype',None) is not _np.uint8:

                _arr = _arr.astype(_np.uint8, copy=False)

            # handle 4D (N,H,W,C) and 3D (H,W,C)

            if getattr(_arr,'ndim',0) == 4:

                if _arr.shape[0] != 1:

                    _arr = _arr[:1, ...]

                if _arr.shape[1] != _H or _arr.shape[2] != _W:

                    _arr = _cv2.resize(_arr[0], (_W, _H))[_np.newaxis, ...]

            elif getattr(_arr,'ndim',0) == 3:

                if _arr.shape[0] != _H or _arr.shape[1] != _W:

                    _arr = _cv2.resize(_arr, (_W, _H))

                _arr = _arr[_np.newaxis, ...]

            else:

                _arr = _np.zeros((1,_H,_W,_C), dtype=_np.uint8)

            inp_nhwc_uint8 = _arr

            if (_dbg is not None and (getattr(_dbg,"size",0)>0 if hasattr(_dbg,"__array__") else bool(_dbg))):

                try:

                    print(f"[DERM-HAILO] {self.in_name} normalized", getattr(inp_nhwc_uint8,'shape',None), getattr(inp_nhwc_uint8,'dtype',None))

                except Exception: pass

        except Exception:

            pass

        # ==== end normalization ====


        # Ensure N=1 batch and uint8 for User-Buffer API

        try:

            import numpy as _np

            _arr = inp_nhwc_uint8

            if getattr(_arr, 'ndim', 0) == 3:

                _arr = _arr[_np.newaxis, ...]  # (1,H,W,C)

            if getattr(_arr, 'dtype', None) is not _np.uint8:

                _arr = _arr.astype(_np.uint8, copy=False)

            inp_nhwc_uint8 = _arr

        except Exception:

            pass

        # === CODEX PATCH 2: PIPE-only mode + watchdog BEGIN ===
        # Optional bypass using pipe.infer with single-frame batch and watchdog
        if os.getenv("DERM_HAILO_PIPE_INFER", "0") == "1" or getattr(self, "_force_pipe", False):
            if os.getenv("DEBUG_DERM", "0") and not getattr(self, "_pipe_log_once", False):
                print("[DERM-HAILO] PIPE infer path enabled (DERM_HAILO_PIPE_INFER=1)")
                self._pipe_log_once = True
            # Resolve classes dynamically to avoid strict imports
            InputVStreamParams = _get_cls("InputVStreamParams")
            OutputVStreamParams = _get_cls("OutputVStreamParams")
            if InputVStreamParams and OutputVStreamParams:
                # Build params from configured group infos
                get_in = getattr(self.cfg, "get_input_vstream_infos", None)
                get_out = getattr(self.cfg, "get_output_vstream_infos", None)
                in_infos  = get_in()  if callable(get_in)  else []
                out_infos = get_out() if callable(get_out) else []
                try:
                    ivp = InputVStreamParams.make(self.cfg, in_infos)
                    ovp = OutputVStreamParams.make(self.cfg, out_infos)
                    in_name  = next(iter(ivp.keys()))
                    out_name = next(iter(ovp.keys()))
                    # Force single-frame on the input param (best-effort)
                    try:
                        setattr(ivp[in_name], "frames_per_infer", 1)
                    except Exception:
                        try:
                            getattr(ivp[in_name], "set_frames_per_infer")(1)
                        except Exception:
                            pass
                    if os.getenv("DEBUG_DERM", "0") and not getattr(self, "_pipe_names_logged", False):
                        try:
                            print(f"[DERM-HAILO] PIPE in='{in_name}' out='{out_name}'")
                        except Exception:
                            pass
                        self._pipe_names_logged = True
                    # Build 1-frame batch
                    batch = inp_nhwc_uint8[np.newaxis, ...].astype(np.uint8, copy=False)
                    # Run infer on a worker thread with timeout
                    import threading
                    result_holder = {}
                    err_holder = {}
                    act = getattr(self.cfg, 'activate', None)
                    # Create contexts on main thread for safe cleanup
                    ctxs = []
                    if callable(act):
                        ctxs.append(self.cfg.activate())
                    ctxs.append(InferVStreams(self.cfg, ivp, ovp, tf_nms_format=False))
                    mgrs = []
                    pipe = None
                    try:
                        for cm in _iterify(ctxs):
                            mgrs.append(cm.__enter__())
                        pipe = mgrs[-1]
                        def worker():
                            try:
                                outputs = pipe.infer({in_name: batch})
                                result_holder['out'] = outputs[out_name]
                            except Exception as e:
                                err_holder['err'] = e
                        t = threading.Thread(target=worker, daemon=True)
                        t.start()
                        t.join(2.0)
                        if t.is_alive():
                            raise RuntimeError("PIPE infer timeout")
                        if 'err' in err_holder:
                            raise err_holder['err']
                        if 'out' in result_holder:
                            return result_holder['out']
                    finally:
                        for cm in _iterify(reversed(ctxs)):
                            try:
                                cm.__exit__(None, None, None)
                            except Exception:
                                pass
                except Exception:
                    # Fallback to existing runner path on any error
                    pass
        # === CODEX PATCH 2: PIPE-only mode + watchdog END ===
        # Default path: use persistent runner with user-buffers
        act = getattr(self.cfg, 'activate', None)
        try:
            if callable(act):
                with self.cfg.activate():
                    out = self.runner.infer({self.in_name: inp_nhwc_uint8})
            else:
                out = self.runner.infer({self.in_name: inp_nhwc_uint8})
            try:
                for k, v in _iterify(out.items()):
                    f = _flat_floats(v)
                    if f.size:
                        print(f"[HAILO OUT] {k} shape={(f.shape,)} dtype=float32")
                    else:
                        print(f"[HAILO OUT] {k} shape=<no-shape> dtype=None")
                    sample = " ".join(f"{x:.3f}" for x in f[:15]) if f.size else ""
                    print(f"[HAILO OUT SAMPLE] {k}: {sample}")
            except Exception:
                pass
        except Exception as e:
            # === CODEX PATCH: UB→PIPE auto-fallback BEGIN ===
            try:
                auto_fb = os.getenv("DERM_AUTO_FALLBACK", "1") != "0"
            except Exception:
                auto_fb = True
            msg = str(e)
            if auto_fb and ("Memory size of vstream" in msg and "does not match the frame count" in msg):
                if not getattr(self, "_force_pipe", False):
                    print("[DERM-HAILO] UB path vstream mismatch — auto-switching to PIPE for this session")
                self._force_pipe = True
                # Re-run through PIPE path immediately
                return self.infer(inp_nhwc_uint8)
            # Re-raise original exception if not a mismatch or fallback disabled
            raise
            # === CODEX PATCH: UB→PIPE auto-fallback END ===
        if self.out_key is None:
            self.out_key = next(iter(out))
        return out[self.out_key]
        # === CODEX PATCH: single-frame + optional pipe.infer END ===


# === D5K FALLBACK DECODER (robust to Hailo list outputs) ===
def _np_try_all(x):
    import numpy as _np
    # Try straight array
    try:
        a = _np.asarray(x, dtype=_np.float32)
        if a.size: return a
    except Exception:
        pass
    # Try .to_numpy()
    try:
        a = x.to_numpy()
        a = _np.asarray(a, dtype=_np.float32)
        if a.size: return a
    except Exception:
        pass
    # Try .buffer / .data
    try:
        buf = getattr(x, 'buffer', None) or getattr(x, 'data', None)
        if buf is not None:
            a = _np.frombuffer(buf, dtype=_np.float32)
            if a.size: return a
    except Exception:
        pass
    # Try memoryview
    try:
        mv = memoryview(x)
        a = _np.frombuffer(mv, dtype=_np.float32)
        if a.size: return a
    except Exception:
        pass
    # Try .tolist()
    try:
        a = _np.asarray(x.tolist(), dtype=_np.float32)
        if a.size: return a
    except Exception:
        pass
    # Try iter
    try:
        a = _np.asarray(list(x), dtype=_np.float32)
        if a.size: return a
    except Exception:
        pass
    return None

def _np_coerce_first(obj):
    # Hailo tends to return a list; take first element
    if isinstance(obj, (list, tuple)) and len(obj)>0:
        cand = obj[0]
    else:
        cand = obj
    return _np_try_all(cand)

def _reshape_or_none(a, shape_hint):
    import numpy as _np
    if a is None: return None
    if (shape_hint is not None and (getattr(shape_hint,"size",0)>0 if hasattr(shape_hint,"__array__") else bool(shape_hint))):
        want = 1
        for v in shape_hint: 
            try: want *= int(v)
            except: return a
        if a.size == want:
            try: return a.reshape(tuple(int(v) for v in shape_hint))
            except Exception: return a
    return a

def _decode_d5k_array(a, cid=0, style='cxcywh', conf=0.25):
    import numpy as _np
    a = _np.asarray(a, dtype=_np.float32)
    if a.ndim!=3 or a.shape[1]<5:
        return _np.zeros((0,6), _np.float32)
    D,M,K = a.shape
    try:
        scores = a[:,4,:]                # (D,K)
        cls    = _np.argmax(scores, axis=1).astype(_np.int64, copy=False)
        confs  = scores[_np.arange(D), cls]
    except Exception:
        return _np.zeros((0,6), _np.float32)
    keep = confs >= float(conf)
    if not _np.any(keep):
        return _np.zeros((0,6), _np.float32)
    idx   = _np.arange(D)[keep]
    cls   = cls[keep]
    confs = confs[keep].astype(_np.float32, copy=False)
    x0 = a[idx,0,cls]; y0 = a[idx,1,cls]; x1 = a[idx,2,cls]; y1 = a[idx,3,cls]
    if style == 'xyxy':                 # SSD coords
        cx = (x0 + x1) * 0.5; cy = (y0 + y1) * 0.5
        ww = (x1 - x0);     hh = (y1 - y0)
    else:                               # YOLO cx,cy,w,h
        cx,cy,ww,hh = x0,y0,x1,y1
    dets = _np.stack([cx,cy,ww,hh,confs,_np.full_like(confs, float(cid))], axis=1).astype(_np.float32, copy=False)
    dets[:,:4] = _np.clip(dets[:,:4], 0.0, 1.0)
    return dets

def _decode_any_d5k(obj, shape_hint=None, cid=0, style='cxcywh', conf=0.25):
    a = _np_coerce_first(obj)
    a = _reshape_or_none(a, shape_hint)
    try:
        return _decode_d5k_array(a, cid=cid, style=style, conf=conf)
    except Exception:
        import numpy as _np
        return _np.zeros((0,6), _np.float32)

# === PADDED COERCION (ragged list -> (D,5,K) float32) ===
def _first_shape_from_infos(infos):
    for oi in (infos or []):
        sh = getattr(oi, 'shape', None)
        if (sh is not None and (getattr(sh,"size",0)>0 if hasattr(sh,"__array__") else bool(sh))): 
            try: return [int(v) for v in sh]
            except: return sh
    return None

def _to_array_padded(obj, shape_hint):
    import numpy as _np
    if not shape_hint or len(shape_hint) < 3:
        # Try best-effort flatten
        try:
            return _np.asarray(obj, dtype=_np.float32)
        except Exception:
            return _np.zeros((0,0,0), _np.float32)
    D, M, K = int(shape_hint[0]), int(shape_hint[1]), int(shape_hint[2])
    arr = _np.zeros((D, M, K), dtype=_np.float32)
    # Expect obj ~ list length D; each item ~ list length M; each channel ~ list/array length K (ragged allowed)
    try:
        for i in range(min(D, len(obj) if hasattr(obj,'__len__') else 0)):
            li = obj[i]
            if not hasattr(li, '__len__'): 
                continue
            for m in range(min(M, len(li))):
                ch = li[m]
                try:
                    v = _np.asarray(ch, dtype=_np.float32).ravel()
                    L = min(K, v.size)
                    if L > 0:
                        arr[i, m, :L] = v[:L]
                except Exception:
                    # leave zeros
                    pass
        return arr
    except Exception:
        return arr

def _decode_from_padded(a, cid=0, style='cxcywh', conf=0.25):
    import numpy as _np
    a = _np.asarray(a, dtype=_np.float32)
    if a.ndim != 3 or a.shape[1] < 5:
        return _np.zeros((0,6), _np.float32)
    D, M, K = a.shape
    # Per-detection best class by channel-4 score
    scores = a[:,4,:]                              # (D,K)
    cls    = _np.argmax(scores, axis=1).astype(_np.int64, copy=False)
    confs  = scores[_np.arange(D), cls]
    keep   = confs >= float(conf)
    if not _np.any(keep): 
        return _np.zeros((0,6), _np.float32)
    idx   = _np.arange(D)[keep]
    cls   = cls[keep]
    confs = confs[keep].astype(_np.float32, copy=False)
    x0 = a[idx,0,cls]; y0 = a[idx,1,cls]; x1 = a[idx,2,cls]; y1 = a[idx,3,cls]
    if style == 'xyxy':                            # SSD coords
        cx = (x0 + x1) * 0.5; cy = (y0 + y1) * 0.5
        ww = (x1 - x0);     hh = (y1 - y0)
    else:                                          # YOLO cx,cy,w,h
        cx, cy, ww, hh = x0, y0, x1, y1
    dets = _np.stack([cx, cy, ww, hh, confs, _np.full_like(confs, float(cid))], axis=1).astype(_np.float32, copy=False)
    dets[:,:4] = _np.clip(dets[:,:4], 0.0, 1.0)
    return dets

class DualHefEngine:
    def __init__(self, hef_lesion, hef_acne, device=None):
        self.hef_lesion = hef_lesion
        self.hef_acne   = hef_acne
        self.device = device
        self._owns_device = False if device is not None else False
        self.m_lesion = None
        self.m_acne   = None

    def start(self):
        # Use shared VDevice if none provided
        if self.device is None:
            self.device = device_manager().get_device()
            self._owns_device = False
        # Configure persistent runners
        self.m_lesion = _Model(self.hef_lesion, self.device)
        self.m_acne   = _Model(self.hef_acne,   self.device)
        self.m_lesion.start()
        self.m_acne.start()
        print(f"[DERM-HAILO] Engine ON -> lesion={self.hef_lesion} acne={self.hef_acne}")
        if os.getenv("DEBUG_DERM", "0") == "1":
            lh,lw,lc=self.m_lesion.hwc; ah,aw,ac=self.m_acne.hwc
            print(f"[DERM-HAILO] Inputs -> lesion:{self.m_lesion.in_name} HWC=({lh},{lw},{lc}) frame_size={self.m_lesion.frame_size_bytes} frames_count=1")
            print(f"[DERM-HAILO] Inputs -> acne:{self.m_acne.in_name} HWC=({ah},{aw},{ac}) frame_size={self.m_acne.frame_size_bytes} frames_count=1")
            outs_les = ", ".join([f"{oi.name} {getattr(oi,'shape','')}" for oi in (self.m_lesion._out_infos or [])])
            outs_acn = ", ".join([f"{oi.name} {getattr(oi,'shape','')}" for oi in (self.m_acne._out_infos or [])])
            print(f"[DERM-HAILO] Outputs -> lesion:{outs_les}")
            print(f"[DERM-HAILO] Outputs -> acne:{outs_acn}")
            in_user, in_dma = _buf_flags(next(iter(self.m_lesion._in_params.values())))
            out_user, out_dma = _buf_flags(next(iter(self.m_lesion._out_params.values())))
            print(f"[DERM-HAILO] BufferMode -> in:{{is_user_buffer={in_user}, is_dma={in_dma}}} out:{{is_user_buffer={out_user}, is_dma={out_dma}}}")

    def stop(self):
        try:
            if self.m_lesion: self.m_lesion.stop()
            if self.m_acne:   self.m_acne.stop()
        finally:
            # Do not exit shared device unless we own it (we don't by default)
            if self._owns_device and self.device:
                try: self.device.__exit__(None,None,None)
                except Exception: pass
                self.device = None
        self.m_lesion = self.m_acne = None

    def process_frame(self, frame_bgr, conf_thres=0.25, debug=False, roi=None):
        assert self.device, "call start() first"
        H, W = frame_bgr.shape[:2]
        # Allow bypassing ROI via env flag
        if (DERM_BYPASS_ROI is not None and (getattr(DERM_BYPASS_ROI,"size",0)>0 if hasattr(DERM_BYPASS_ROI,"__array__") else bool(DERM_BYPASS_ROI))):
            roi = None
        if roi is not None:
            # Accept either (x1,y1,x2,y2) or (x,y,w,h) — and be robust if roi is scalar/invalid.
            try:
                r = list(np.asarray(roi).reshape(-1).tolist())
            except Exception:
                r = []
            if len(r) >= 4:
                x1, y1, x2, y2 = [int(v) for v in r[:4]]
                if x2 <= x1 or y2 <= y1:
                    # Interpret as (x,y,w,h)
                    x, y, w, h = x1, y1, x2, y2
                    x1, y1, x2, y2 = x, y, x + w, y + h
                x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    roi = None
            else:
                roi = None
        if roi is None:
            img640, (x0,y0,s) = _center_square_rgb(frame_bgr)
            inp = img640[None, ...]
            if inp.dtype != np.uint8 or not inp.flags['C_CONTIGUOUS']:
                inp = np.ascontiguousarray(inp, dtype=np.uint8)
            # Run both models
            out1 = self.m_lesion.infer(inp)
            out2 = self.m_acne.infer(inp)

            # Try robust D5K fallback first (for (D,5,K) list outputs)
            les_shape = None
            acn_shape = None
            try:
                les_shape = tuple(getattr(next(iter(self.m_lesion._out_infos)), "shape", ()))
            except Exception:
                pass
            try:
                acn_shape = tuple(getattr(next(iter(self.m_acne._out_infos)), "shape", ()))
            except Exception:
                pass
            det_f1 = _decode_any_d5k(out1, shape_hint=les_shape, cid=0, style='xyxy',  conf=float(conf_thres))  # SSD
            det_f2 = _decode_any_d5k(out2, shape_hint=acn_shape, cid=1, style='cxcywh', conf=float(conf_thres)) # YOLO
            det_all = None
            if isinstance(det_f1, np.ndarray) and det_f1.size:
                det_all = det_f1
            if isinstance(det_f2, np.ndarray) and det_f2.size:
                det_all = det_f2 if det_all is None else np.vstack([det_all, det_f2])

            # If fallback produced nothing, use existing unified decoder
            # Unified decode to normalized (M,6) then to pixel xyxy (M,6)
            out_dict = {"lesion": _to_listish(out1), "acne": _to_listish(out2)}
            if det_all is None or not isinstance(det_all, np.ndarray) or det_all.size == 0:
                det_all = _decode_hailo_dual_nms(out_dict)
# Inject a synthetic box if requested and nothing detected
            if DERM_FAKE and (not isinstance(det_all, np.ndarray) or det_all.size == 0):
                det_all = np.asarray([[0.5,0.5,0.3,0.3,0.6,0.0]], dtype=np.float32)
            # periodic compact debug
            if getattr(self, "_dbg_ctr", 0) < 10 or (getattr(self, "_dbg_ctr", 0) % 120 == 0):
                samp = det_all[0].tolist() if isinstance(det_all, np.ndarray) and det_all.size else None
                print(f"[DERM-HAILO] det_all: shape={getattr(det_all,'shape',None)} sample={samp} conf>={CONF_THRESH:.2f}", flush=True)
            self._dbg_ctr = getattr(self, "_dbg_ctr", 0) + 1

            boxes_xyxy = _to_pixel_xyxy(det_all, PREPROC_W, PREPROC_H)


            # [DINO_DRAW] Use professional, sticky markers with tracker

            try:

                import ai_features.dermatologist.professional_derma_markings as _marks

                # -- build full Nx6 boxes and use BGR tile for drawing --
                try:
                    import numpy as _np
                    _boxes6 = None
                    if isinstance(boxes_xyxy, _np.ndarray) and isinstance(det_all, _np.ndarray) and boxes_xyxy.shape[0]==det_all.shape[0]:
                        _boxes6 = _np.zeros((boxes_xyxy.shape[0], 6), _np.float32)
                        _boxes6[:, :4] = boxes_xyxy.astype(_np.float32)
                        _boxes6[:, 4]  = det_all[:, 4].astype(_np.float32)
                        _boxes6[:, 5]  = det_all[:, 5].astype(_np.float32)
                    else:
                        _boxes6 = boxes_xyxy
                except Exception:
                    _boxes6 = boxes_xyxy
                try:
                    _img_bgr = cv2.cvtColor(img640, cv2.COLOR_RGB2BGR)
                except Exception:
                    _img_bgr = img640
                _marks.draw_markers_dino_style(_img_bgr, _boxes6, conf_thresh=float(conf_thres))

                # [FORCE_BOXES] Draw classic rectangles + confidence and fix counters
                try:
                    import numpy as _np, cv2 as _cv2
                    _arr = _np.asarray(_boxes6, dtype=_np.float32)
                    if _arr.ndim==2 and _arr.shape[1]>=6 and _arr.size:
                        for _r in _arr:
                            _x1,_y1,_x2,_y2 = int(_r[0]), int(_r[1]), int(_r[2]), int(_r[3])
                            _conf = float(_r[4]); _cid = int(_r[5]+0.5)
                            _col = (0,255,0) if _cid==0 else (0,165,255)
                            _cv2.rectangle(_img_bgr, (_x1,_y1), (_x2,_y2), _col, 2)
                            _cv2.putText(_img_bgr, f"{_conf:.2f}", (_x1, max(0,_y1-6)), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, _col, 1, _cv2.LINE_AA)
                        # Ensure lesion/acne counters are plain ints for the HUD
                        try:
                            _les = int((_arr[:,5].round().astype(_np.int32)==0).sum())
                            _acn = int((_arr[:,5].round().astype(_np.int32)==5).sum())
                            if isinstance(counts, dict):
                                counts['lesion'] = _les
                                counts['acne'] = _acn
                        except Exception:
                            pass
                except Exception:
                    pass
                # [FALLBACK] Draw tiny anchor dots if nothing rendered by the pro drawer
                try:
                    import numpy as _np
                    _arr = _np.asarray(_boxes6, dtype=_np.float32)
                    if _arr.ndim == 2 and _arr.shape[1] >= 4:
                        _thr = float(conf_thres) if conf_thres is not None else 0.25
                        _drawn = 0
                        # sort by confidence if available
                        _idx = _np.arange(_arr.shape[0])
                        if _arr.shape[1] >= 5:
                            _idx = _np.argsort(-_arr[:,4])
                        for _k in _idx:
                            _r = _arr[int(_k)]
                            if _arr.shape[1] >= 5 and float(_r[4]) < _thr:
                                continue
                            _x1,_y1,_x2,_y2 = [int(v) for v in _r[:4]]
                            _cx = (_x1 + _x2) // 2; _cy = (_y1 + _y2) // 2
                            cv2.circle(_img_bgr, (_cx,_cy), 2, (0,255,255), -1)
                            _drawn += 1
                            if _drawn >= 12:
                                break
                except Exception:
                    pass

            except Exception:

                pass

            # Draw markers onto the center-square by pasting the annotated 640x640

            vis = frame_bgr.copy()

            vis[y0:y0+s, x0:x0+s] = cv2.resize(_img_bgr, (s, s), interpolation=cv2.INTER_LINEAR)


            # Return a clean, GUI-friendly payload

            # Convert preproc (640x640) boxes to FULL-FRAME XYXY int32 for the GUI

            try:

                import numpy as _np

                if isinstance(boxes_xyxy, _np.ndarray):

                    if boxes_xyxy.size:

                        _bf = boxes_xyxy.astype(_np.float32, copy=True)  # [x1,y1,x2,y2,conf,cls] in 640x640

                        _bf[:,[0,2]] = _bf[:,[0,2]] * float(scale) + float(x0)

                        _bf[:,[1,3]] = _bf[:,[1,3]] * float(scale) + float(y0)

                    else:

                        _bf = _np.zeros((0,6), _np.float32)

                else:

                    _bf = _np.zeros((0,6), _np.float32)

                if _bf.ndim == 2 and _bf.shape[1] >= 4:

                    _boxes4 = _bf[:, :4].astype(_np.int32, copy=False)

                    _scores = (_bf[:, 4] if _bf.shape[1] > 4 else _np.ones((_boxes4.shape[0],), _np.float32))

                    _classes = (_bf[:, 5].astype(_np.int32, copy=False) if _bf.shape[1] > 5 else _np.zeros((_boxes4.shape[0],), _np.int32))

                else:

                    _boxes4 = _np.zeros((0,4), _np.int32)

                    _scores = _np.zeros((0,), _np.float32)

                    _classes = _np.zeros((0,), _np.int32)

            except Exception:

                # absolute fallback

                _boxes4 = _np.zeros((0,4), _np.int32) if 'np' in globals() else []

                _scores = _np.zeros((0,), _np.float32) if 'np' in globals() else []

                _classes = _np.zeros((0,), _np.int32) if 'np' in globals() else []

            

            counts = {

                "lesion": int((_classes == 0).sum()) if hasattr(_classes, "sum") else 0,

                "acne":   int((_classes == 1).sum()) if hasattr(_classes, "sum") else 0,

                "total":  int(_boxes4.shape[0]) if hasattr(_boxes4, "shape") else 0,

                "boxes":  _boxes4,          # <- FULL-FRAME XYXY int32 (GUI-friendly)

                "scores": _scores,          # optional

                "classes": _classes,        # optional (0=lesion,1=acne)

                "dets":   det_all.astype(_np.float32, copy=False) if hasattr(det_all, "astype") else _np.zeros((0,6), _np.float32),

            }

            if os.getenv("DEBUG_DERM","0") == "1":

                try:

                    print(f"[DERM-HAILO] GUI payload -> boxes={getattr(_boxes4,'shape',None)} scores={getattr(_scores,'shape',None)} classes={getattr(_classes,'shape',None)}")

                    # === PATCH: promote GUI payload into meta for downstream ===
                    try:
                        import numpy as _np
                        if isinstance(meta, dict):
                            _bx = _np.asarray(boxes,   _np.int32)
                            _sc = _np.asarray(scores,  _np.float32).reshape(-1)
                            _cl = _np.asarray(classes, _np.int32).reshape(-1)
                            n = int(_bx.shape[0]) if _bx.ndim==2 else 0
                            _sc = _sc[:n]; _cl = _cl[:n]
                            meta['boxes']   = _bx
                            meta['scores']  = _sc
                            meta['classes'] = _cl
                    except Exception:
                        pass


                except Exception:

                    pass

            # [COUNTS_FIX] Ensure plain-int lesion/acne counters from classes (env-mappable)
            try:
                import os as _os, numpy as _np
                if isinstance(counts, dict):
                    # Prefer existing classes list, else derive from dets/boxes
                    if 'classes' in counts:
                        _cls = _np.asarray(counts['classes']).astype(_np.float32)
                    elif 'dets' in counts:
                        _d = _np.asarray(counts['dets'], _np.float32)
                        _cls = _d[:,5] if (_d.ndim==2 and _d.shape[1]>=6) else _np.array([], _np.float32)
                    elif 'boxes' in counts:
                        _b = _np.asarray(counts['boxes'], _np.float32)
                        _cls = _b[:,5] if (_b.ndim==2 and _b.shape[1]>=6) else _np.array([], _np.float32)
                    else:
                        _cls = _np.array([], _np.float32)
                    _cls_i = _np.rint(_cls).astype(_np.int32, copy=False)
                    def _parse_set(s, default):
                        try:
                            raw = _os.getenv(s, default)
                            vals = [v.strip() for v in str(raw).split(',') if v.strip()!='']
                            return set(int(v) for v in vals)
                        except Exception:
                            return set(int(default)) if str(default).isdigit() else set()
                    _lesion_ids = _parse_set('DERM_LESION_CIDS', '0')
                    _acne_ids   = _parse_set('DERM_ACNE_CIDS',   '5')
                    counts['lesion'] = int(_np.isin(_cls_i, list(_lesion_ids)).sum())
                    counts['acne']   = int(_np.isin(_cls_i, list(_acne_ids)).sum())
            except Exception:
                pass
            return vis, counts
        else:
            # ROI path
            crop = frame_bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img640 = cv2.resize(rgb, (640,640)).astype(np.uint8)
            if not img640.flags['C_CONTIGUOUS']:
                img640 = np.ascontiguousarray(img640)
            inp = img640[None, ...]
            if (debug is not None and (getattr(debug,"size",0)>0 if hasattr(debug,"__array__") else bool(debug))):
                print("Input tensor -> shape:", inp.shape, "dtype:", inp.dtype, "contig:", inp.flags['C_CONTIGUOUS'])
            out1 = self.m_lesion.infer(inp)
            out2 = self.m_acne.infer(inp)
            # ROI unified decode via robust helpers
            
            # Robust, shape-aware decode (handles ragged Hailo lists)
            import numpy as _np
            s1 = _first_shape_from_infos(getattr(self.m_lesion, "_out_infos", None))
            s2 = _first_shape_from_infos(getattr(self.m_acne,   "_out_infos", None))
            a1 = _to_array_padded(out1, s1)
            a2 = _to_array_padded(out2, s2)
            det_les = _decode_from_padded(a1, cid=0, style='xyxy',   conf=float(os.getenv('CONF_LESION','0.05')))
            det_acn = _decode_from_padded(a2, cid=1, style='cxcywh', conf=float(os.getenv('CONF_ACNE','0.05')))
            det_all = _np.vstack([det_les, det_acn]) if (det_les.size or det_acn.size) else np.zeros((0,6), np.float32)

            if DERM_FAKE and (not isinstance(det_all, np.ndarray) or det_all.size == 0):
                det_all = np.asarray([[0.5,0.5,0.3,0.3,0.6,0.0]], dtype=np.float32)
            boxes_xyxy = _to_pixel_xyxy(det_all, PREPROC_W, PREPROC_H)
            # Build unified detections (M,6) [cx,cy,w,h,conf,cls]
            if isinstance(det_all, np.ndarray) and det_all.ndim == 2 and det_all.shape[1] == 6:
                det_all = det_all.astype(np.float32, copy=False)
            else:
                det_all = _np.empty((0, 6), dtype=np.float32)

            # Convert to pixel xyxy for drawing / GUI
            boxes_xyxy = _to_pixel_xyxy(det_all, PREPROC_W, PREPROC_H)
            if det_all.ndim != 2 or det_all.shape[1] != 6:
                det_all = _np.zeros((0,6), np.float32)
            det_all = _np.nan_to_num(det_all, nan=0.0, posinf=0.0, neginf=0.0)
            det_all = _np.ascontiguousarray(det_all, dtype=np.float32)
            # Filter by CONF_THRESH (env-configured)
            if det_all.size:
                det_all = det_all[det_all[:,4] >= float(CONF_THRESH)]
                # Optional compact logs
                try:
                    for row in _iterify(det_all):
                        cid = int(row[5]); nm = 'lesion' if cid==0 else 'acne'
                        _safe_print(f"[DECODE] id={cid} name={nm} conf={row[4]:.2f} xywh_norm={row[0]:.3f},{row[1]:.3f},{row[2]:.3f},{row[3]:.3f}")
                except Exception:
                    pass
            # removed fragile debug loop over `dets2` (not guaranteed to exist / iterable)
            vis = frame_bgr.copy()
            crop_h, crop_w = crop.shape[:2]
            sx = crop_w / 640.0
            sy = crop_h / 640.0
            # Draw directly from unified pixel-space boxes
            for row in _iterify((boxes_xyxy if isinstance(boxes_xyxy, np.ndarray) else np.empty((0,6), np.float32))):
                r = np.asarray(row).reshape(-1)
                x1p, y1p, x2p, y2p = [float(v) for v in r[:4]]
                conf = float(r[4]) if r.size > 4 else 1.0
                cid  = int(r[5])   if r.size > 5 else 0
                gx1 = int(x1 + x1p * sx); gy1 = int(y1 + y1p * sy)
                gx2 = int(x1 + x2p * sx); gy2 = int(y1 + y2p * sy)
                color = (0,255,0) if cid==0 else (0,165,255)
                cv2.rectangle(vis, (gx1,gy1), (gx2,gy2), color, 2)
                cv2.putText(vis, f"{float(conf):.2f}", (gx1, max(0,gy1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Convert normalized dets to pixel xyxy in preproc (640x640) for GUI
            H = 640; W = 640
            if det_all.size:
                cx, cy, w, h = det_all[:,0], det_all[:,1], det_all[:,2], det_all[:,3]
                x1 = (cx - 0.5*w) * W
                y1 = (cy - 0.5*h) * H
                x2 = (cx + 0.5*w) * W
                y2 = (cy + 0.5*h) * H
                boxes_xyxy = np.stack([x1, y1, x2, y2, det_all[:,4], det_all[:,5]], axis=1).astype(np.float32, copy=False)
            else:
                boxes_xyxy = np.zeros((0,6), dtype=np.float32)
            return vis, {"lesion": int((det_all[:,5] == 0).sum()), "acne": int((det_all[:,5] == 1).sum()), "total": int(boxes_xyxy.shape[0]), "boxes": boxes_xyxy, "dets": det_all}

    def infer_derm(self, frame_bgr, roi=None, conf_acne=0.35, conf_lesion=0.35):
        # ROI required; return quietly if missing
        if roi is None:
            return (None, None)
        try:
            x, y, w, h = [int(v) for v in roi]
        except Exception:
            # Try convert (x1,y1,x2,y2) to (x,y,w,h)
            try:
                x1, y1, x2, y2 = [int(v) for v in roi]
                x, y, w, h = x1, y1, max(0, x2 - x1), max(0, y2 - y1)
            except Exception:
                return (None, None)
        if w <= 0 or h <= 0:
            return (None, None)
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            return (None, None)
        try:
            ih, iw, ic = self.m_lesion.hwc
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img, scale, pad_x, pad_y = _letterbox(rgb, (ih, iw))
            arr = np.ascontiguousarray(img.transpose(2,0,1)) if self.m_lesion.layout=='CHW' else np.ascontiguousarray(img)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            bytes_written = int(arr.nbytes)
            need = int(self.m_lesion.frame_size_bytes)
            if bytes_written != need or not arr.flags['C_CONTIGUOUS']:
                if os.getenv('DEBUG_DERM','0')=='1':
                    now=time.time(); key='_last_sz_mismatch'
                    if not hasattr(self, key) or now-getattr(self, key,0)>1.0:
                        print(f"[DERM-HAILO] INPUT_SIZE_MISMATCH -> expected={need} got={bytes_written} layout={self.m_lesion.layout} shape={arr.shape} dtype={arr.dtype}")
                        setattr(self, key, now)
                return (None, 'PAUSE')
            if os.getenv('DEBUG_DERM','0')=='1':
                now=time.time(); key='_last_preproc'
                if not hasattr(self, key) or now-getattr(self, key,0)>1.0:
                    print(f"[DERM-HAILO] Preproc -> layout={self.m_lesion.layout} HxW={ih}x{iw} bytes={bytes_written} contiguous={int(arr.flags['C_CONTIGUOUS'])} dtype=uint8")
                    setattr(self, key, now)
            out_les = self.m_lesion.infer(arr)
            out_acn = self.m_acne.infer(arr)
            # Normalize list-based NMS if present to avoid ragged arrays
            if isinstance(out_les, (list, tuple)):
                dets_arr_les = _nms_list_to_dets(out_les, conf_thresh=min(conf_lesion, 0.99))
            else:
                dets_arr_les = np.zeros((0,6), dtype=np.float32)
            if isinstance(out_acn, (list, tuple)):
                dets_arr_acn = _nms_list_to_dets(out_acn, conf_thresh=min(conf_acne, 0.99))
            else:
                dets_arr_acn = np.zeros((0,6), dtype=np.float32)
            # Fallback to legacy parse if arrays empty
            if dets_arr_les.size == 0:
                dets_les = _nms(_nms_parse_bridge(out_les, conf_lesion))
            else:
                dets_les = []
            if dets_arr_acn.size == 0:
                dets_acn = _nms(_nms_parse_bridge(out_acn, conf_acne))
            else:
                dets_acn = []
            # Build ROI visualization at model input size (BGR) with 50% opacity markers
            vis_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            overlay = vis_bgr.copy()
            for d in _iterify(dets_les):
                x1 = int(d["x1"]); y1 = int(d["y1"]); x2 = int(d["x2"]); y2 = int(d["y2"])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for d in _iterify(dets_acn):
                x1 = int(d["x1"]); y1 = int(d["y1"]); x2 = int(d["x2"]); y2 = int(d["y2"])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
            vis = cv2.addWeighted(overlay, 0.5, vis_bgr, 0.5, 0.0)
            counts = {"acne": int(len(dets_acn)), "lesion": int(len(dets_les))}
            if os.getenv("DEBUG_DERM","0") == "1":
                if not hasattr(self, "_last_log"):
                    self._last_log = 0.0
                now = time.time()
                if now - self._last_log > 1.0:
                    print(f"[DERM-HAILO] Detections -> acne:{counts['acne']} lesion:{counts['lesion']}")
                    self._last_log = now
            return (vis, counts)
        except Exception as e:
            if os.getenv("DEBUG_DERM", "0") == "1":
                print(f"[DERM-HAILO] ERROR -> {type(e).__name__}: {e}")
            return (None, 'PAUSE')

    # Backwards-compatible ROI API as requested by GUI integration
    def infer_roi(self, frame_bgr, roi, conf_acne=0.35, conf_lesion=0.35, **kwargs):
        res = self.infer_derm(frame_bgr, roi=roi, conf_acne=conf_acne, conf_lesion=conf_lesion)
        vis, counts = res
        if counts == 'PAUSE' or vis is None:
            return (None, counts)
        # Ensure ROI-sized visualization (convert model-input sized vis to ROI size if needed)
        try:
            x, y, w, h = [int(v) for v in roi]
        except Exception:
            return (vis, counts)
        if vis.shape[0] != h or vis.shape[1] != w:
            vis = cv2.resize(vis, (w, h))
        return (vis, counts)


# === CODEX PATCH #2: Worker SHM→infer loop BEGIN ===
def run_worker_shm_loop(control_q, result_q, hb_q,
                        shm_name: str = 'derm_frame_v1', frame_size: int = 1228800,
                        meta_shm_name: str = 'derm_meta_v1',
                        hef_lesion: str | None = None, hef_acne: str | None = None):
    """Spawn-process worker loop reading frames from named SHM and posting results.

    - Attaches to derm_frame_v1 (pixels) and derm_meta_v1 (seq,flags)
    - Polls little-endian uint32 seq; on change, infers exactly one frame
    - Posts results into result_q for the GUI pump to draw
    - Keeps heartbeats via hb_q
    """
    # Attach SHMs
    shm = meta = None
    if _shm is not None and shm_name:
        try:
            shm = _shm.SharedMemory(name=shm_name, create=False)
            print(f"[DERM-HAILO] WorkerProc: attached SHM name={shm_name} size={int(frame_size)}")
        except Exception:
            shm = None
    if _shm is not None and meta_shm_name:
        try:
            meta = _shm.SharedMemory(name=meta_shm_name, create=False)
            print(f"[DERM-HAILO] WorkerProc: attached META name={meta_shm_name} size=8")
        except Exception:
            meta = None
    # Build engine
    print("[DERM-HAILO] WorkerProc: building Hailo engine(s)...")
    lesion_path = hef_lesion or os.getenv('HEF_LESION', os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
    acne_path   = hef_acne   or os.getenv('HEF_ACNE',   os.path.expanduser('~/derma/models/acnenew.hef'))
    eng = DualHefEngine(hef_lesion=lesion_path, hef_acne=acne_path, device=None)
    t0 = time.time()
    eng.start()
    # Poll loop
    last_seq = 0
    try:
        if meta is not None:
            last_seq = int.from_bytes(meta.buf[0:4], 'little', signed=False)
    except Exception:
        last_seq = 0
    first_done = False
    last_hb = 0.0
    while True:
        # Heartbeat ~1s
        now = time.time()
        if now - last_hb > 1.0:
            try:
                hb_q.put_nowait({'alive': True})
                print("[DERM-HAILO] WorkerProc heartbeat")
            except Exception:
                pass
            last_hb = now
        # Non-blocking control
        try:
            cmd = control_q.get_nowait()
            if isinstance(cmd, dict):
                cc = cmd.get('cmd')
                if cc == 'shutdown':
                    break
                if cc == 'stop':
                    time.sleep(0.05)
                    continue
        except Exception:
            pass
        # SHM META polling
        cur_seq = last_seq
        try:
            if meta is not None:
                cur_seq = int.from_bytes(meta.buf[0:4], 'little', signed=False)
        except Exception:
            cur_seq = last_seq
        if shm is None or meta is None:
            time.sleep(0.02)
            continue
        if cur_seq == last_seq:
            time.sleep(0.01)
            continue
        # New frame ready
        last_seq = cur_seq
        if (_META_TRACE is not None and (getattr(_META_TRACE,"size",0)>0 if hasattr(_META_TRACE,"__array__") else bool(_META_TRACE))):
            try:
                print(f"[DERM-HAILO] WorkerProc META read seq={cur_seq}")
            except Exception:
                pass
        try:
            frame = np.ndarray((640, 640, 3), dtype=np.uint8, buffer=shm.buf)
        except Exception:
            time.sleep(0.005)
            continue
        # Run one infer on the current frame buffer
        det_all = _np.empty((0,6), dtype=_np.float32)
        boxes_xyxy = _np.empty((0,6), dtype=_np.float32)
        vis = None
        try:
            vis, counts = eng.process_frame(frame, conf_thres=float(os.getenv('CONF_ACNE','0.35')))
            if isinstance(counts, dict):
                d = counts.get('dets', _np.empty((0,6), dtype=_np.float32))
                b = counts.get('boxes', _np.empty((0,6), dtype=_np.float32))
                if isinstance(d, _np.ndarray):
                    det_all = d.astype(_np.float32, copy=False)
                if isinstance(b, _np.ndarray):
                    boxes_xyxy = b.astype(_np.float32, copy=False)
        except Exception as e:
            import traceback as _tb
            print("[DERM-HAILO] WorkerProc ERROR:", repr(e))
            _tb.print_exc()
            det_all = _np.empty((0,6), dtype=_np.float32)
            boxes_xyxy = _np.empty((0,6), dtype=_np.float32)
        # Startup watchdog and first-success log
        if not first_done:
            if time.time() - t0 > 10.0:
                print("[DERM-HAILO] WorkerProc watchdog: startup exceeded 10s, aborting run")
                try:
                    result_q.put_nowait({'status': 'aborted', 'msg':'startup_timeout'})
                except Exception:
                    pass
                break
            print("[DERM-HAILO] WorkerProc: first inference completed")
            first_done = True
        # Required logs and post results
        try:
            print(f"[DERM-HAILO] WorkerProc: got frame seq={cur_seq}")
        except Exception:
            pass
        # Build JSON-safe payload from robust arrays
        try:
            payload = {}
            n = int(len(boxes_xyxy)) if isinstance(boxes_xyxy, _np.ndarray) else 0
            payload['status'] = 'ok'
            payload['seq'] = cur_seq
            payload['n'] = n
            payload['counts'] = {
                'lesion': int(counts.get('lesion', 0)) if isinstance(counts, dict) else 0,
                'acne':   int(counts.get('acne', 0)) if isinstance(counts, dict) else 0,
                'total':  n,
            }
            payload['boxes'] = boxes_xyxy.tolist() if isinstance(boxes_xyxy, _np.ndarray) else []
            payload['dets']  = det_all.tolist() if isinstance(det_all, _np.ndarray) else []
            if vis is not None:
                payload['vis'] = vis
            result_q.put_nowait(payload)
            print(f"[DERM-HAILO] WorkerProc: posted results (seq={cur_seq}, n_boxes={n})")
        except Exception:
            pass
    # Close SHMs on exit (detach only)
    try:
        if shm is not None:
            shm.close()
    except Exception:
        pass
    try:
        if meta is not None:
            meta.close()
    except Exception:
        pass
# === CODEX PATCH #2: Worker SHM→infer loop END ===


# --- SAFETY BRIDGE (ROI path): always resolve NMS parsing ---
def _nms_parse_bridge(arr, conf=0.0, img_w=640, img_h=640):
    """Safe entry the ROI path can call; resolves to _parse_nms if present, else falls back to _decode_any_d5k."""
    fn = globals().get('_parse_nms', None)
    if fn is not None:
        return fn(arr, conf=conf, img_w=img_w, img_h=img_h)
    # fallback: try our d5k wrapper (it now understands NMS too)
    return _decode_any_d5k(arr, conf=conf)
# --- BRIDGE END ---


def _coerce_to_nms_c5n(arr):
    """Try to reshape arr to (C,5,N) with N=100 (Hailo NMS max boxes). Returns (ok, reshaped)."""
    import numpy as _np
    a = _np.asarray(arr)
    if a.ndim == 3 and a.shape[1] == 5:
        return True, a
    # try flatten -> (C,5,100)
    flat = a.ravel()
    if flat.size % (5*100) == 0:
        C = flat.size // (5*100)
        try:
            a2 = flat.reshape(C, 5, 100)
            return True, a2
        except Exception:
            return False, a
    return False, a

def _resolve_conf_thresh(conf_thres):
    """Return a float threshold. 0.0 must be respected (not treated as falsy)."""
    if conf_thres is None:
        return _get_conf_thresh()
    try:
        return float(conf_thres)
    except Exception:
        return _get_conf_thresh()
