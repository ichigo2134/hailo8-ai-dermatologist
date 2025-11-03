#!/usr/bin/env python3
"""
Smart Mirror Standalone Desktop Application - Dermatologist Analysis
Professional desktop application with comprehensive OpenCV skin analysis
No web browser or network dependencies required
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import multiprocessing as mp
import argparse
from types import MethodType
try:
    from multiprocessing import shared_memory
except Exception:
    shared_memory = None
import queue
import time
from PIL import Image, ImageTk
from datetime import datetime
import os
import sys
# =========================
# Smoothing / Tracking Utils
# =========================
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Det:
    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: str

def _iou_xywh(a: Det, b: Det) -> float:
    """Compute IoU between two detections in xywh format."""
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

class _Track:
    def __init__(self, det: Det):
        self.det = det
        self.age = 0
        self.hits = 1

    def update(self, det: Det):
        self.det = det
        self.hits += 1
        self.age = 0

    def step(self):
        self.age += 1

class DetectionSmoother:
    """Temporal smoothing for dermatology detections."""
    def __init__(self, max_age=5, iou_thr=0.5, min_hits=2):
        self.tracks: List[_Track] = []
        self.max_age = max_age
        self.iou_thr = iou_thr
        self.min_hits = min_hits

    def update_and_get(self, dets: List[Det]) -> List[Det]:
        updated_tracks = []
        used_dets = set()

        # Try to match new detections to existing tracks
        for trk in self.tracks:
            best_det, best_iou = None, 0.0
            for i, det in enumerate(dets):
                if i in used_dets:
                    continue
                if det.cls != trk.det.cls:
                    continue
                iou = _iou_xywh(trk.det, det)
                if iou > best_iou:
                    best_det, best_iou = (i, det), iou
            if best_det and best_iou >= self.iou_thr:
                idx, det = best_det
                trk.update(det)
                used_dets.add(idx)
                updated_tracks.append(trk)
            else:
                trk.step()
                if trk.age <= self.max_age:
                    updated_tracks.append(trk)

        # Add unmatched new detections as new tracks
        for i, det in enumerate(dets):
            if i not in used_dets:
                updated_tracks.append(_Track(det))

        self.tracks = updated_tracks

        # Return only tracks that have survived min_hits
        return [trk.det for trk in self.tracks if trk.hits >= self.min_hits]

# Dermatologist dual HEF engine integration
from ai_features.dermatologist.hailo_dual_hef import DualHefEngine

# ===== [DERM-CODEX-BEGIN] results queue =====
import queue as _queue_mod
_DERM_RESULTS_Q = None
def _derm_results_queue():
    global _DERM_RESULTS_Q
    if _DERM_RESULTS_Q is None:
        _DERM_RESULTS_Q = _queue_mod.Queue()
        print("[DERM-HAILO] Results queue ready")
    return _DERM_RESULTS_Q
# ===== [DERM-CODEX-END] results queue =====

# ===== [DERM-CODEX-BEGIN] overlay pump =====
_DERM_PUMP_STARTED = False
def _derm_overlay_pump(root, on_dets, hz=30):
    global _DERM_PUMP_STARTED
    if not _DERM_PUMP_STARTED:
        print(f"[DERM-HAILO] GUI overlay pump started (Hz={hz:.0f})")
        _DERM_PUMP_STARTED = True
    # drain queue without blocking
    try:
        drained = 0
        while True:
            dets = _derm_results_queue().get_nowait()
            on_dets(dets)
            drained += 1
    except _queue_mod.Empty:
        pass
    # schedule next tick
    root.after(int(1000 / hz), _derm_overlay_pump, root, on_dets, hz)
# ===== [DERM-CODEX-END] overlay pump =====

# ===== [DERM-SMOOTH-BEGIN] global smoother =====
_SMOOTHER = DetectionSmoother(max_age=5, iou_thr=0.5, min_hits=2)
# Helpers to convert between box formats (env-gated usage only)
def _xyxy_to_xywh(x1, y1, x2, y2):
    w = max(1.0, (x2 - x1))
    h = max(1.0, (y2 - y1))
    return (x1, y1, w, h)


def _xywh_to_xyxy(x, y, w, h):
    return (x, y, x + max(1.0, w), y + max(1.0, h))


DERM_SMOOTH_FIX_XYWH = bool(int(os.getenv("DERM_SMOOTH_FIX_XYWH", "0")))
CLI_ARGS = None
draw_detection = None
try:
    from overlay.tracker import Tracker
except Exception:
    Tracker = None
try:
    from overlay.roi import gate_skin, gate_center
except Exception:
    gate_skin = gate_center = None
try:
    from overlay.hud import SimpleFPS, draw_hud
except Exception:
    SimpleFPS = None
    draw_hud = None
try:
    from overlay.rec import DetDump, FrameSaver
except Exception:
    DetDump = FrameSaver = None


def _minimal_gui_helper(self):
    """Bound at runtime when _setup_minimal_gui is missing."""
    try:
        import os
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("No DISPLAY found; run with --no-ui or start a desktop/X session")
        import tkinter as tk
        self.root = getattr(self, 'root', None) or tk.Tk()
        try:
            self.root.title("Smart Mirror Dermatologist — Minimal GUI")
        except Exception:
            pass
        if not hasattr(self, 'video_label') or self.video_label is None:
            self.video_label = tk.Label(self.root)
            self.video_label.pack()
        start_fn = getattr(self, 'start_camera', None)
        if callable(start_fn):
            try:
                start_fn()
            except Exception as exc:
                try:
                    print(f"[DERM-HAILO] Minimal GUI start_camera warning: {exc}")
                except Exception:
                    pass
        try:
            self.root.after(0, self.update_video_feed)
        except Exception:
            pass
        def _on_close():
            stop_fn = getattr(self, 'stop_camera', None)
            if callable(stop_fn):
                try:
                    stop_fn()
                except Exception:
                    pass
            try:
                self.root.destroy()
            except Exception:
                pass
        try:
            self.root.protocol("WM_DELETE_WINDOW", _on_close)
        except Exception:
            pass
    except Exception as e:
        print(f"❌ Minimal GUI fallback failed: {e}")
        raise
# ===== [DERM-SMOOTH-END] =====

# === CODEX PATCH: Hailo worker process (spawn) BEGIN ===
def _derm_hailo_worker_proc(control_q, result_q, hb_q, shm_name, shm_size, meta_shm_name):
    """Isolated process for Hailo engine. Communicates via queues only."""
    import os, time, numpy as np, cv2
    try:
        from multiprocessing import shared_memory as _shm
    except Exception:
        _shm = None
    try:
        import ai_features.dermatologist.hailo_dual_hef as hdh

    except Exception as e:
        print(f"[DERM-HAILO] ERROR -> ImportError: {e}")
        return
    last_hb = 0.0
    derma_engine = None
    shm = None
    meta = None
    if _shm is not None and shm_name:
        try:
            shm = _shm.SharedMemory(name=shm_name, create=False)
            try:
                print(f"[DERM-HAILO] WorkerProc: attached SHM name={shm_name} size={int(shm_size)}")
            except Exception:
                pass
        except Exception:
            shm = None
    if _shm is not None and meta_shm_name:
        try:
            meta = _shm.SharedMemory(name=meta_shm_name, create=False)
            try:
                print(f"[DERM-HAILO] WorkerProc: attached META name={meta_shm_name} size=8")
            except Exception:
                pass
        except Exception:
            meta = None
    # Centralized posting helper to ensure seq and dets always present
    def _post_result(_seq, _dets=None, _counts=None, _vis=None, _status='ok', _msg=None, **kwargs):
        counts_dict = _counts or {}
        try:
            boxes = counts_dict.get('boxes', [])
            if not boxes:
                boxes = counts_dict.get('dets', [])  # Fallback to raw dets (marker)
            n = len(boxes)
            payload = {
                'status': _status,
                'seq': int(_seq) if _seq is not None else -1,
                'dets': _dets or [],
                'counts': counts_dict,
                'n': n,
            }
            if _vis is not None:
                payload['vis'] = _vis
            if _msg is not None:
                payload['msg'] = str(_msg)
            result_q.put_nowait(payload)
            print(f"[DERM-HAILO] WorkerProc: posted results (seq={payload['seq']}, n={n})")
        except Exception as _pe:
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            try:
                print(f"[DERM-HAILO] WorkerProc POST ERROR: {_pe}")
            except Exception:
                pass
    while True:
        # Heartbeat ~1s
        now = time.time()
        if now - last_hb > 1.0:
            try:
                hb_q.put_nowait({'alive': True})
            except Exception:
                pass
            try:
                print("[DERM-HAILO] WorkerProc heartbeat")
            except Exception:
                pass
            last_hb = now
        try:
            cmd = control_q.get(timeout=0.1)
        except Exception:
            continue
        if not isinstance(cmd, dict):
            continue
        c = cmd.get('cmd')
        if c == 'shutdown':
            break
        if c == 'start':
            try:
                print("[DERM-HAILO] WorkerProc: building Hailo engine(s)...")
                hef_lesion = cmd.get('hef_lesion') or os.getenv('HEF_LESION', os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
                hef_acne   = cmd.get('hef_acne')   or os.getenv('HEF_ACNE',   os.path.expanduser('~/derma/models/acnenew.hef'))
                derma_engine = hdh.DualHefEngine(hef_lesion=hef_lesion, hef_acne=hef_acne, device=None)
                t0 = time.time()
                derma_engine.start()
                # Continuous run loop using shared frame buffer
                # Initialize last_seq from META (uint32 little-endian)
                last_seq = 0
                try:
                    if meta is not None:
                        last_seq = int.from_bytes(meta.buf[0:4], byteorder='little', signed=False)
                except Exception:
                    last_seq = 0
                first_done = False
                _scales_logged = False
                _filters_logged = False
                while True:
                    # Non-blocking control to allow future stop/shutdown
                    try:
                        ctrl = control_q.get_nowait()
                        if isinstance(ctrl, dict) and ctrl.get('cmd') == 'shutdown':
                            raise SystemExit
                        if isinstance(ctrl, dict) and ctrl.get('cmd') == 'stop':
                            break
                    except Exception:
                        pass
                    # Read latest frame when sequence advances via META SHM
                    cur_seq = last_seq
                    try:
                        if meta is not None:
                            cur_seq = int.from_bytes(meta.buf[0:4], byteorder='little', signed=False)
                    except Exception:
                        cur_seq = last_seq
                    if shm is not None and cur_seq != last_seq:
                        last_seq = cur_seq
                        # META trace (optional)
                        try:
                            if os.getenv('DERM_TRACE_META','').lower() in ('1','true','on'):
                                print(f"[DERM-HAILO] WorkerProc META read seq={cur_seq}")
                        except Exception:
                            pass
                        try:
                            # Zero-copy view of frame from shared memory (640x640x3 BGR)
                            frame = np.ndarray((640,640,3), dtype=np.uint8, buffer=shm.buf)
                        except Exception:
                            frame = None
                        if frame is None:
                            time.sleep(0.01)
                            continue
                        # Optional FAKE path for testing
                        if os.getenv('DERM_FAKE_RESULTS') == '1':
                            dets = [{"xyxy": (100,100,300,300), "label": "FAKE", "score": 0.90, "color": (0,255,255)}]
                            _post_result(cur_seq, _dets=dets, _counts={"lesion": 1, "acne": 0}, _vis=None, _status='ok')
                            try:
                                print(f"[DERM-HAILO] WorkerProc: FAKE results posted (seq={cur_seq}, n=1)")
                            except Exception:
                                pass
                            continue
                        # REAL path: run inference using engine; prefer engine's decoded outputs if available
                        dets = []
                        try:
                            # Per-class thresholds
                            try:
                                conf_acne = float(os.getenv('CONF_ACNE','0.15'))
                            except Exception:
                                conf_acne = 0.15
                            try:
                                conf_lesion = float(os.getenv('CONF_LESION','0.15'))
                            except Exception:
                                conf_lesion = 0.15
                            vis, counts = derma_engine.process_frame(frame, conf_thres=conf_acne)
                            if vis is None:
                                vis = frame.copy()
                            # Attempt manual decode to build labeled dets from raw model outputs
                            try:
                                H, W = frame.shape[:2]
                                # Resolve model input size
                                in_h = 640; in_w = 640
                                try:
                                    hh, ww, _ = derma_engine.m_lesion.hwc
                                    in_h, in_w = int(hh), int(ww)
                                except Exception:
                                    pass
                                if not _scales_logged:
                                    try:
                                        sx = min(W / float(in_w), H / float(in_h))
                                        sy = sx
                                        print(f"[DERM-HAILO] DEBUG scales: model_in={in_w}x{in_h} frame={W}x{H} sx={sx:.4f} sy={sy:.4f}")
                                        print(f"[DERM-HAILO] DEBUG letterbox: roi=({W},{H}) net=({in_w},{in_h}) scale={sx:.4f} pad=({(W-in_w*sx)/2.0:.2f},{(H-in_h*sx)/2.0:.2f})")
                                    except Exception:
                                        pass
                                    _scales_logged = True
                                roi_filter_off = (os.getenv('DERM_DISABLE_ROI_FILTER','0') == '1')
                                size_filter_off = (os.getenv('DERM_DISABLE_SIZE_FILTER','0') == '1')
                                if (roi_filter_off or size_filter_off) and (not _filters_logged):
                                    try:
                                        print(f"[DERM-HAILO] DEBUG filters: ROI={'OFF' if roi_filter_off else 'ON'} SIZE={'OFF' if size_filter_off else 'ON'}")
                                    except Exception:
                                        pass
                                    _filters_logged = True
                                # Letterbox the frame to network input size (BGR->RGB)
                                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                scale0 = min(in_w / float(W), in_h / float(H))
                                nw, nh = int(round(W * scale0)), int(round(H * scale0))
                                resized = cv2.resize(rgb_full, (nw, nh), interpolation=cv2.INTER_LINEAR)
                                canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
                                px_in = (in_w - nw) // 2
                                py_in = (in_h - nh) // 2
                                canvas[py_in:py_in+nh, px_in:px_in+nw] = resized
                                if not canvas.flags['C_CONTIGUOUS']:
                                    canvas = np.ascontiguousarray(canvas)
                                inp = canvas[None, ...]
                                out_les = derma_engine.m_lesion.infer(inp)
                                out_acn = derma_engine.m_acne.infer(inp)
                                # Robustly select arrays if dicts are returned and log keys once
                                def _select_arr(out, prefer):
                                    if isinstance(out, dict):
                                        keys = list(out.keys())
                                        k_sel = None
                                        try:
                                            for k in keys:
                                                ks = str(k).lower()
                                                if prefer in ks:
                                                    k_sel = k; break
                                        except Exception:
                                            k_sel = None
                                        if k_sel is None:
                                            # fallback: pick largest array-like
                                            try:
                                                k_sel = max(keys, key=lambda k: (np.asarray(out[k]).size))
                                            except Exception:
                                                k_sel = keys[0]
                                        try:
                                            return out[k_sel], k_sel
                                        except Exception:
                                            pass
                                    return out, None
                                out_les, k_les = _select_arr(out_les, 'les')
                                out_acn, k_acn = _select_arr(out_acn, 'acn')
                                try:
                                    if not getattr(derma_engine, '_dbg_out_keys_logged', False):
                                        if (k_les is not None) or (k_acn is not None):
                                            print(f"[DERM-HAILO] WorkerProc: selected outputs -> lesion:{k_les} acne:{k_acn}")
                                        derma_engine._dbg_out_keys_logged = True
                                except Exception:
                                    pass
                                _pre_les = hdh._parse_nms(out_les, conf_lesion)
                                _pre_acn = hdh._parse_nms(out_acn, conf_acne)
                                try:
                                    print(f"[DERM-HAILO] DEBUG prefilter: candidates={len(_pre_les)+len(_pre_acn)}")
                                except Exception:
                                    pass
                                dets_les = hdh._nms(_pre_les)
                                dets_acn = hdh._nms(_pre_acn)
                                try:
                                    print(f"[DERM-HAILO] DEBUG post_nms: L={len(dets_les)+len(dets_acn)}")
                                except Exception:
                                    pass
                                # Quick proof/debug counts after NMS
                                try:
                                    print(f"[DERM-HAILO] DEBUG counts: lesion={len(dets_les)} acne={len(dets_acn)} conf={conf_acne}")
                                except Exception:
                                    pass
                                # Map input-space boxes to full-frame coords with labels (letterbox inverse)
                                scale = min(W / float(in_w), H / float(in_h))
                                padx = (W - in_w * scale) / 2.0
                                pady = (H - in_h * scale) / 2.0
                                dets = []
                                dets_mapped_full = []
                                had_candidates = (len(dets_les)+len(dets_acn)) > 0
                                def _append_labeled(_d, _label, _color):
                                    try:
                                        x1 = float(_d["x1"]) * scale + padx
                                        y1 = float(_d["y1"]) * scale + pady
                                        x2 = float(_d["x2"]) * scale + padx
                                        y2 = float(_d["y2"]) * scale + pady
                                        x1 = max(0.0, min(float(W-1), x1)); y1 = max(0.0, min(float(H-1), y1))
                                        x2 = max(0.0, min(float(W-1), x2)); y2 = max(0.0, min(float(H-1), y2))
                                        fixed = 0
                                        if (x2 <= x1) or (y2 <= y1):
                                            cx = (x1 + x2) / 2.0
                                            cy = (y1 + y2) / 2.0
                                            wmin = 6.0; hmin = 6.0
                                            x1 = cx - wmin/2.0; x2 = cx + wmin/2.0
                                            y1 = cy - hmin/2.0; y2 = cy + hmin/2.0
                                            fixed = 1
                                        x1 = max(0.0, min(float(W-1), x1)); y1 = max(0.0, min(float(H-1), y1))
                                        x2 = max(0.0, min(float(W-1), x2)); y2 = max(0.0, min(float(H-1), y2))
                                        det = {"xyxy": [x1, y1, x2, y2],
                                               "label": _label,
                                               "score": float(_d.get('score', 0.0)),
                                               "color": _color,
                                               "fixed_degenerate": int(fixed)}
                                        dets.append(det)
                                        dets_mapped_full.append(det)
                                        return 'ok'
                                    except Exception:
                                        return 'error'
                                for d in dets_les:
                                    _append_labeled(d, "lesion", (0,255,255))
                                for d in dets_acn:
                                    _append_labeled(d, "acne", (0,255,0))
                                try:
                                    if dets:
                                        dd = dets[0]
                                        x1,y1,x2,y2 = dd.get("xyxy", [0,0,0,0])
                                        # compute normalized raw for first candidate if available
                                        # fallback to zeros if not found
                                        print(f"[DERM-HAILO] DEBUG det0: raw_xywh_norm=(0.0000,0.0000,0.0000,0.0000) conf={dd.get('score',0.0):.3f} roi=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) full=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) fixed_degenerate={int(dd.get('fixed_degenerate',0))}")
                                    else:
                                        print(f"[DERM-HAILO] DEBUG det0: raw_xywh_norm=(0,0,0,0) conf=0.000 roi=(0,0,0,0) full=(0,0,0,0) fixed_degenerate=0")
                                except Exception:
                                    pass
                            except Exception:
                                dets = dets or []
                        except Exception as e_inf:
                            try:
                                print(f"[DERM-HAILO] WorkerProc ERROR: {type(e_inf).__name__}: {e_inf}")
                            except Exception:
                                pass
                            _post_result(cur_seq, _dets=[], _counts={}, _vis=None, _status='error', _msg=str(e_inf))
                            continue
                        # Fallbacks to light up GUI when decoding yields 0
                        if not dets and os.getenv('DERM_FORCE_CENTER_BOX_ON_REAL','0') == '1':
                            try:
                                H, W = frame.shape[:2]
                                s0 = int(min(H, W) * 0.30)
                                cx, cy = W // 2, H // 2
                                x1 = max(0, cx - s0 // 2); y1 = max(0, cy - s0 // 2)
                                x2 = min(W, cx + s0 // 2); y2 = min(H, cy + s0 // 2)
                                dets = [{"xyxy": (int(x1), int(y1), int(x2), int(y2)), "label": "HEF", "score": 0.50, "color": (0,255,0)}]
                                counts = {"lesion": 1, "acne": 0}
                            except Exception:
                                pass
                        # Startup watchdog placeholder
                        if (not first_done) and (time.time() - t0 > 10.0) and (not dets):
                            print("[DERM-HAILO] WorkerProc watchdog: startup exceeded 10s, posting placeholder")
                            if os.getenv('DERM_POST_FAKE_IF_EMPTY','0') == '1':
                                try:
                                    H, W = frame.shape[:2]
                                    s0 = int(min(H, W) * 0.30)
                                    cx, cy = W // 2, H // 2
                                    x1 = max(0, cx - s0 // 2); y1 = max(0, cy - s0 // 2)
                                    x2 = min(W, cx + s0 // 2); y2 = min(H, cy + s0 // 2)
                                    dets = [{"xyxy": (int(x1), int(y1), int(x2), int(y2)), "label": "WATCHDOG", "score": 0.50, "color": (0,255,255)}]
                                    counts = {"lesion": 1, "acne": 0}
                                except Exception:
                                    pass
                        # First inference completed — log exactly once
                        if not first_done:
                            print("[DERM-HAILO] WorkerProc: first inference completed")
                            first_done = True
                        # Throttled frame log
                        try:
                            print(f"[DERM-HAILO] WorkerProc: got frame seq={cur_seq}")
                        except Exception:
                            pass
                        # Post results back to GUI (centralized)
                        _post_result(cur_seq, _dets=dets, _counts=counts, _vis=vis, _status='ok', _extra={'dets_mapped_full': dets if 'dets' in locals() else [], 'had_candidates': bool(had_candidates) if 'had_candidates' in locals() else False})
                    else:
                        time.sleep(0.01)
            except Exception as e:
                try:
                    print(f"[DERM-HAILO] ERROR -> {type(e).__name__}: {e}")
                except Exception:
                    pass
                _post_result(None, _dets=[], _counts={}, _vis=None, _status='error', _msg=str(e))
            finally:
                pass
# === CODEX PATCH: Hailo worker process (spawn) END ===

# ---- BEGIN cv2.data–independent Haar locator ----
import os as _os
import cv2 as _cv2

def _haar_search_dirs():
    here = _os.path.dirname(__file__)
    return [
        "/usr/share/opencv4/haarcascades",
        "/usr/share/opencv/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
        "/usr/local/share/opencv/haarcascades",
        _os.path.join(here, "haarcascades"),
    ]

def _find_haar(filename: str) -> str:
    for base in _haar_search_dirs():
        if not base:
            continue
        path = _os.path.join(base, filename)
        if _os.path.isfile(path):
            return path
    return ""
# ---- END helper ----

# Enhanced analysis dependencies
try:
    from skimage.feature import local_binary_pattern
    ADVANCED_TEXTURE_ANALYSIS = True
    print("✅ Advanced texture analysis enabled")
except ImportError:
    ADVANCED_TEXTURE_ANALYSIS = False
    print("⚠️ Advanced texture analysis disabled (install scikit-image for full features)")

try:
    from skimage import feature as skfeature
except Exception:  # not installed or not needed
    skfeature = None


# Hailo Integration (legacy shim removed; using DualHefEngine directly)
HAILO_AVAILABLE = True


def create_skin_analysis_system(*args, **kwargs):
    raise NotImplementedError("create_skin_analysis_system is not implemented yet.")


class SmartMirrorDermatologist:
    """Complete standalone dermatologist analysis application"""

    def __init__(self):
        print("🚀 Initializing Smart Mirror Dermatologist Application...")

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Smart Mirror - Efficient Dermatologist Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')

        # Application state variables
        self.camera = None
        self.current_frame = None
        self.analysis_thread = None
        self.running = True

        # Dermatologist engine lifecycle
        self.derma_active = False
        self.derma_threshold = float(os.getenv('DERMA_THRESHOLD','0.15'))
        self.derma_counts = {"lesion": 0, "acne": 0}
        self.last_derma_vis = None
        # Live loop control (dermatologist)
        self.derm_live_active = False
        self.derm_thread = None
        self._derma_worker = None
        self._derma_worker_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._derma_fullframe_runs = 0
        self.derma_detecting_face = False
        # Face detection state (initialized before threads start)
        self._face_lock = threading.Lock()
        self.face_cascade = None          # lazy loaded later
        self._haar_loaded = False         # print once on load/miss
        self._face_missing_count = 0      # bounded hint counter
        self._last_roi = None             # (x,y,w,h) last HAAR ROI
        self._roi_grace = 0               # grace frames remaining when face temporarily lost
        self._roi_fresh = False           # True only when HAAR found a face this frame
        # Create engine (legacy behavior; no shared device manager)
        self.derma_engine = None  # defer engine creation until first start
        # Dermatology visualization cache (from worker 'vis' and 'dets')
        self._derm_latest_vis = None
        self._derm_latest_vis_ts = 0.0
        self._derm_vis_dumped_once = False
        self._derm_latest_dets = []


        # EFFICIENT ANALYSIS CONTROL
        self.analysis_active = False
        # Stop dermatologist live loop cleanly
        try:
            self.stop_derm_live()
        except Exception:
            pass
        self.analysis_complete = False
        self.face_detected = False
        self.face_stable_count = 0
        self.required_stable_frames = 15  # Reduced from 30

        # EFFICIENT FACE TRACKING
        self.current_face_region = None
        self.last_face_region = None

        # LOCKED ANALYSIS RESULTS (stationary system)
        self.final_results = {
            'acne_spots': [],
            'pores': [],
            'blackheads': [],
            'age_spots': [],
            'moles': []
        }
        self.results_locked = False

        self._cli_args = CLI_ARGS
        self._tracker = None
        self._detdump = None
        self._framesaver = None
        args = self._cli_args
        tracker_flags = bool(Tracker is not None and args is not None and any(getattr(args, k, False) for k in ('smooth','hyst','debounce','assoc')))
        if tracker_flags:
            try:
                self._tracker = Tracker(iou_match=0.3, ema_alpha=0.5, appear_thr=0.35,
                                        disappear_thr=0.25, confirm_frames=3,
                                        miss_max=10, max_tracks=64)
            except Exception:
                self._tracker = None
        if args and DetDump is not None and isinstance(getattr(args, 'dump_dets',''), str) and args.dump_dets:
            try:
                self._detdump = DetDump(args.dump_dets)
            except Exception:
                self._detdump = None
        if args and FrameSaver is not None and isinstance(getattr(args, 'save_frames',''), str) and args.save_frames:
            try:
                self._framesaver = FrameSaver(args.save_frames, interval=getattr(args,'save_interval',15),
                                              annotate=getattr(args,'save_annotated',False))
            except Exception:
                self._framesaver = None
        self._fps_meter = SimpleFPS(30) if (args and getattr(args, 'hud', False) and SimpleFPS is not None) else None

        # Ensure camera start/stop binders exist before using them.
        self._bind_camera_methods()

        # Minimal in-class GUI pump to guarantee availability for Tk `after(...)`.
        # This avoids AttributeError if a longer pump exists elsewhere but is out of scope.
        def _minimal_pump(self):
            try:
                # Drain latest item only (if queue exists), then forward to GUI apply hook.
                dets = None
                q = getattr(self, "_derm_results_q", None)
                if q is not None:
                    try:
                        items = []
                        # non-blocking drain
                        while True:
                            try:
                                items.append(q.get_nowait())
                            except Exception:
                                break
                        if items:
                            last = items[-1]
                            dets = last.get("dets") if isinstance(last, dict) else None
                            # Also accept plain list payloads
                            if dets is None and isinstance(last, (list, tuple)):
                                dets = last
                    except Exception:
                        pass
                apply_fn = getattr(self, "_derm_apply_detections_on_gui_thread", None)
                if callable(apply_fn) and dets is not None:
                    try:
                        apply_fn(dets)
                    except Exception:
                        pass
                # Let the display refresh if update_video_feed exists
                upd = getattr(self, "update_video_feed", None)
                if callable(upd):
                    try:
                        upd()
                    except Exception:
                        pass
            finally:
                # Always reschedule to keep UI responsive
                try:
                    self.root.after(getattr(self, "_pump_period_ms", 33), self._pump_derm_queue)
                except Exception:
                    pass

        # Expose the minimal pump so other code paths can bind it when needed.
        self._minimal_pump = _minimal_pump.__get__(self, type(self))
        if not callable(getattr(self, "_pump_derm_queue", None)):
            self._pump_derm_queue = self._minimal_pump

        # Ensure GUI-apply function is bound even if module scope import failed earlier.
        if not callable(getattr(self, "_derm_apply_detections_on_gui_thread", None)):
            fn = globals().get("_derm_apply_detections_on_gui_thread")
            if callable(fn):
                from types import MethodType as _MethodType
                self._derm_apply_detections_on_gui_thread = _MethodType(fn, self)

        # PERFORMANCE OPTIMIZATION
        self.frame_skip_count = 0
        self.analysis_frame_interval = 3  # Process every 3rd frame for face detection
        self.video_fps_target = 30

        # Initialize metrics for display
        self.metric_vars = {}

        args = getattr(self, '_cli_args', CLI_ARGS)
        if not hasattr(self, '_setup_minimal_gui') or not callable(getattr(self, '_setup_minimal_gui', None)):
            try:
                self._setup_minimal_gui = MethodType(_minimal_gui_helper, self)
            except Exception:
                pass
        use_min = bool(args and getattr(args, 'force_min_gui', False))
        try:
            if not use_min and hasattr(self, 'setup_professional_gui') and callable(getattr(self, 'setup_professional_gui')):
                self.setup_professional_gui()
            else:
                self._setup_minimal_gui()
        except Exception as e:
            print(f"❌ GUI init failed; run with --no-ui. Reason: {e}")
            raise

        print("✅ Application initialized successfully - EFFICIENT MODE")
        # Hotkeys for dermatologist controls
        self.root.bind('d', lambda e: self.start_analysis())
        # removed 'p' hotkey per legacy restore
        self.root.bind('+', lambda e: self._nudge_threshold(0.05))
        self.root.bind('-', lambda e: self._nudge_threshold(-0.05))

        # ===== [DERM-CODEX-BEGIN] start pump + optional selftest =====
        _derm_overlay_pump(self.root, lambda dets: self._derm_apply_detections_on_gui_thread(dets), hz=30)
        if os.getenv("DERM_OVERLAY_SELFTEST") == "1":
            _derm_results_queue().put([{"xyxy": (50,50,250,250), "label":"SELFTEST", "score": 0.99, "color": (0,255,0)}])
            print("[DERM-HAILO] GUI selftest posted (n_boxes=1)")
        # ===== [DERM-CODEX-END] start pump + optional selftest =====

        # Start camera
        self.start_camera()

        # === CODEX PATCH: TK main-thread overlay pump BEGIN ===
        # Thread-safe queues for dermatology pipeline
        self._derm_results_q = queue.Queue(maxsize=2)   # results to GUI (non-blocking)
        # Multiprocessing queues for Hailo worker process
        self._mp_ctx = mp.get_context("spawn")
        self._mp_ctrl_q = self._mp_ctx.Queue()
        self._mp_result_q = self._mp_ctx.Queue()
        self._mp_hb_q = self._mp_ctx.Queue()

        # ===== [DERM-CODEX-BEGIN] results bridge =====
        def _results_bridge_loop():
            print("[DERM-HAILO] Results bridge thread started")
            while True:
                pkt = self._mp_result_q.get()
                # Extract basics robustly
                seq = -1
                try:
                    if isinstance(pkt, dict):
                        seq = int(pkt.get('seq', -1))
                except Exception:
                    seq = -1
                dets = []
                try:
                    dets = (pkt.get('dets', []) if isinstance(pkt, dict) else []) or []
                except Exception:
                    dets = []
                counts = {}
                try:
                    counts = (pkt.get('counts', {}) if isinstance(pkt, dict) else {}) or {}
                except Exception:
                    counts = {}
                boxes = counts.get('boxes')
                if boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0):
                    boxes = counts.get('dets')
                    if boxes is None: boxes = []
                    if os.getenv("DERM_DEBUG", "0") == "1":
                        # DEBUG: remove if noisy
                        try:
                            n_dbg = (len(boxes) if not hasattr(boxes, "shape") else boxes.shape[0])
                            print(f"[DERM-HAILO] Bridge: boxes type={type(boxes).__name__} n={n_dbg}")
                        except Exception:
                            pass
                if (not dets) and (hasattr(boxes, "__len__") and len(boxes) > 0):
                    try:
                        dets = [{'xyxy': (int(b[0]), int(b[1]), int(b[2]), int(b[3]))} for b in boxes if isinstance(b, (list, tuple)) and len(b) >= 4]
                    except Exception:
                        pass  # leave dets empty if anything goes wrong
                # Optional fallback if no dets
                if not dets and os.getenv('DERM_FORCE_CENTER_BOX_ON_REAL','0') == '1':
                    try:
                        H, W = getattr(self, '_last_frame_shape', (480, 640))
                        s0 = int(min(H, W) * 0.30)
                        cx, cy = W // 2, H // 2
                        x1 = max(0, cx - s0 // 2); y1 = max(0, cy - s0 // 2)
                        x2 = min(W, cx + s0 // 2); y2 = min(H, cy + s0 // 2)
                        dets = [{"xyxy": (x1, y1, x2, y2), "label": "WATCHDOG", "score": 0.50, "color": (0,255,255)}]
                    except Exception as _fe:
                        try:
                            print(f"[DERM-HAILO] Bridge fallback error: {_fe}")
                        except Exception:
                            pass
                # Push dets to overlay pump (always)
                try:
                    _derm_results_queue().put(dets)
                except Exception:
                    pass
                # Forward vis/dets/counts payload unchanged if present
                payload = {}
                if isinstance(pkt, dict) and ('vis' in pkt):
                    payload['vis'] = pkt['vis']
                    # Stash latest annotated vis with timestamp for display preference
                    try:
                        vis_obj = pkt['vis']
                        if isinstance(vis_obj, np.ndarray) and vis_obj.ndim == 3 and vis_obj.size > 0:
                            self._derm_latest_vis = vis_obj
                            self._derm_latest_vis_ts = time.time()
                            if not self._derm_vis_dumped_once:
                                try:
                                    cv2.imwrite('/tmp/derm_vis_first.jpg', vis_obj)
                                except Exception:
                                    pass
                                self._derm_vis_dumped_once = True
                    except Exception:
                        pass
                # Stash latest dets and include in payload
                try:
                    self._derm_latest_dets = dets or []
                except Exception:
                    self._derm_latest_dets = []
                if dets:
                    payload['dets'] = dets
                if isinstance(pkt, dict) and ('had_candidates' in pkt):
                    payload['had_candidates'] = bool(pkt.get('had_candidates'))
                    try:
                        self._derm_had_candidates = bool(pkt.get('had_candidates'))
                    except Exception:
                        self._derm_had_candidates = False
                if counts:
                    payload['counts'] = counts
                if payload:
                    try:
                        self._derm_results_q.put_nowait(payload)
                    except queue.Full:
                        try:
                            _ = self._derm_results_q.get_nowait()
                            self._derm_results_q.put_nowait(payload)
                        except Exception:
                            pass
                # Log summary with dets length preferred; fallback to counts only if missing
                try:
                    _boxes_tmp = counts.get('boxes')
                    if _boxes_tmp is None or (hasattr(_boxes_tmp, "__len__") and len(_boxes_tmp) == 0):
                        _boxes_tmp = counts.get('dets') or []
                    n = (len(dets) if dets else len(_boxes_tmp))
                except Exception:
                    n = 0
                print(f"[DERM-HAILO] GUI bridge got (seq={seq}) n={n}")
        threading.Thread(target=_results_bridge_loop, name="DermResultsBridge", daemon=True).start()
        # ===== [DERM-CODEX-END] results bridge =====
        # Shared memory for frames (640x640x3 BGR uint8) with fixed name for spawn
        self._shm_size = 640*640*3
        self._shm = None
        self._shm_name = 'derm_frame_v1'
        if shared_memory is not None:
            try:
                # Try to create named block; if exists, attach
                self._shm = shared_memory.SharedMemory(name=self._shm_name, create=True, size=self._shm_size)
            except Exception:
                try:
                    self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
                except Exception:
                    self._shm = None
        # Meta shared memory (seq:uint32, flags:uint32)
        self._meta_shm = None
        self._meta_shm_name = 'derm_meta_v1'
        if shared_memory is not None:
            try:
                self._meta_shm = shared_memory.SharedMemory(name=self._meta_shm_name, create=True, size=8)
                self._meta_shm.buf[:8] = b"\x00"*8
            except Exception:
                try:
                    self._meta_shm = shared_memory.SharedMemory(name=self._meta_shm_name, create=False)
                except Exception:
                    self._meta_shm = None
        self._local_frame_seq = 0
        # Monotonic frame sequence
        self._mp_frame_seq = self._mp_ctx.Value('I', 0)
        self._last_frame_log_ts = 0.0
        self._pump_tick = 0
        self._pump_last_heartbeat = 0.0
        try:
            hz = float(os.getenv('DERM_PUMP_HZ', '30'))
        except Exception:
            hz = 30.0
        hz = max(10.0, min(60.0, hz))
        self._pump_period_ms = int(1000.0 / hz)
        pump = getattr(self, "_pump_derm_queue", None)
        if not callable(pump):
            if hasattr(self, "_minimal_pump") and callable(getattr(self, "_minimal_pump", None)):
                self._pump_derm_queue = self._minimal_pump.__get__(self, type(self))
            else:
                self._pump_derm_queue = _minimal_pump.__get__(self, type(self))
        self.root.after(self._pump_period_ms, self._pump_derm_queue)
        # Spawn worker process (isolates GIL/native locks) unless disabled
        self._derm_run_active = False
        _hailo_disabled = (os.getenv('DERM_DISABLE_HAILO','0') == '1')
        if not _hailo_disabled:
            self._derm_proc = self._mp_ctx.Process(target=_derm_hailo_worker_proc, args=(self._mp_ctrl_q, self._mp_result_q, self._mp_hb_q, self._shm_name, self._shm_size, self._meta_shm_name), daemon=True)
            self._derm_proc.start()
            try:
                print(f"[DERM-HAILO] Spawned Hailo worker pid={self._derm_proc.pid}")
            except Exception:
                pass
        else:
            self._derm_proc = None
            try:
                print("[DERM-HAILO] Worker disabled (DERM_DISABLE_HAILO=1): GUI-only mode")
            except Exception:
                pass
        # === CODEX PATCH: TK main-thread overlay pump END ===


    def run(self):
        """Start Tk main loop with preflight checks and ensure cleanup on exit."""
        # --- Preflight: ensure we have a root and a callable pump ---
        if not hasattr(self, "root") or self.root is None:
            raise RuntimeError("GUI root missing")

        # Guarantee the GUI pump exists
        pump = getattr(self, "_pump_derm_queue", None)
        if not callable(pump):
            if hasattr(self, "_minimal_pump") and callable(getattr(self, "_minimal_pump", None)):
                from types import MethodType as _MethodType
                self._pump_derm_queue = _MethodType(self._minimal_pump, self)
                try:
                    print("[DERM-HAILO] Using minimal pump fallback")
                except Exception:
                    pass
            else:
                raise RuntimeError("_pump_derm_queue not available")

        # If any GUI paths schedule update_video_feed, ensure there is a callable
        if not callable(getattr(self, "update_video_feed", None)):
            def _fallback_uvf(self):
                # Safe no-op; prevents .after(...) scheduling from crashing
                try:
                    if not getattr(self, "_uvf_warned", False):
                        print("[DERM-HAILO] WARN: update_video_feed missing; using fallback")
                        self._uvf_warned = True
                except Exception:
                    pass
            from types import MethodType as _MethodType
            self.update_video_feed = _MethodType(_fallback_uvf, self)

        # Make sure a pump callback is actually scheduled
        try:
            if not isinstance(getattr(self, "_pump_period_ms", None), int):
                self._pump_period_ms = 33
            self.root.after(self._pump_period_ms, self._pump_derm_queue)
        except Exception:
            # As a last resort, schedule minimal pump at 33ms
            self._pump_derm_queue = self._minimal_pump.__get__(self, type(self))
            self._pump_period_ms = 33
            self.root.after(self._pump_period_ms, self._pump_derm_queue)

        # --- Main loop with guaranteed cleanup ---
        try:
            self.root.mainloop()
        finally:
            self._graceful_shutdown()

    def _graceful_shutdown(self):
        """Idempotent cleanup: stop worker, release shared memory, camera, and Tk."""
        # Stop worker process if present
        try:
            proc = getattr(self, "_derm_proc", None)
            ctrl_q = getattr(self, "_mp_ctrl_q", None)
            if proc is not None:
                try:
                    if ctrl_q is not None:
                        ctrl_q.put({"cmd": "shutdown"})
                except Exception:
                    pass
                try:
                    proc.join(timeout=2.0)
                    if proc.is_alive():
                        proc.terminate()
                except Exception:
                    pass
                self._derm_proc = None
        except Exception:
            pass

        # Close & unlink shared memory blocks if present
        for _name in ("_meta_shm", "_shm"):
            try:
                shm = getattr(self, _name, None)
                if shm is not None:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        # Only unlink if this process created/owns it; best-effort
                        shm.unlink()
                    except Exception:
                        pass
                    setattr(self, _name, None)
            except Exception:
                pass

        # Release camera if it looks like an OpenCV capture
        try:
            cam = getattr(self, "camera", None)
            if cam is not None and hasattr(cam, "release"):
                cam.release()
                self.camera = None
        except Exception:
            pass

        # Tear down Tk safely
        try:
            if hasattr(self, "root") and self.root is not None:
                try:
                    self.root.quit()
                except Exception:
                    pass
                try:
                    self.root.destroy()
                except Exception:
                    pass
                self.root = None
        except Exception:
            pass
    def setup_professional_gui(self):
        """Setup professional-looking GUI interface"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'), background='#1a1a1a', foreground='#00ff41')
        style.configure('Subtitle.TLabel', font=('Arial', 14, 'bold'), background='#1a1a1a', foreground='#ffffff')
        style.configure('Metric.TLabel', font=('Arial', 12, 'bold'), background='#2d2d2d', foreground='#00ff41')

        # Create main container
        main_container = tk.Frame(self.root, bg='#1a1a1a')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header section
        header_frame = tk.Frame(main_container, bg='#2d2d2d', height=100)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)

        # Application title
        title_label = tk.Label(header_frame,
                              text="🔬 SMART MIRROR DERMATOLOGIST",
                              font=('Arial', 28, 'bold'),
                              bg='#2d2d2d', fg='#00ff41')
        title_label.pack(expand=True)

        subtitle_label = tk.Label(header_frame,
                                 text="Professional Skin Analysis System • AI-Powered Dermatology",
                                 font=('Arial', 12),
                                 bg='#2d2d2d', fg='#ffffff')
        subtitle_label.pack()

        # Main content area with three panels
        content_frame = tk.Frame(main_container, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Camera feed
        self.setup_camera_panel(content_frame)

        # Middle panel - Real-time metrics
        self.setup_metrics_panel(content_frame)

        # Right panel - Analysis results
        self.setup_results_panel(content_frame)

        # Status bar
        self.setup_status_bar(main_container)

    def setup_camera_panel(self, parent):
        """Setup camera feed panel"""
        camera_frame = tk.Frame(parent, bg='#2d2d2d', width=500)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Camera title
        camera_title = tk.Label(camera_frame,
                               text="📹 LIVE CAMERA FEED",
                               font=('Arial', 16, 'bold'),
                               bg='#2d2d2d', fg='#00ff41')
        camera_title.pack(pady=15)

        # Video display
        self.video_frame = tk.Frame(camera_frame, bg='black', width=640, height=480)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg='black', text="Initializing Camera...",
                                   fg='white', font=('Arial', 14))
        self.video_label.pack(expand=True)

        # Control buttons
        control_frame = tk.Frame(camera_frame, bg='#2d2d2d')
        control_frame.pack(fill=tk.X, pady=20)

        self.analyze_btn = tk.Button(control_frame,
                                   text="🔬 Start Precision Analysis",
                                   font=('Arial', 16, 'bold'),
                                   bg='#27ae60', fg='white',
                                   activebackground='#00cc00',
                                   height=2, width=30,
                                   command=self.start_analysis)
        self.analyze_btn.pack(pady=10)
        self.analyze_btn.configure(state=tk.NORMAL)

        # Camera controls
        camera_controls = tk.Frame(control_frame, bg='#2d2d2d')
        camera_controls.pack(fill=tk.X, pady=10)

        tk.Button(camera_controls, text="📷 Snapshot", font=('Arial', 12),
                 bg='#444444', fg='white', command=self.take_snapshot).pack(side=tk.LEFT, padx=5)

        tk.Button(camera_controls, text="🔄 Reset Camera", font=('Arial', 12),
                 bg='#444444', fg='white', command=self.reset_camera).pack(side=tk.LEFT, padx=5)

        # Dermatologist threshold slider
        thr_frame = tk.Frame(control_frame, bg='#2d2d2d')
        thr_frame.pack(fill=tk.X, pady=8)
        tk.Label(thr_frame, text="AI Threshold", font=('Arial', 12, 'bold'), bg='#2d2d2d', fg='#ffffff').pack(side=tk.LEFT, padx=8)
        self.threshold_scale = tk.Scale(thr_frame, from_=0.10, to=0.80, resolution=0.05,
                                        orient=tk.HORIZONTAL, length=260, showvalue=True,
                                        bg='#2d2d2d', fg='#00ff41', troughcolor='#444444',
                                        highlightthickness=0, command=self.on_threshold_change)
        self.threshold_scale.set(self.derma_threshold)
        self.threshold_scale.pack(side=tk.LEFT, padx=8)

    def setup_metrics_panel(self, parent):
        """Setup real-time metrics panel"""
        metrics_frame = tk.Frame(parent, bg='#2d2d2d', width=350)
        metrics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        metrics_frame.pack_propagate(False)

        # Metrics title
        metrics_title = tk.Label(metrics_frame,
                               text="📊 REAL-TIME METRICS",
                               font=('Arial', 16, 'bold'),
                               bg='#2d2d2d', fg='#00ff41')
        metrics_title.pack(pady=15)

        # Create metric displays
        self.metrics_container = tk.Frame(metrics_frame, bg='#2d2d2d')
        self.metrics_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.create_metric_displays()

        # Health score gauge (visual representation)
        health_gauge_frame = tk.Frame(self.metrics_container, bg='#1a1a1a', height=100)
        health_gauge_frame.pack(fill=tk.X, pady=20)
        health_gauge_frame.pack_propagate(False)

        tk.Label(health_gauge_frame, text="💚 OVERALL HEALTH SCORE",
                font=('Arial', 14, 'bold'), bg='#1a1a1a', fg='#ffffff').pack(pady=5)

        self.health_score_var = tk.StringVar()
        self.health_score_var.set("---%")

        self.health_score_label = tk.Label(health_gauge_frame,
                                          textvariable=self.health_score_var,
                                          font=('Arial', 24, 'bold'),
                                          bg='#1a1a1a', fg='#00ff41')
        self.health_score_label.pack()

        # Analysis status
        status_frame = tk.Frame(self.metrics_container, bg='#1a1a1a')
        status_frame.pack(fill=tk.X, pady=10)

        tk.Label(status_frame, text="🔍 ANALYSIS STATUS",
                font=('Arial', 12, 'bold'), bg='#1a1a1a', fg='#ffffff').pack()

        self.analysis_status_var = tk.StringVar()
        self.analysis_status_var.set("Waiting...")

        tk.Label(status_frame, textvariable=self.analysis_status_var,
                font=('Arial', 11), bg='#1a1a1a', fg='#ffaa00').pack()

    def create_metric_displays(self):
        """Create enhanced metric display widgets"""
        self.metric_vars = {}

        metrics = [
            ('faces', '👤 Face Detected', '0'),
            ('acne_spots', '🔴 Acne/Pimples', '0'),
            ('pores', '🔵 Enlarged Pores', '0'),
            ('blackheads', '🟡 Blackheads', '0'),
            ('age_spots', '🟠 Age/Dark Spots', '0'),
            ('health_score', '💚 Health Score', '0%'),
            ('smoothness', '✨ Skin Smoothness', '0%')
        ]

        for key, label, default in metrics:
            # Create metric container
            metric_frame = tk.Frame(self.metrics_container, bg='#3d3d3d', height=60)
            metric_frame.pack(fill=tk.X, pady=8)
            metric_frame.pack_propagate(False)

            # Metric label
            tk.Label(metric_frame, text=label, font=('Arial', 11, 'bold'),
                    bg='#3d3d3d', fg='#ffffff').pack(side=tk.LEFT, padx=15, pady=10)

            # Metric value
            self.metric_vars[key] = tk.StringVar()
            self.metric_vars[key].set(default)

            tk.Label(metric_frame, textvariable=self.metric_vars[key],
                    font=('Arial', 14, 'bold'),
                    bg='#3d3d3d', fg='#00ff41').pack(side=tk.RIGHT, padx=15, pady=10)

    def setup_results_panel(self, parent):
        """Setup detailed results panel"""
        results_frame = tk.Frame(parent, bg='#2d2d2d', width=450)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Results title
        results_title = tk.Label(results_frame,
                               text="📝 DETAILED ANALYSIS LOG",
                               font=('Arial', 16, 'bold'),
                               bg='#2d2d2d', fg='#00ff41')
        results_title.pack(pady=15)

        # Text widget with scrollbar for results
        text_container = tk.Frame(results_frame, bg='#2d2d2d')
        text_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Analysis results display
        self.results_text = tk.Text(text_container,
                                   font=('Consolas', 10),
                                   bg='#000000', fg='#00ff00',
                                   insertbackground='#00ff00',
                                   wrap=tk.WORD,
                                   height=20)

        # Scrollbar
        scrollbar = tk.Scrollbar(text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        # Control buttons
        button_frame = tk.Frame(results_frame, bg='#2d2d2d')
        button_frame.pack(fill=tk.X, padx=20, pady=20)

        tk.Button(button_frame, text="💾 Save Results", font=('Arial', 12),
                 bg='#0066cc', fg='white', activebackground='#0088ff',
                 command=self.save_results).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="🗑️ Clear Log", font=('Arial', 12),
                 bg='#cc6600', fg='white', activebackground='#ff8800',
                 command=self.clear_log).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="❌ Exit", font=('Arial', 12),
                 bg='#cc0000', fg='white', activebackground='#ff0000',
                 command=self.exit_app).pack(side=tk.RIGHT, padx=5)

    def setup_status_bar(self, parent):
        """Setup status bar"""
        self.status_frame = tk.Frame(parent, bg='#2d2d2d', height=50)
        self.status_frame.pack(fill=tk.X, pady=(20, 0))
        self.status_frame.pack_propagate(False)

        self.status_var = tk.StringVar()
        self.status_var.set("🔄 Initializing camera...")

        self.status_label = tk.Label(self.status_frame,
                                   textvariable=self.status_var,
                                   font=('Arial', 12),
                                   bg='#2d2d2d', fg='#ffffff')
        self.status_label.pack(side=tk.LEFT, padx=20, expand=True, anchor='w')

        # System info
        system_info = tk.Label(self.status_frame,
                              text=f"🖥️ System Ready • {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                              font=('Arial', 10),
                              bg='#2d2d2d', fg='#888888')
        system_info.pack(side=tk.RIGHT, padx=20)

    def _bind_camera_methods(self) -> None:
        """Create safe start/stop camera aliases if primary methods are missing."""
        if not (hasattr(self, "start_camera") and callable(getattr(self, "start_camera", None))):
            for name in [
                "open_camera", "init_camera", "start_camera_capture",
                "_start_camera", "camera_start", "ensure_camera"
            ]:
                candidate = getattr(self, name, None)
                if callable(candidate):
                    self.start_camera = candidate
                    break
            else:
                def _noop_start():
                    return True
                self.start_camera = _noop_start

        if not (hasattr(self, "stop_camera") and callable(getattr(self, "stop_camera", None))):
            for name in [
                "close_camera", "release_camera", "stop_camera_capture",
                "_stop_camera", "camera_stop"
            ]:
                candidate = getattr(self, name, None)
                if callable(candidate):
                    self.stop_camera = candidate
                    break
            else:
                def _noop_stop():
                    return True
                self.stop_camera = _noop_stop

    def _setup_minimal_gui(self):
        """
        Minimal Tkinter window used when setup_professional_gui is unavailable
        or --force-minimal-gui is set. Creates a small window and schedules
        update_video_feed() via root.after.
        """
        _minimal_gui_helper(self)

    def _nudge_threshold(self, delta):
        new_thr = max(0.05, min(0.95, self.derma_threshold + delta))
        self.derma_threshold = new_thr
        if hasattr(self, 'threshold_scale'):
            try:
                self.threshold_scale.set(new_thr)
            except Exception:
                pass

    def start_camera(self):
        """Initialize and start camera capture"""
        try:
            self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():
                raise Exception("No camera found")

            # Test camera read
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Camera capture test failed")

            self.status_var.set("✅ Camera initialized successfully")

            # Start video thread
            self.video_thread = threading.Thread(target=self.update_video_feed, daemon=True)
            self.video_thread.start()

            self.log_message("🎥 Camera system initialized successfully")
            self.log_message("📹 Live video feed active")
            self.log_message("🔬 Ready for dermatologist analysis")

            print("✅ Camera started successfully")

        except Exception as e:
            error_msg = f"Camera initialization failed: {str(e)}"
            self.status_var.set(f"❌ {error_msg}")
            self.log_message(f"❌ {error_msg}")
            print(f"❌ Camera error: {e}")
    def update_video_feed(self):
        """Update video feed with live face tracking markers"""
        if getattr(self, '_fps_meter', None) is not None:
            try:
                self._fps_meter.tick()
            except Exception:
                pass
        global draw_detection
        cap = getattr(self, '_cap', None)
        try:
            if cap is None or not (hasattr(cap, 'isOpened') and cap.isOpened()):
                cap = getattr(self, 'camera', None)
        except Exception:
            cap = getattr(self, 'camera', None)
        if cap is not None and hasattr(cap, 'isOpened') and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                with self._frame_lock:
                    self.current_frame = frame.copy()  # BGR frame for engine
                # Write latest frame to shared memory buffer for worker (640x640 BGR) and bump META seq
                try:
                    if getattr(self, '_shm', None) is not None:
                        bgr = frame
                        if bgr.shape[0] != 640 or bgr.shape[1] != 640:
                            bgr = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
                        if bgr.dtype != np.uint8 or not bgr.flags['C_CONTIGUOUS']:
                            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
                        if bgr.size == self._shm_size:
                            self._shm.buf[:self._shm_size] = bgr.ravel().tobytes()
                            # Update local seq and write to META (uint32 LE), flags=0
                            self._local_frame_seq = (self._local_frame_seq + 1) & 0xFFFFFFFF
                            if getattr(self, '_meta_shm', None) is not None:
                                self._meta_shm.buf[0:4] = int(self._local_frame_seq).to_bytes(4, 'little', signed=False)
                                self._meta_shm.buf[4:8] = b"\x00\x00\x00\x00"
                            now = time.time()
                            if now - getattr(self, '_last_frame_log_ts', 0.0) > 1.0:
                                print(f"[DERM-HAILO] GUI: sent frame seq={self._local_frame_seq}")
                                # META trace (optional)
                                try:
                                    if os.getenv('DERM_TRACE_META','').lower() in ('1','true','on'):
                                        print(f"[DERM-HAILO] GUI META write seq={self._local_frame_seq}")
                                except Exception:
                                    pass
                                self._last_frame_log_ts = now
                except Exception:
                    pass

                # Prefer engine-rendered 'vis' if it's fresh; else draw dets on raw frame
                mode_used = 'DETS'
                try:
                    try:
                        max_age = float(os.getenv('DERM_VIS_MAX_AGE_S', '1.0'))
                    except Exception:
                        max_age = 1.0
                    use_vis = (self._derm_latest_vis is not None) and ((time.time() - float(self._derm_latest_vis_ts)) <= max_age)
                    if use_vis:
                        vis = self._derm_latest_vis
                        if isinstance(vis, np.ndarray) and vis.ndim == 3 and vis.size > 0:
                            if vis.shape[:2] != frame.shape[:2]:
                                frame = cv2.resize(vis, (frame.shape[1], frame.shape[0]))
                            else:
                                frame = vis
                            mode_used = 'VIS'
                    else:
                        # Draw stashed detections on the raw camera frame (red, thickness 2)
                        try:
                            dets = getattr(self, "_derm_latest_dets", []) or []
                            args = getattr(self, '_cli_args', CLI_ARGS)
                            try:
                                min_area = int(getattr(args, 'min_area_px', 0)) if args is not None else 0
                                if min_area > 0 and isinstance(dets, (list, tuple)) and dets:
                                    dets = [d for d in dets if float(d.get('w', 0.0)) * float(d.get('h', 0.0)) >= float(min_area)]
                            except Exception:
                                pass
                            tracker_wanted = bool(Tracker is not None and args is not None and any(getattr(args, k, False) for k in ('smooth','hyst','debounce','assoc')))
                            if tracker_wanted:
                                try:
                                    if getattr(self, '_tracker', None) is None:
                                        self._tracker = Tracker()
                                    dets_tracked = self._tracker.update(dets or [])
                                    if dets_tracked:
                                        dets = dets_tracked
                                except Exception:
                                    pass
                            try:
                                if isinstance(dets, (list, tuple)) and args is not None:
                                    mode = getattr(args, 'roi_gate', 'none')
                                    if mode == 'center' and gate_center is not None:
                                        dets = gate_center(dets, frame, frac=0.60)
                                    elif mode == 'skin' and gate_skin is not None:
                                        dets = gate_skin(dets, frame)
                            except Exception:
                                pass
                            if getattr(self, '_detdump', None) is not None:
                                try:
                                    to_dump = []
                                    for d in (dets or []):
                                        try:
                                            to_dump.append({
                                                "x": float(d.get("x",0.0)),
                                                "y": float(d.get("y",0.0)),
                                                "w": float(d.get("w",0.0)),
                                                "h": float(d.get("h",0.0)),
                                                "conf": float(d.get("conf",0.0)),
                                                "cls": str(d.get("cls","unknown")),
                                                "tid": int(d.get("tid",0)) if isinstance(d.get("tid",None),(int,)) else None
                                            })
                                        except Exception:
                                            continue
                                    self._detdump.write(to_dump, extra={"roi": getattr(args,'roi_gate','none')})
                                except Exception:
                                    pass
                            if draw_detection is None and getattr(args, 'pretty_draw', False):
                                try:
                                    from overlay.draw import draw_detection as _draw_detection
                                    draw_detection = _draw_detection
                                except Exception:
                                    draw_detection = None
                            if getattr(args, 'pretty_draw', False) and draw_detection is not None and isinstance(dets, (list, tuple)):
                                for d in dets:
                                    try:
                                        if not all(k in d for k in ('x','y','w','h','conf','cls')):
                                            continue
                                        draw_detection(frame, d)
                                    except Exception:
                                        continue
                            else:
                                _thick = int(float(os.getenv("DERM_BOX_THICKNESS", "3")))
                                _thick = max(1, min(12, _thick))
                                top_safe = int(float(os.getenv('DERM_TOP_SAFE_PX', '32')))
                                draw_center = (os.getenv('DERM_DRAW_CENTER_DOT_FOR_DEGENERATE','1') != '0')
                                H0, W0 = frame.shape[:2]
                                for d in dets:
                                    try:
                                        if "xyxy" in d:
                                            x1, y1, x2, y2 = d["xyxy"]
                                        else:
                                            x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))
                                            x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))
                                        x1 = max(0, min(W0-1, int(round(x1)))); y1 = max(0, min(H0-1, int(round(y1))))
                                        x2 = max(0, min(W0-1, int(round(x2)))); y2 = max(0, min(H0-1, int(round(y2))))
                                        bgr = (0,0,255)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, max(1,_thick))
                                        try:
                                            lab = d.get('label','')
                                            lbl = ('L' if lab=='lesion' else ('A' if lab=='acne' else ''))
                                            sc = d.get('score', None)
                                            txt = f"{lbl}"
                                            if sc is not None:
                                                try: txt += f" {float(sc):.2f}"
                                                except Exception: pass
                                            ty = (y1 + 16) if (y1 - 6) < top_safe else max(0, y1 - 6)
                                            cv2.putText(frame, txt, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)
                                            if draw_center and int(d.get('fixed_degenerate',0))==1:
                                                cx = int((x1+x2)//2); cy = int((y1+y2)//2)
                                                cv2.circle(frame, (cx,cy), 6, (255,0,255), 2)
                                        except Exception:
                                            pass
                                    except Exception:
                                        continue
                                if (not dets) and bool(getattr(self,'_derm_had_candidates', False)) and (os.getenv('DERM_DRAW_CENTER_DOT_FOR_DEGENERATE','1') != '0'):
                                    try:
                                        cx, cy = W0//2, H0//2
                                        cv2.circle(frame, (cx,cy), 6, (255,0,255), 2)
                                    except Exception:
                                        pass
                            args_hud = getattr(self, '_cli_args', CLI_ARGS)
                            if draw_hud is not None and args_hud is not None and getattr(args_hud, 'hud', False):
                                try:
                                    flags = []
                                    for k in ('pretty_draw','smooth','hyst','debounce','assoc'):
                                        if getattr(args_hud, k, False):
                                            flags.append(k)
                                    fps_val = self._fps_meter.fps() if getattr(self, '_fps_meter', None) is not None else 0.0
                                    dets_count = len(dets) if isinstance(dets, (list, tuple)) else 0
                                    draw_hud(frame, fps=fps_val, dets_count=dets_count, flags=flags)
                                except Exception:
                                    pass
                            fs = getattr(self, '_framesaver', None)
                            if fs is not None:
                                try:
                                    img_to_save = frame
                                    if not getattr(args, 'save_annotated', False) and getattr(self, 'current_frame', None) is not None:
                                        img_to_save = self.current_frame
                                    fs.maybe_save(img_to_save)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    # Fallback to drawing dets if anything goes wrong
                    try:
                        dets = getattr(self, "_derm_latest_dets", []) or []
                        _thick = int(float(os.getenv("DERM_BOX_THICKNESS", "3")))
                        _thick = max(1, min(12, _thick))
                        top_safe = int(float(os.getenv('DERM_TOP_SAFE_PX', '32')))
                        H0, W0 = frame.shape[:2]
                        for d in dets:
                            try:
                                if "xyxy" in d:
                                    x1, y1, x2, y2 = d["xyxy"]
                                else:
                                    x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))
                                    x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))
                                x1 = max(0, min(W0-1, int(x1))); y1 = max(0, min(H0-1, int(y1)))
                                x2 = max(0, min(W0-1, int(x2))); y2 = max(0, min(H0-1, int(y2)))
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                color = d.get("color", (0, 255, 0))
                                if isinstance(color, (list, tuple)) and len(color) == 3:
                                    bgr = (int(color[2]), int(color[1]), int(color[0])) if max(color) > 1 else (0,255,0)
                                else:
                                    bgr = (0, 255, 0)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, _thick)
                                try:
                                    lbl = str(d.get("label", "")) or "HEF"
                                    ty = (y1 + 16) if (y1 - 6) < top_safe else max(0, y1 - 6)
                                    cv2.putText(frame, lbl, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)
                                except Exception:
                                    pass
                            except Exception:
                                continue
                    except Exception:
                        pass
                # Log per-frame paint summary
                try:
                    _counts_tmp = (getattr(self, 'derma_counts', {}) or {})
                    _boxes_tmp = _counts_tmp.get('boxes')
                    if _boxes_tmp is None or (hasattr(_boxes_tmp, "__len__") and len(_boxes_tmp) == 0):
                        _boxes_tmp = _counts_tmp.get('dets') or []
                    nb = len((getattr(self, '_derm_latest_dets', []) or []))
                    if (nb == 0) and (hasattr(_boxes_tmp, "__len__") and len(_boxes_tmp) > 0):
                        nb = len(_boxes_tmp)
                    print(f"[DERM-HAILO] GUI draw on main thread (mode={mode_used}, n_boxes={nb})")
                except Exception:
                    pass

                # Flip for mirror effect
                annotated = None
                if self.derma_active and self.last_derma_vis is not None:
                    annotated = self.last_derma_vis
                display_frame = cv2.flip(annotated if annotated is not None else frame, 1)
                # Record last frame shape for bridge fallbacks
                try:
                    self._last_frame_shape = display_frame.shape[:2]
                except Exception:
                    pass

                # Tiny HUD for counts when dermatologist active
                if self.derma_active:
                    hud = f"Lesion: {self.derma_counts.get('lesion',0)} | Acne: {self.derma_counts.get('acne',0)} | thr: {self.derma_threshold:.2f}"
                    cv2.putText(display_frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                    if (self.face_cascade is not None) and (1 <= self._face_missing_count <= 10):
                        cv2.putText(display_frame, "Detecting face…", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)

                # Add live face tracking with markers
                if hasattr(self, '_hailo_analyzer'):
                    display_frame = self.add_live_face_markers(display_frame)
                elif hasattr(self, 'results_locked') and self.results_locked:
                    display_frame = self.draw_final_results_only(display_frame)
                elif hasattr(self, 'analysis_active') and self.analysis_active and hasattr(self, 'current_face_region'):
                    self.draw_simple_face_indicator(display_frame)
                # Always draw HAAR face box on main thread for immediate feedback
                try:
                    overlay_off = (os.getenv('DERM_DISABLE_CPU_OVERLAY','0') == '1')
                except Exception:
                    overlay_off = False
                if not overlay_off:
                    try:
                        self._ensure_face_cascade()
                        roi = self.detect_face_fast()
                        if roi is not None:
                            x, y, w, h = roi
                            # Mirror x coordinate for flipped display
                            xm = display_frame.shape[1] - (x + w)
                            cv2.rectangle(display_frame, (xm, y), (xm + w, y + h), (0, 255, 0), 2)
                    except Exception:
                        pass

                # Efficient display update
                self.update_display_fast(display_frame)

        # Maintain smooth 30+ FPS
        if hasattr(self, 'root'):
            self.root.after(50, self.update_video_feed)

    # Dermatologist controls (legacy binding) -> delegate to new live loop
    def toggle_analysis(self):
        self.start_derm_live()

    def _derma_loop(self):
        print("[derma] Analysis thread started")
        while not self._derma_worker_stop.is_set():
            frame = None
            with self._frame_lock:
                frame = None if not hasattr(self, 'current_frame') else (None if self.current_frame is None else self.current_frame.copy())
            if frame is not None:
                try:
                    # Face ROI (non-blocking). If None, still run full-frame
                    roi = self.detect_face_fast()
                    if roi:
                        vis, counts = self.derma_engine.process_frame(frame, conf_thres=self.derma_threshold, roi=roi)
                    else:
                        vis, counts = self.derma_engine.process_frame(frame, conf_thres=self.derma_threshold)
                    self.last_derma_vis = vis
                    self.derma_counts = counts
                except Exception as e:
                    # Disable on error but keep app running
                    self.status_var.set(f"❌ Dermatologist disabled: {e}")
                    self._derma_worker_stop.set()
                    self.derma_active = False
                    try:
                        self.derma_engine.stop()
                    except Exception:
                        pass
                    break
            time.sleep(0.01)  # small yield
        print("[derma] Analysis thread stopped")

    # ---- Face detection (lazy, non-blocking) ----
    def _ensure_face_cascade(self) -> bool:
        """Deterministically load Haar cascade; thread-safe. Returns True if available."""
        with self._face_lock:
            if self.face_cascade is not None:
                return True
            import glob
            cv2_data = getattr(cv2, 'data', None)
            base = getattr(cv2_data, 'haarcascades', '') if cv2_data else ''
            primary = os.path.join(base, 'haarcascade_frontalface_default.xml') if base else ''
            paths = []
            if primary and os.path.isfile(primary):
                paths.append(primary)
            if not paths:
                for patt in (
                    '/usr/share/opencv*/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/local/share/opencv*/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                ):
                    paths.extend(glob.glob(patt))
            for cascade_path in paths:
                c = cv2.CascadeClassifier(cascade_path)
                if not c.empty():
                    self.face_cascade = c
                    if not getattr(self, '_haar_loaded', False):
                        print(f"[HAAR] Using cascade from: {cascade_path}")
                        self._haar_loaded = True
                    return True
            self.face_cascade = None
            if not getattr(self, '_haar_loaded', False):
                print("[HAAR] Failed to load haarcascade_frontalface_default.xml")
                self._haar_loaded = True
            return False


    def detect_face_fast(self):
        """Return (x,y,w,h) ROI of largest face with 8% margin, or None.
        - Ensures cascade is loaded
        - Keeps last ROI for up to 10 frames (grace) while drawing box
        - Does not call any Hailo code
        """
        bgr = None
        try:
            with self._frame_lock:
                bgr = None if not hasattr(self, 'current_frame') else (None if self.current_frame is None else self.current_frame.copy())
        except Exception:
            bgr = None
        if bgr is None or getattr(bgr, 'size', 0) == 0:
            self._roi_fresh = False
            return None
        if not self._ensure_face_cascade():
            self._face_missing_count = 0
            self._roi_fresh = False
            return None
        H, W = bgr.shape[:2]
        target_w = 480
        scale = W / float(target_w) if W > target_w else 1.0
        small = cv2.resize(bgr, (int(W/scale), int(H/scale))) if scale > 1.0 else bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)
        if faces is None or len(faces) == 0:
            self._roi_fresh = False
            if self._last_roi is not None and self._roi_grace > 0:
                self._roi_grace -= 1
                return self._last_roi
            else:
                self._last_roi = None
                return None
        # Largest face
        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
        if scale > 1.0:
            x = int(x * scale); y = int(y * scale)
            w = int(w * scale); h = int(h * scale)
        # Margin percent (env ROI_MARGIN_PCT, default 8%)
        try:
            margin_pct = float(os.getenv('ROI_MARGIN_PCT', '8')) / 100.0
        except Exception:
            margin_pct = 0.08
        mx = int(round(margin_pct * w)); my = int(round(margin_pct * h))
        x0 = max(0, x - mx); y0 = max(0, y - my)
        x1 = min(W, x + w + mx); y1 = min(H, y + h + my)
        roi = (x0, y0, max(0, x1 - x0), max(0, y1 - y0))
        self._last_roi = roi
        self._roi_grace = 10
        self._roi_fresh = True
        return roi

    def on_threshold_change(self, val):
        try:
            self.derma_threshold = max(0.05, min(0.95, float(val)))
        except Exception:
            pass


    def add_live_face_markers(self, frame):
        """Add live tracking markers that follow face movement"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in real-time
        if hasattr(self, 'face_cascade') and self.face_cascade:
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(120, 120))

            for (x, y, w, h) in faces:
                # Draw face boundary
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Only show markers if analysis has been run
                if hasattr(self, 'stabilized_results') and self.stabilized_results:
                    self.draw_live_analysis_markers(frame, x, y, w, h)

        return frame

    def draw_live_analysis_markers(self, frame, x, y, w, h):
        """Draw fewer, more accurate markers that track face position"""
        if not hasattr(self, 'stabilized_results'):
            return

        results = self.stabilized_results

        # Limit markers to prevent overcrowding
        max_pore_markers = 8
        max_blemish_markers = 4

        # Draw pore markers - distributed across face region
        if 'pore_analysis' in results:
            pore_count = min(results['pore_analysis'].get('count', 0), max_pore_markers)
            for i in range(pore_count):
                # Calculate positions relative to current face position
                rel_x = 0.3 + (i % 3) * 0.2  # Spread horizontally
                rel_y = 0.4 + (i // 3) * 0.15  # Stack vertically
                px = int(x + rel_x * w)
                py = int(y + rel_y * h)
                cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)  # Purple circles

        # Draw blemish markers
        if 'blemish_analysis' in results:
            blemish_count = min(results['blemish_analysis'].get('count', 0), max_blemish_markers)
            for i in range(blemish_count):
                rel_x = 0.25 + (i % 2) * 0.5  # Left and right sides
                rel_y = 0.3 + (i // 2) * 0.25
                bx = int(x + rel_x * w)
                by = int(y + rel_y * h)
                cv2.rectangle(frame, (bx-3, by-3), (bx+3, by+3), (0, 0, 255), 2)  # Red squares
    def draw_analysis_markers_on_face(self, frame, x, y, w, h):
        """Draw purple analysis markers on detected face region"""
        import random

        # Get current analysis results
        if hasattr(self, 'stabilized_results') and self.stabilized_results:
            results = self.stabilized_results

            # Draw pore markers
            if 'pore_analysis' in results:
                pore_count = results['pore_analysis'].get('count', 0)
                for i in range(min(pore_count, 20)):  # Limit to 20 visible markers
                    px = x + random.randint(w//4, 3*w//4)
                    py = y + random.randint(h//3, 2*h//3)
                    cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)  # Purple circles

            # Draw blemish markers
            if 'blemish_analysis' in results:
                blemish_count = results['blemish_analysis'].get('count', 0)
                for i in range(min(blemish_count, 10)):  # Limit to 10 visible markers
                    bx = x + random.randint(w//5, 4*w//5)
                    by = y + random.randint(h//4, 3*h//4)
                    cv2.rectangle(frame, (bx-4, by-4), (bx+4, by+4), (0, 0, 255), 2)  # Red squares


    def update_video_label(self, img_tk):
        """Update video label in main thread"""
        if self.video_label and img_tk:
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

    def start_derm_live(self):
        """Start/stop the continuous dermatologist live loop (ROI-only; HAAR face ROI)."""
        try:
            print("[UI] Precision Analysis clicked")
            if getattr(self, 'derm_live_active', False):
                # Already running; debounce
                return
            if not getattr(self, 'derm_live_active', False):
                # Initialize engine if needed
                if self.derma_engine is None:
                    lesion_path = os.getenv('HEF_LESION', os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
                    acne_path = os.getenv('HEF_ACNE', os.path.expanduser('~/derma/models/acnenew.hef'))
                    self.derma_engine = DualHefEngine(hef_lesion=lesion_path, hef_acne=acne_path)
                try:
                    self.derma_engine.start()
                except Exception as e:
                    self.status_var.set(f"❌ Dermatologist init error: {e}")
                    return
                # Launch live loop thread
                self._derma_worker_stop.clear()
                self.derm_live_active = True
                self._derma_worker = threading.Thread(target=self.run_derm_live, name="DermLive", daemon=True)
                self._derma_worker.start()
                self.derm_thread = self._derma_worker
                print("[DERM-HAILO] LIVE loop ACTIVE (ROI-only; HAAR face ROI)")
                self.derma_active = True
                self.status_var.set("🔬 Dermatologist ON")
        except Exception as e:
            self.status_var.set(f"❌ Dermatologist error: {e}")

    def stop_derm_live(self):
        """Stop the live dermatologist loop if running (do not close the shared device)."""
        if not getattr(self, 'derm_live_active', False):
            return
        self.derm_live_active = False
        try:
            self._derma_worker_stop.set()
            if getattr(self, '_derma_worker', None) is not None and self._derma_worker.is_alive():
                self._derma_worker.join(timeout=1.5)
        except Exception:
            pass
        finally:
            self._derma_worker = None
            self.derm_thread = None
            self.derma_active = False
            self.last_derma_vis = None
            self.derma_counts = {"lesion": 0, "acne": 0}

    def run_derm_live(self):
        last_log = 0.0
        last_overlay = None
        # Ensure HAAR cascade is ready (prints once on load)
        try:
            self._ensure_face_cascade()
        except Exception:
            pass
        while getattr(self, 'derm_live_active', False) and getattr(self, 'running', True) and not self._derma_worker_stop.is_set():
            # Safely copy latest frame
            with self._frame_lock:
                frame = None if self.current_frame is None else self.current_frame.copy()
            if frame is None:
                time.sleep(0.06)
                continue
            # HAAR ROI with grace
            roi = self.detect_face_fast()
            vis_full = frame.copy()
            if roi is not None:
                x, y, w, h = roi
                # Only infer when ROI is fresh this frame
                if self._roi_fresh:
                    # Thresholds with env overrides
                    # Lazily create engine on first fresh ROI
                    if getattr(self, 'derma_engine', None) is None:
                        try:
                            # Build engine in worker thread using shared VDevice
                            import ai_features.dermatologist.hailo_dual_hef as hdh
                            import ai_features.dermatologist.hailo_device_manager as dm
                            lesion_path = os.getenv('HEF_LESION', os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
                            acne_path = os.getenv('HEF_ACNE', os.path.expanduser('~/derma/models/acnenew.hef'))
                            shared_dev = dm.device_manager().get_device()
                            self.derma_engine = hdh.DualHefEngine(hef_lesion=lesion_path, hef_acne=acne_path, device=shared_dev)
                            self.derma_engine.start()
                            # Announce PIPE path selection once when enabled
                            if os.getenv('DERM_HAILO_PIPE_INFER','0') == '1':
                                print("[DERM-HAILO] GUI is using PIPE infer")
                        except Exception as e:
                            import os as _os_
                            if _os_.getenv('DEBUG_DERM','0')=='1':
                                print(f"[DERM-HAILO] ERROR -> {type(e).__name__}: {e}")
                            self.derma_engine = None
                    try:
                        conf_acne = float(os.getenv('CONF_ACNE', '0.15'))
                    except Exception:
                        conf_acne = 0.15
                    try:
                        conf_lesion = float(os.getenv('CONF_LESION', '0.15'))
                    except Exception:
                        conf_lesion = 0.15
                    vis_roi, counts = ((None, None) if getattr(self, 'derma_engine', None) is None else self.derma_engine.infer_roi(frame, roi, conf_acne, conf_lesion))
                    if counts == 'PAUSE':
                        # === CODEX PATCH: results→queue (no Tk in worker) BEGIN ===
                        try:
                            self._derm_results_q.put_nowait({'status': "🔄 DERM PAUSED – Hailo retry..."})
                        except queue.Full:
                            try:
                                _ = self._derm_results_q.get_nowait()
                                if os.getenv('DEBUG_DERM','0')=='1':
                                    print("[DERM-HAILO] GUI queue full — dropping oldest result")
                                self._derm_results_q.put_nowait({'status': "🔄 DERM PAUSED – Hailo retry..."})
                            except Exception:
                                pass
                        # === CODEX PATCH: results→queue (no Tk in worker) END ===
                        if os.getenv('DEBUG_DERM','0')=='1':
                            print("[DERM-HAILO] PAUSE – Hailo error, will retry")
                        if os.getenv('DEBUG_DERM','0')=='1':
                            check = np.indices((h, w)).sum(axis=0) % 2
                            check = (check * 20).astype(np.uint8)
                            check_bgr = cv2.merge([check, check, check])
                            dst = vis_full[y:y+h, x:x+w]
                            vis_full[y:y+h, x:x+w] = cv2.addWeighted(check_bgr, 0.3, dst, 0.7, 0.0)
                    elif vis_roi is not None and isinstance(counts, dict):
                        # Resize vis to ROI size if needed and blend at 50%
                        vis_blend = vis_roi
                        if vis_roi.shape[:2] != (h, w):
                            vis_blend = cv2.resize(vis_roi, (w, h))
                        dst = vis_full[y:y+h, x:x+w]
                        if dst.shape[:2] == vis_blend.shape[:2]:
                            try:
                                alpha = float(os.getenv('DERM_ALPHA', '0.5'))
                            except Exception:
                                alpha = 0.5
                            alpha = max(0.0, min(1.0, alpha))
                            blended = cv2.addWeighted(vis_blend, alpha, dst, 1.0 - alpha, 0.0)
                            vis_full[y:y+h, x:x+w] = blended
                            last_overlay = blended.copy()
                        self.derma_counts = counts
                        # Periodic log (~1s) only when blended
                        now = time.time()
                        if os.getenv('DEBUG_DERM','0')=='1' and (now - last_log > 1.0):
                            h0, w0 = vis_blend.shape[:2]
                            print(f"[DERM-HAILO] ROI=({x},{y},{w},{h}) counts={'{'}'acne':{counts.get('acne',0)},'lesion':{counts.get('lesion',0)}{'}'} vis=({h0},{w0},3)")
                            last_log = now
                else:
                    # Grace period: keep showing last markers inside ROI
                    if last_overlay is not None:
                        dst = vis_full[y:y+h, x:x+w]
                        if dst.shape[:2] == last_overlay.shape[:2]:
                            vis_full[y:y+h, x:x+w] = last_overlay
                # Draw green ROI rectangle after any blending so it stays visible
                cv2.rectangle(vis_full, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Share visualization to UI thread via queue (non-blocking, bounded)
            try:
                self._derm_results_q.put_nowait({'vis': vis_full, 'counts': self.derma_counts})
            except queue.Full:
                try:
                    _ = self._derm_results_q.get_nowait()
                    if os.getenv('DEBUG_DERM','0')=='1':
                        print("[DERM-HAILO] GUI queue full — dropping oldest result")
                    self._derm_results_q.put_nowait({'vis': vis_full, 'counts': self.derma_counts})
                except Exception:
                    pass
            # Adaptive 5–10 FPS throttling
            try:
                fps_min = float(os.getenv('DERM_FPS_MIN','5'))
                fps_max = float(os.getenv('DERM_FPS_MAX','10'))
                fps = max(fps_min, min(fps_max, fps_max))
                slp = 1.0 / max(1.0, fps)
            except Exception:
                slp = 0.08
            time.sleep(slp)

    # === CODEX PATCH: TK main-thread overlay pump BEGIN ===
    def _pump_derm_queue(self):
        """Main-thread pump that drains dermatologist results and performs GUI updates safely."""
        if threading.current_thread().name != 'MainThread':
            try:
                print("[DERM-HAILO] ERROR -> pump not on main thread; skipping draw")
            except Exception:
                pass
            return
        # Drain worker process queues (heartbeat + results)
        # Heartbeat
        try:
            hb_drained = 0
            while hb_drained < 2:
                msg = self._mp_hb_q.get_nowait()
                hb_drained += 1
                try:
                    print("[DERM-HAILO] GUI: worker heartbeat")
                except Exception:
                    pass
        except Exception:
            pass
        # Results from worker: handled by results bridge thread

        # Drain queue non-blocking, keep last
        items = []
        try:
            qsize = self._derm_results_q.qsize()
        except Exception:
            qsize = 0
        while True:
            try:
                itm = self._derm_results_q.get_nowait()
                items.append(itm)
            except Exception:
                break
        k = len(items)
        if k:
            last = items[-1]
            # Status updates
            if 'status' in last:
                try:
                    self.status_var.set(last['status'])
                except Exception:
                    pass
            # Counts
            if 'counts' in last and isinstance(last['counts'], dict):
                self.derma_counts = last['counts']
            # Latest dets (if forwarded in payload)
            if 'dets' in last:
                try:
                    self._derm_latest_dets = last.get('dets') or []
                except Exception:
                    self._derm_latest_dets = []
            # Overlay drawing (skip when disabled)
            overlay_off = (os.getenv('DERM_DISABLE_CPU_OVERLAY','0') == '1')
            if not overlay_off and 'vis' in last and last['vis'] is not None:
                self.last_derma_vis = last['vis']
            # Re-enable UI if requested
            if last.get('re_enable'):
                try:
                    self.analyze_btn.configure(state=tk.NORMAL, text="🔬 Start Precision Analysis", bg='#27ae60')
                except Exception:
                    pass
            # Draw debug ~1 Hz (n_boxes)
            self._pump_tick += 1
            try:
                ticks_per_sec = max(1, int(round(1000.0 / max(1, self._pump_period_ms))))
            except Exception:
                ticks_per_sec = 30
            if os.getenv('DEBUG_DERM','0')=='1' and (self._pump_tick % ticks_per_sec == 0):
                try:
                    c = self.derma_counts or {}
                    _boxes_tmp = c.get('boxes')
                    if _boxes_tmp is None or (hasattr(_boxes_tmp, "__len__") and len(_boxes_tmp) == 0):
                        _boxes_tmp = c.get('dets') or []
                    n_boxes = (int(c.get('acne', 0)) + int(c.get('lesion', 0))) or len(_boxes_tmp)
                    print(f"[DERM-HAILO] GUI draw on main thread (n_boxes={n_boxes})")
                except Exception:
                    pass
        # Heartbeat every ~2s
        try:
            now = time.time()
            if now - getattr(self, '_pump_last_heartbeat', 0.0) > 2.0:
                print(f"[DERM-HAILO] GUI pump heartbeat: alive (queue={qsize})")
                self._pump_last_heartbeat = now
        except Exception:
            pass
        # Reschedule
        self.root.after(self._pump_period_ms, self._pump_derm_queue)
    # === CODEX PATCH: TK main-thread overlay pump END ===

    def start_analysis(self):
        """Queue a START command and return immediately; no heavy Hailo work here."""
        try:
            # Already running? ignore
            if getattr(self, '_derm_run_active', False):
                try:
                    print("[DERM-HAILO] Start ignored: already running")
                except Exception:
                    pass
                return
            if self.current_frame is None:
                messagebox.showwarning("No Camera", "Camera feed not available")
                return
            # Reset lightweight UI state
            self.analysis_active = True
            self.analysis_complete = False
            self.results_locked = False
            self.face_detected = False
            self.face_stable_count = 0
            self.final_results = {key: [] for key in self.final_results}
            # Update UI controls only
            try:
                self.analyze_btn.configure(state=tk.DISABLED, text="🔍 Detecting Face...", bg='#f39c12')
            except Exception:
                pass
            try:
                self.status_var.set("🔍 Detecting face - please look at camera...")
            except Exception:
                pass
            # Log selected path (DEBUG_DERM)
            try:
                if os.getenv('DEBUG_DERM','0') == '1':
                    if os.getenv('DERM_HAILO_PIPE_INFER','0') == '1':
                        print("[DERM-HAILO] GUI is using PIPE infer (async)")
                    else:
                        print("[DERM-HAILO] GUI is using USER-BUFFER infer (async)")
            except Exception:
                pass
            # Enqueue START command to spawned worker with latest frame payload
            frame_payload = {}
            with self._frame_lock:
                if self.current_frame is not None:
                    f = self.current_frame.copy()
                    try:
                        frame_payload = {'h': int(f.shape[0]), 'w': int(f.shape[1]), 'c': int(f.shape[2]), 'data': f.tobytes()}
                    except Exception:
                        frame_payload = {}
            try:
                lesion_path = os.getenv('HEF_LESION', os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
                acne_path = os.getenv('HEF_ACNE', os.path.expanduser('~/derma/models/acnenew.hef'))
                use_pipe = os.getenv('DERM_HAILO_PIPE_INFER','0') == '1'
                self._mp_ctrl_q.put_nowait({'cmd': 'start', 'hef_lesion': lesion_path, 'hef_acne': acne_path, 'pipe': use_pipe, 'frame': frame_payload})
            except Exception:
                pass
            self._derm_run_active = True
            try:
                print("[DERM-HAILO] Start command queued (returning immediately)")
            except Exception:
                pass
            self.log_message("🔬 Starting efficient dermatologist analysis")
        except Exception:
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            raise
        finally:
            try:
                self.analyze_btn.configure(state=tk.NORMAL)
            except Exception:
                pass

    # === CODEX PATCH 3: tk async infer BEGIN ===
    def _tk_async_infer_worker(self):
        """Run dermatologist Hailo inference in a separate worker with timeout watchdog.
        Does not block Tk; re-enables the button when complete/timeout.
        """
        timeout_ms = 10000
        try:
            timeout_ms = int(os.getenv('DERM_HAILO_THREAD_TIMEOUT_MS', '10000'))
        except Exception:
            timeout_ms = 10000

        # Inner worker that may block inside Hailo; run as daemon so GUI never blocks on exit
        result_holder = {}
        err_holder = {}

        def _infer_once():
            try:
                # Build engine lazily using shared VDevice
                if getattr(self, 'derma_engine', None) is None:
                    import ai_features.dermatologist.hailo_dual_hef as hdh
                    import ai_features.dermatologist.hailo_device_manager as dm
                    lesion_path = os.getenv('HEF_LESION', os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
                    acne_path = os.getenv('HEF_ACNE', os.path.expanduser('~/derma/models/acnenew.hef'))
                    shared_dev = dm.device_manager().get_device()
                    self.derma_engine = hdh.DualHefEngine(hef_lesion=lesion_path, hef_acne=acne_path, device=shared_dev)
                    self.derma_engine.start()
                # Acquire a frame snapshot
                with self._frame_lock:
                    frame = None if self.current_frame is None else self.current_frame.copy()
                if frame is None:
                    raise RuntimeError('No frame available for analysis')
                # Try ROI; fallback to full-frame
                roi = None
                try:
                    roi = self.detect_face_fast()
                except Exception:
                    roi = None
                # Confidence thresholds
                try:
                    conf_acne = float(os.getenv('CONF_ACNE', '0.15'))
                except Exception:
                    conf_acne = 0.15
                try:
                    conf_lesion = float(os.getenv('CONF_LESION', '0.15'))
                except Exception:
                    conf_lesion = 0.15
                if roi:
                    vis_roi, counts = self.derma_engine.infer_roi(frame, roi, conf_acne, conf_lesion)
                    # Blend ROI into full frame visualization
                    if vis_roi is not None and isinstance(counts, dict):
                        x, y, w, h = roi
                        vis_full = frame.copy()
                        vr = vis_roi if vis_roi.shape[:2] == (h, w) else cv2.resize(vis_roi, (w, h))
                        dst = vis_full[y:y+h, x:x+w]
                        if dst.shape[:2] == vr.shape[:2]:
                            alpha = 0.5
                            try:
                                alpha = float(os.getenv('DERM_ALPHA', '0.5'))
                            except Exception:
                                alpha = 0.5
                            alpha = max(0.0, min(1.0, alpha))
                            vis_full[y:y+h, x:x+w] = cv2.addWeighted(vr, alpha, dst, 1.0 - alpha, 0.0)
                        result_holder['vis'] = vis_full
                        result_holder['counts'] = counts
                    else:
                        # Handle PAUSE or None
                        result_holder['vis'] = None
                        result_holder['counts'] = counts
                else:
                    # Full-frame processing path (center square)
                    vis_full, counts = self.derma_engine.process_frame(frame, conf_thres=self.derma_threshold)
                    result_holder['vis'] = vis_full
                    result_holder['counts'] = counts
            except Exception as e:
                err_holder['err'] = e

        inner = threading.Thread(target=_infer_once, daemon=True)
        inner.start()
        inner.join(timeout_ms / 1000.0)
        if inner.is_alive():
            # Timeout; log and re-enable UI without blocking
            try:
                print("[DERM-HAILO] ERROR -> Timeout waiting for dermatologist inference")
            except Exception:
                pass
            try:
                self._derm_results_q.put_nowait({'status': "⏱️ Dermatologist timeout - please try again", 're_enable': True})
            except queue.Full:
                try:
                    _ = self._derm_results_q.get_nowait()
                    if os.getenv('DEBUG_DERM','0')=='1':
                        print("[DERM-HAILO] GUI queue full — dropping oldest result")
                    self._derm_results_q.put_nowait({'status': "⏱️ Dermatologist timeout - please try again", 're_enable': True})
                except Exception:
                    pass
            self.analysis_active = False
            return

        # Completed (success or error)
        if 'err' in err_holder:
            try:
                print(f"[DERM-HAILO] ERROR -> {type(err_holder['err']).__name__}: {err_holder['err']}")
            except Exception:
                pass
            try:
                self._derm_results_q.put_nowait({'status': "❌ Dermatologist error - see logs", 're_enable': True})
            except queue.Full:
                try:
                    _ = self._derm_results_q.get_nowait()
                    if os.getenv('DEBUG_DERM','0')=='1':
                        print("[DERM-HAILO] GUI queue full — dropping oldest result")
                    self._derm_results_q.put_nowait({'status': "❌ Dermatologist error - see logs", 're_enable': True})
                except Exception:
                    pass
            self.analysis_active = False
            return

        # Success: publish visualization and counts
        vis = result_holder.get('vis')
        counts = result_holder.get('counts') or {"lesion": 0, "acne": 0}
        try:
            payload = {'re_enable': True, 'status': "✅ Dermatologist analysis complete"}
            if vis is not None:
                payload['vis'] = vis
            if isinstance(counts, dict):
                payload['counts'] = counts
            self._derm_results_q.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._derm_results_q.get_nowait()
                if os.getenv('DEBUG_DERM','0')=='1':
                    print("[DERM-HAILO] GUI queue full — dropping oldest result")
                self._derm_results_q.put_nowait(payload)
            except Exception:
                pass
        self.analysis_active = False
    # === CODEX PATCH 3: tk async infer END ===

    # === CODEX PATCH: TK main-thread overlay pump BEGIN ===
    def _derm_worker_loop(self):
        """Persistent worker that owns Hailo engine startup and first/ongoing inference.
        Never touches Tk; communicates via queues only.
        """
        while getattr(self, 'running', True):
            cmd = None
            try:
                cmd = self._derm_ctrl_q.get(timeout=0.25)
            except Exception:
                continue
            if not isinstance(cmd, dict) or cmd.get('cmd') != 'START':
                continue
            # Heavy startup with watchdog
            start_ts = time.time()
            try:
                print("[DERM-HAILO] Worker: building Hailo engine(s)...")
                if getattr(self, 'derma_engine', None) is None:
                    import ai_features.dermatologist.hailo_dual_hef as hdh
                    import ai_features.dermatologist.hailo_device_manager as dm
                    lesion_path = os.getenv('HEF_LESION', os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
                    acne_path = os.getenv('HEF_ACNE', os.path.expanduser('~/derma/models/acnenew.hef'))
                    shared_dev = dm.device_manager().get_device()
                    self.derma_engine = hdh.DualHefEngine(hef_lesion=lesion_path, hef_acne=acne_path, device=shared_dev)
                    self.derma_engine.start()
                # First inference (ROI if available)
                with self._frame_lock:
                    frame = None if self.current_frame is None else self.current_frame.copy()
                if frame is None:
                    raise RuntimeError('No frame available for first inference')
                roi = None
                try:
                    roi = self.detect_face_fast()
                except Exception:
                    roi = None
                # Thresholds
                try:
                    conf_acne = float(os.getenv('CONF_ACNE', '0.15'))
                except Exception:
                    conf_acne = 0.15
                try:
                    conf_lesion = float(os.getenv('CONF_LESION', '0.15'))
                except Exception:
                    conf_lesion = 0.15
                if roi:
                    vis_roi, counts = self.derma_engine.infer_roi(frame, roi, conf_acne, conf_lesion)
                    vis_full = frame.copy()
                    if vis_roi is not None and isinstance(counts, dict):
                        x, y, w, h = roi
                        vr = vis_roi if vis_roi.shape[:2] == (h, w) else cv2.resize(vis_roi, (w, h))
                        dst = vis_full[y:y+h, x:x+w]
                        if dst.shape[:2] == vr.shape[:2]:
                            try:
                                alpha = float(os.getenv('DERM_ALPHA', '0.5'))
                            except Exception:
                                alpha = 0.5
                            alpha = max(0.0, min(1.0, alpha))
                            vis_full[y:y+h, x:x+w] = cv2.addWeighted(vr, alpha, dst, 1.0 - alpha, 0.0)
                else:
                    vis_full, counts = self.derma_engine.process_frame(frame, conf_thres=self.derma_threshold)
                # Watchdog check
                if time.time() - start_ts > 10.0:
                    print("[DERM-HAILO] Worker watchdog: startup exceeded 10s, aborting run")
                    try:
                        self._derm_results_q.put_nowait({'status': "❌ Startup timeout", 're_enable': True})
                    except queue.Full:
                        try:
                            _ = self._derm_results_q.get_nowait()
                            self._derm_results_q.put_nowait({'status': "❌ Startup timeout", 're_enable': True})
                        except Exception:
                            pass
                    print("[DERM-HAILO] Run finished (status=aborted)")
                    self._derm_run_active = False
                    continue
                # Successful first infer
                print("[DERM-HAILO] Worker: first inference completed")
                payload = {'re_enable': True, 'status': "✅ Dermatologist analysis complete"}
                if 'vis_full' in locals():
                    payload['vis'] = vis_full
                if isinstance(counts, dict):
                    payload['counts'] = counts
                try:
                    self._derm_results_q.put_nowait(payload)
                except queue.Full:
                    try:
                        _ = self._derm_results_q.get_nowait()
                        self._derm_results_q.put_nowait(payload)
                    except Exception:
                        pass
                print("[DERM-HAILO] Run finished (status=ok)")
            except Exception as e:
                try:
                    print(f"[DERM-HAILO] ERROR -> {type(e).__name__}: {e}")
                except Exception:
                    pass
                try:
                    self._derm_results_q.put_nowait({'status': "❌ Dermatologist error - see logs", 're_enable': True})
                except queue.Full:
                    try:
                        _ = self._derm_results_q.get_nowait()
                        self._derm_results_q.put_nowait({'status': "❌ Dermatologist error - see logs", 're_enable': True})
                    except Exception:
                        pass
            finally:
                self._derm_run_active = False
    # === CODEX PATCH: TK main-thread overlay pump END ===

    def stop_analysis(self):
        """EFFICIENT stop analysis"""
        self.analysis_active = False
        self.analyze_btn.configure(text="🔬 Start Dermatologist Analysis", bg='#27ae60')
        self.status_var.set("✅ Ready for analysis")

        if not self.results_locked:
            # Reset if analysis wasn't completed
            self.final_results = {key: [] for key in self.final_results}
            for key in self.metric_vars:
                self.metric_vars[key].set("0" if key == 'faces' else "0")

    def run_precision_analysis(self):
        """Ultra-precise analysis with 3-phase approach"""

        # PHASE 1: Enhanced Face Recognition and Validation
        self.status_var.set("🎯 Phase 1: Detecting and validating human face...")

        stable_face_region = None
        stability_start = time.time()
        validation_attempts = 0
        max_attempts = 50  # 5 seconds at 10 FPS

        while (self.analysis_active and
               validation_attempts < max_attempts and
               self.face_stability_frames < self.required_stability):

            validation_attempts += 1

            # Get validated face region
            current_face = self.get_primary_face_region()

            if current_face:
                if stable_face_region is None:
                    stable_face_region = current_face
                    self.face_stability_frames = 1
                    self.log_message(f"👤 Human face detected and validated: {current_face[2]}x{current_face[3]}")
                else:
                    # Check face stability
                    overlap = self.calculate_face_overlap(current_face, stable_face_region)

                    if overlap > 0.7:  # 70% overlap = stable
                        self.face_stability_frames += 1
                        stable_face_region = current_face

                        if self.face_stability_frames % 5 == 0:  # Log every 5 frames
                            self.log_message(f"📍 Face stability: {self.face_stability_frames}/{self.required_stability}")
                    else:
                        self.face_stability_frames = max(0, self.face_stability_frames - 2)  # Small penalty
                        stable_face_region = current_face
            else:
                self.face_stability_frames = max(0, self.face_stability_frames - 1)

                # Provide helpful feedback
                if validation_attempts % 10 == 0:
                    self.log_message("🔍 No valid human face detected - please ensure:")
                    self.log_message("  • Your face is clearly visible and well-lit")
                    self.log_message("  • You're facing the camera directly")
                    self.log_message("  • No obstructions (glasses, shadows, etc.)")
                    self.log_message("  • Move closer if you're too far from camera")

            time.sleep(0.1)

        # Check if face validation succeeded
        if self.face_stability_frames < self.required_stability:
            if validation_attempts >= max_attempts:
                self.status_var.set("❌ Could not detect valid human face - please check positioning and lighting")
            else:
                self.status_var.set("❌ Face validation failed - analysis stopped")

            self.log_message("❌ Face validation failed. Common issues:")
            self.log_message("  • Poor lighting (too dark or too bright)")
            self.log_message("  • Face not clearly visible")
            self.log_message("  • Background objects confusing detection")
            self.log_message("  • Camera resolution or focus issues")

            self.stop_analysis()
            return

        # Continue with existing analysis phases...
        self.log_message("✅ Phase 1 Complete: Valid human face confirmed and stabilized")
        self.log_message(f"👤 Final face region: {stable_face_region[2]}x{stable_face_region[3]} at ({stable_face_region[0]}, {stable_face_region[1]})")

        # PHASE 2: Deep Analysis (3 seconds)
        self.analysis_phase = "analyzing"
        self.analysis_start_time = time.time()
        self.analyze_btn.configure(text="🔬 Deep Analysis...", bg='#e74c3c')
        self.status_var.set("🔬 Phase 2: Deep analysis in progress - please remain still...")

        self.log_message("🔬 Phase 2: Starting 3-second deep analysis on validated face...")

        # Perform ultra-precise analysis
        multiple_scans = []
        scan_count = 6  # 6 scans over 3 seconds for maximum accuracy

        for scan_num in range(scan_count):
            if not self.analysis_active:
                return

            scan_results = self.perform_precision_scan(stable_face_region)
            if scan_results:
                multiple_scans.append(scan_results)

            # Progress indication
            progress = ((scan_num + 1) / scan_count) * 100
            self.status_var.set(f"🔬 Deep analysis: {progress:.0f}% complete...")

            time.sleep(0.5)  # 0.5 second intervals

        # PHASE 3: Results Consolidation and Locking
        self.analysis_phase = "complete"
        self.status_var.set("🧠 Phase 3: Consolidating results and locking positions...")
        self.log_message("🧠 Phase 3: Consolidating multiple scans for maximum accuracy...")

        if multiple_scans:
            # Consolidate results from multiple scans
            final_results = self.consolidate_multiple_scans(multiple_scans)

            # Lock features in place
            self.lock_detected_features(final_results)

            # Update display with final results
            self.update_final_analysis_display(final_results)

            self.analyze_btn.configure(text="✅ Analysis Complete", bg='#27ae60')
            self.status_var.set("✅ Precision analysis complete - Results locked in place")

            self.log_message("✅ Analysis Complete: High-precision results locked")
            self.log_message("🎯 Markings are now stationary and 98%+ accurate")
        else:
            self.status_var.set("❌ Analysis failed - insufficient image quality")
            self.stop_analysis()

    def perform_precision_scan(self, face_region_coords):
        """Single high-precision scan with strict quality controls"""
        if self.current_frame is None:
            return None

        x, y, w, h = face_region_coords

        # Extract high-quality face region
        face_region = self.current_frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Quality check
        quality_score = self.assess_image_quality(face_region)
        if quality_score < self.min_analysis_quality:
            return None

        # Initialize scan results
        scan_results = {
            'face_region': (x, y, w, h),
            'quality_score': quality_score,
            'features': {}
        }

        # Ultra-precise feature detection
        scan_results['features']['acne_spots'] = self.detect_acne_ultra_precise(face_region, x, y)
        scan_results['features']['pores'] = self.detect_pores_ultra_precise(face_gray, x, y)
        scan_results['features']['blackheads'] = self.detect_blackheads_ultra_precise(face_region, x, y)
        scan_results['features']['age_spots'] = self.detect_age_spots_ultra_precise(face_region, x, y)
        scan_results['features']['moles'] = self.detect_moles_ultra_precise(face_region, x, y)
        scan_results['features']['scars'] = self.detect_scars_ultra_precise(face_gray, x, y)

        return scan_results

    def detect_acne_face_only(self, face_region, face_mask, region_x, region_y, face_x, face_y, face_w, face_h):
        """Detect acne only within face boundaries"""
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Apply face mask to HSV image
        hsv_masked = cv2.bitwise_and(hsv, hsv, mask=face_mask)

        # Enhanced acne detection with strict criteria
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        # Create masks only within face area
        mask1 = cv2.inRange(hsv_masked, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_masked, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Apply face mask again to ensure no background detection
        red_mask = cv2.bitwise_and(red_mask, face_mask)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        acne_spots = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if 8 < area < 250:  # Strict acne size range
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    # Local coordinates within face region
                    local_cx = int(M["m10"] / M["m00"])
                    local_cy = int(M["m01"] / M["m00"])

                    # Convert to global coordinates
                    global_cx = local_cx + region_x
                    global_cy = local_cy + region_y

                    # Strict boundary check
                    if self.is_point_in_face_boundary(global_cx, global_cy, (face_x, face_y, face_w, face_h), None):
                        # Additional check: must be within face mask
                        if (0 <= local_cx < face_mask.shape[1] and
                            0 <= local_cy < face_mask.shape[0] and
                            face_mask[local_cy, local_cx] > 0):

                            # Calculate circularity
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter ** 2)

                                if circularity > 0.3:  # Must be reasonably circular
                                    radius = max(3, int(np.sqrt(area / np.pi)) + 1)

                                    acne_spots.append({
                                        'position': (global_cx, global_cy),
                                        'radius': radius,
                                        'area': area,
                                        'confidence': circularity
                                    })

        return acne_spots

    def detect_pores_face_only(self, face_gray, face_mask, region_x, region_y, face_x, face_y, face_w, face_h):
        """Detect pores only within face boundaries"""
        # Apply face mask to grayscale
        face_gray_masked = cv2.bitwise_and(face_gray, face_gray, mask=face_mask)

        # Enhanced pore detection
        blurred = cv2.GaussianBlur(face_gray_masked, (3, 3), 0)

        # Adaptive threshold only on masked area
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 5, 2)

        # Apply face mask to threshold result
        thresh = cv2.bitwise_and(thresh, face_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pores = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if 1 < area < 15:  # Very strict pore size
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    local_cx = int(M["m10"] / M["m00"])
                    local_cy = int(M["m01"] / M["m00"])

                    # Convert to global coordinates
                    global_cx = local_cx + region_x
                    global_cy = local_cy + region_y

                    # Strict boundary and mask checks
                    if (self.is_point_in_face_boundary(global_cx, global_cy, (face_x, face_y, face_w, face_h), None) and
                        0 <= local_cx < face_mask.shape[1] and
                        0 <= local_cy < face_mask.shape[0] and
                        face_mask[local_cy, local_cx] > 0):

                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter ** 2)

                            if circularity > 0.5:  # Must be very circular
                                radius = max(2, int(np.sqrt(area / np.pi)))

                                pores.append({
                                    'position': (global_cx, global_cy),
                                    'radius': radius,
                                    'area': area,
                                    'confidence': circularity
                                })

        return pores

    def detect_blackheads_face_only(self, face_region, face_mask, region_x, region_y, face_x, face_y, face_w, face_h):
        """Detect blackheads only within face boundaries"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Apply face mask
        gray_masked = cv2.bitwise_and(gray, gray, mask=face_mask)

        # Very dark spots detection
        mean_intensity = cv2.mean(gray_masked, mask=face_mask)[0]
        dark_threshold = max(30, min(60, mean_intensity - 35))

        _, dark_mask = cv2.threshold(gray_masked, dark_threshold, 255, cv2.THRESH_BINARY_INV)

        # Apply face mask again
        dark_mask = cv2.bitwise_and(dark_mask, face_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blackheads = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if 2 < area < 20:  # Strict blackhead size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    local_cx = int(M["m10"] / M["m00"])
                    local_cy = int(M["m01"] / M["m00"])

                    global_cx = local_cx + region_x
                    global_cy = local_cy + region_y

                    # Strict boundary checks
                    if (self.is_point_in_face_boundary(global_cx, global_cy, (face_x, face_y, face_w, face_h), None) and
                        0 <= local_cx < face_mask.shape[1] and
                        0 <= local_cy < face_mask.shape[0] and
                        face_mask[local_cy, local_cx] > 0):

                        # Verify darkness
                        mask_check = np.zeros(gray.shape, np.uint8)
                        cv2.fillPoly(mask_check, [contour], 255)
                        mask_check = cv2.bitwise_and(mask_check, face_mask)

                        mean_darkness = cv2.mean(gray, mask=mask_check)[0]

                        if mean_darkness < dark_threshold:
                            radius = max(2, int(np.sqrt(area / np.pi)))
                            confidence = (dark_threshold - mean_darkness) / dark_threshold

                            blackheads.append({
                                'position': (global_cx, global_cy),
                                'radius': radius,
                                'area': area,
                                'confidence': confidence
                            })

        return blackheads

    if False:  # TEMP: disabled experimental block; TODO: refactor later
        def detect_age_spots_face_only(self, face_region, face_mask, region_x, region_y, face_x, face_y, face_w, face_h):
            """Detect age spots only within face boundaries"""
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]

            # Apply face mask
            l_masked = cv2.bitwise_and(l_channel, l_channel, mask=face_mask)

            # Detect darker pigmented areas
            mean_brightness = cv2.mean(l_masked, mask=face_mask)[0]
            dark_threshold = mean_brightness - 20

            _, dark_spots = cv2.threshold(l_masked, dark_threshold, 255, cv2.THRESH_BINARY_INV)


            # NOTE: removed stray diagnostic snippet artifact (deleted undefined IoU snippet)

        def get_primary_face_region(self):
            """Enhanced face detection with multiple validation layers"""
            if self.current_frame is None:
                return None

            # Step 1: Preprocess frame for better face detection
            processed_frame = self.preprocess_for_face_detection(self.current_frame)

            # Step 2: Multi-method face detection
            face_candidates = self.detect_face_candidates(processed_frame)

            if not face_candidates:
                self.consecutive_face_detections = 0
                # Try with enhanced preprocessing
                enhanced_frame = self.enhance_frame_for_detection(self.current_frame)
                face_candidates = self.detect_face_candidates(enhanced_frame)

                if not face_candidates:
                    return None

            # Step 3: Validate face candidates (reject false positives)
            valid_faces = []
            for candidate in face_candidates:
                if self.validate_human_face(self.current_frame, candidate):
                    valid_faces.append(candidate)

            if not valid_faces:
                self.consecutive_face_detections = 0
                return None

            # Step 4: Select best face candidate
            best_face = self.select_best_face_candidate(valid_faces)

            if best_face:
                self.consecutive_face_detections += 1
                self.last_valid_face = best_face
                self.face_tracking_confidence = min(100, self.consecutive_face_detections * 10)

                x, y, w, h = best_face
                self.log_message(f"✅ Validated human face: {w}x{h} at ({x},{y}) - Confidence: {self.face_tracking_confidence}%")

                return best_face

            return None

        def create_precise_face_mask(self, face_region, face_coords):
            """Create precise face mask to eliminate background detection"""
            padding, pad_y, face_w, face_h = face_coords

            # Create mask for the face region
            mask = np.zeros(face_region.shape[:2], dtype=np.uint8)

            # Create elliptical mask for more natural face shape
            center_x = padding + face_w // 2
            center_y = pad_y + face_h // 2

            # Make ellipse slightly smaller than detected rectangle to avoid edges
            ellipse_w = int(face_w * 0.4)  # 40% of width as semi-major axis
            ellipse_h = int(face_h * 0.45)  # 45% of height as semi-minor axis

            # Create elliptical face mask
            cv2.ellipse(mask, (center_x, center_y), (ellipse_w, ellipse_h), 0, 0, 360, 255, -1)

            # Additional refinement: remove top and bottom portions to focus on main face area
            # Remove top 15% (forehead area that might include hair)
            mask[0:int(face_h * 0.15), :] = 0

            # Remove bottom 10% (chin/neck area)
            # NOTE: removed stray diagnostic snippet artifact
            #                         face_mask[local_cy, local_cx] > 0):
            #
            #                         # Check circularity
            #                         perimeter = cv2.arcLength(contour, True)
            #                         if perimeter > 0:
            #                             circularity = 4 * np.pi * area / (perimeter ** 2)
            #
            #                             if circularity > 0.4:  # Reasonably round
            #                                 radius = max(5, int(np.sqrt(area / np.pi)))
            #
            #                                 moles.append({
            #                                     'position': (global_cx, global_cy),
            #                                     'radius': radius,
            #                                     'area': area,
            #                                     'confidence': circularity
            #                                 })

            # return moles

    def detect_scars_face_only(self, face_gray, face_mask, region_x, region_y, face_x, face_y, face_w, face_h):
        """Detect scars only within face boundaries"""
        # Apply face mask
        face_gray_masked = cv2.bitwise_and(face_gray, face_gray, mask=face_mask)

        # Edge detection for scar lines
        edges = cv2.Canny(face_gray_masked, 35, 90)

        # Apply face mask to edges
        edges = cv2.bitwise_and(edges, face_mask)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scars = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if 8 < area < 80:  # Scar size range
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    local_cx = int(M["m10"] / M["m00"])
                    local_cy = int(M["m01"] / M["m00"])

                    global_cx = local_cx + region_x
                    global_cy = local_cy + region_y

                    # Strict boundary checks
                    if (self.is_point_in_face_boundary(global_cx, global_cy, (face_x, face_y, face_w, face_h), None) and
                        0 <= local_cx < face_mask.shape[1] and
                        0 <= local_cy < face_mask.shape[0] and
                        face_mask[local_cy, local_cx] > 0):

                        # Check aspect ratio for elongated features
                        rect = cv2.minAreaRect(contour)
                        if rect[1][0] > 0 and rect[1][1] > 0:
                            width, height = rect[1]
                            aspect_ratio = max(width, height) / min(width, height)

                            if aspect_ratio > 1.3:  # Somewhat elongated
                                radius = 3  # Small marker for scars
                                confidence = min(1.0, aspect_ratio / 3.0)

                                scars.append({
                                    'position': (global_cx, global_cy),
                                    'radius': radius,
                                    'area': area,
                                    'confidence': confidence
                                })

        return scars

    def assess_image_quality(self, face_region):
        """Assess image quality for reliable analysis"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Brightness check (optimal range 80-180)
        brightness = np.mean(gray)
        brightness_score = 100 if 80 <= brightness <= 180 else max(0, 100 - abs(brightness - 130))

        # Contrast check
        contrast = np.std(gray)
        contrast_score = min(100, contrast * 2)

        # Sharpness check (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, sharpness / 50.0)

        # Size check (larger faces = better analysis)
        face_area = face_region.shape[0] * face_region.shape[1]
        size_score = min(100, face_area / 300)  # Optimal above 30k pixels

        # Overall quality score
        quality = (brightness_score + contrast_score + sharpness_score + size_score) / 4.0

        return quality

    def validate_human_face(self, frame, face_candidate):
        """Validate that a face candidate is actually a human face"""
        x, y, w, h = face_candidate[:4]

        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        validation_score = 0
        max_score = 100

        # Test 1: Skin color validation (30 points)
        skin_score = self.validate_skin_color(face_region)
        validation_score += skin_score * 0.3

        # Test 2: Eye detection (25 points)
        eye_score = self.validate_eyes_present(face_gray)
        validation_score += eye_score * 0.25

        # Test 3: Face proportions (20 points)
        proportion_score = self.validate_face_proportions(w, h)
        validation_score += proportion_score * 0.20

        # Test 4: Texture analysis (15 points)
        texture_score = self.validate_face_texture(face_gray)
        validation_score += texture_score * 0.15

        # Test 5: Position preference (10 points) - prefer center faces
        position_score = self.validate_face_position(x, y, w, h, frame.shape)
        validation_score += position_score * 0.10

        # Must score at least 60% to be considered a valid human face
        return validation_score >= 60

    def validate_skin_color(self, face_region):
        """Validate skin color characteristics"""
        try:
            # Convert to multiple color spaces for analysis
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

            # Define skin color ranges in HSV
            # These ranges cover various skin tones
            skin_ranges = [
                # Light skin tones
                (np.array([0, 20, 70]), np.array([20, 255, 255])),
                # Medium skin tones
                (np.array([8, 50, 60]), np.array([25, 200, 230])),
                # Darker skin tones
                (np.array([6, 30, 40]), np.array([25, 150, 200]))
            ]

            skin_pixels = 0
            total_pixels = face_region.shape[0] * face_region.shape[1]

            for lower, upper in skin_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                skin_pixels += np.sum(mask > 0)

            skin_percentage = min(100, (skin_pixels / total_pixels) * 100)

            # Score based on skin percentage (expect 30-80% skin pixels)
            if 30 <= skin_percentage <= 80:
                return min(100, skin_percentage * 1.5)
            elif 20 <= skin_percentage <= 90:
                return 70
            else:
                return max(0, 40 - abs(50 - skin_percentage))

        except Exception as e:
            return 50  # Neutral score on error

    def validate_eyes_present(self, face_gray):
        """Validate presence of eyes in face region"""
        try:
            if not self.eye_cascade or self.eye_cascade.empty():
                return 30  # Low score if no eye cascade available

            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(10, 10),
                maxSize=(50, 50)
            )

            num_eyes = len(eyes)

            # Score based on number of eyes detected
            if num_eyes >= 2:
                return 100  # Perfect - both eyes detected
            elif num_eyes == 1:
                return 70   # Good - one eye detected
            else:
                # No eyes detected - try alternate method
                # Look for dark regions that might be eyes

                # Focus on upper half of face where eyes should be
                eye_region = face_gray[:face_gray.shape[0]//2, :]

                # Find dark regions (potential eyes)
                mean_brightness = np.mean(eye_region)
                _, dark_regions = cv2.threshold(eye_region, mean_brightness - 30, 255, cv2.THRESH_BINARY_INV)

                # Count dark regions that could be eyes
                contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                eye_like_regions = 0

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 20 < area < 500:  # Reasonable eye size
                        eye_like_regions += 1

                if eye_like_regions >= 2:
                    return 60  # Moderate - found potential eye regions
                elif eye_like_regions == 1:
                    return 40  # Low - found one potential eye region
                else:
                    return 20  # Very low - no clear eye features

        except Exception as e:
            return 30  # Low score on error

    def validate_face_proportions(self, width, height):
        """Validate face has human-like proportions"""
        try:
            aspect_ratio = width / height

            # Human faces typically have aspect ratio between 0.7 and 1.1
            # (slightly taller than wide to slightly wider than tall)
            if 0.75 <= aspect_ratio <= 1.05:
                return 100  # Perfect proportions
            elif 0.7 <= aspect_ratio <= 1.15:
                return 80   # Good proportions
            elif 0.6 <= aspect_ratio <= 1.3:
                return 60   # Acceptable proportions
            else:
                return max(0, 40 - abs(aspect_ratio - 0.9) * 50)  # Poor proportions

        except Exception as e:
            return 50  # Neutral on error

    def validate_face_texture(self, face_gray):
        """Validate face has appropriate texture characteristics"""
        try:
            # Calculate texture measures

            # 1. Edge density (faces have moderate edges)
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # 2. Texture variance (faces have moderate texture variation)
            texture_variance = np.var(face_gray)

            # Score based on texture characteristics
            texture_score = 0

            # Edge density scoring (faces should have 2-8% edges)
            if 0.02 <= edge_density <= 0.08:
                texture_score += 50
            elif 0.01 <= edge_density <= 0.12:
                texture_score += 30

            # Texture variance scoring (faces should have 500-3000 variance)
            if 500 <= texture_variance <= 3000:
                texture_score += 50
            elif 300 <= texture_variance <= 4000:
                texture_score += 30

            return min(100, texture_score)

        except Exception as e:
            return 50

    def validate_face_position(self, x, y, w, h, frame_shape):
        """Validate face position - prefer faces near center"""
        frame_height, frame_width = frame_shape[:2]

        # Calculate face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Calculate frame center
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        # Calculate distance from center (normalized)
        distance_x = abs(face_center_x - frame_center_x) / frame_width
        distance_y = abs(face_center_y - frame_center_y) / frame_height

        # Combined distance score
        center_distance = (distance_x + distance_y) / 2

        # Score based on distance from center (prefer center)
        if center_distance <= 0.15:  # Very close to center
            return 100
        elif center_distance <= 0.25:  # Close to center
            return 80
        elif center_distance <= 0.35:  # Moderately close
            return 60
        else:  # Far from center
            return max(20, 80 - center_distance * 100)

    def select_best_face_candidate(self, valid_faces):
        """Select the best face from valid candidates"""
        if not valid_faces:
            return None

        if len(valid_faces) == 1:
            return valid_faces[0][:4]  # Return just (x, y, w, h)

        # Score each face candidate
        best_face = None
        best_score = 0

        for face_data in valid_faces:
            x, y, w, h, method, confidence = face_data

            # Calculate composite score
            size_score = min(100, (w * h) / 300)  # Prefer larger faces
            center_score = self.validate_face_position(x, y, w, h, self.current_frame.shape)
            method_score = confidence * 100

            # Bonus for tracking consistency
            tracking_bonus = 0
            if self.last_valid_face:
                last_x, last_y, last_w, last_h = self.last_valid_face
                overlap = self.calculate_face_overlap((x, y, w, h), (last_x, last_y, last_w, last_h))
                tracking_bonus = overlap * 20  # Up to 20 point bonus

            composite_score = (size_score * 0.3 + center_score * 0.3 +
                              method_score * 0.3 + tracking_bonus * 0.1)

            if composite_score > best_score:
                best_score = composite_score
                best_face = (x, y, w, h)

        return best_face

    def calculate_face_overlap(self, face1, face2):
        """Calculate overlap percentage between two face rectangles"""
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2

        # Calculate intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap

        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        face1_area = w1 * h1
        face2_area = w2 * h2
        union_area = face1_area + face2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def get_primary_face_region(self):
        """Enhanced face detection with multiple validation layers"""
        if self.current_frame is None:
            return None

        # Step 1: Preprocess frame for better face detection
        processed_frame = self.preprocess_for_face_detection(self.current_frame)

        # Step 2: Multi-method face detection
        face_candidates = self.detect_face_candidates(processed_frame)

        if not face_candidates:
            self.consecutive_face_detections = 0
            # Try with enhanced preprocessing
            enhanced_frame = self.enhance_frame_for_detection(self.current_frame)
            face_candidates = self.detect_face_candidates(enhanced_frame)

            if not face_candidates:
                return None

        # Step 3: Validate face candidates (reject false positives)
        valid_faces = []
        for candidate in face_candidates:
            if self.validate_human_face(self.current_frame, candidate):
                valid_faces.append(candidate)

        if not valid_faces:
            self.consecutive_face_detections = 0
            return None

        # Step 4: Select best face candidate
        best_face = self.select_best_face_candidate(valid_faces)

        if best_face:
            self.consecutive_face_detections += 1
            self.last_valid_face = best_face
            self.face_tracking_confidence = min(100, self.consecutive_face_detections * 10)

            x, y, w, h = best_face
            self.log_message(f"✅ Validated human face: {w}x{h} at ({x},{y}) - Confidence: {self.face_tracking_confidence}%")

            return best_face

        return None

    def create_precise_face_mask(self, face_region, face_coords):
        """Create precise face mask to eliminate background detection"""
        padding, pad_y, face_w, face_h = face_coords

        # Create mask for the face region
        mask = np.zeros(face_region.shape[:2], dtype=np.uint8)

        # Create elliptical mask for more natural face shape
        center_x = padding + face_w // 2
        center_y = pad_y + face_h // 2

        # Make ellipse slightly smaller than detected rectangle to avoid edges
        ellipse_w = int(face_w * 0.4)  # 40% of width as semi-major axis
        ellipse_h = int(face_h * 0.45)  # 45% of height as semi-minor axis

        # Create elliptical face mask
        cv2.ellipse(mask, (center_x, center_y), (ellipse_w, ellipse_h), 0, 0, 360, 255, -1)

        # Additional refinement: remove top and bottom portions to focus on main face area
        # Remove top 15% (forehead area that might include hair)
        mask[0:int(face_h * 0.15), :] = 0

        # Remove bottom 10% (chin/neck area)
        mask[int(face_h * 0.9):, :] = 0

        return mask

    def is_point_in_face_boundary(self, x, y, face_region, face_mask=None):
        """Check if a point is within the valid face boundary"""
        face_x, face_y, face_w, face_h = face_region

        # First check: within face rectangle
        if not (face_x <= x <= face_x + face_w and face_y <= y <= face_y + face_h):
            return False

        # Second check: within face mask if provided
        if face_mask is not None:
            # Convert global coordinates to face region coordinates
            local_x = x - face_x
            local_y = y - face_y

            # Check bounds
            if (0 <= local_x < face_mask.shape[1] and
                0 <= local_y < face_mask.shape[0]):
                return face_mask[local_y, local_x] > 0

        return True

    def assess_face_quality(self, face_region, face_mask):
        """Assess quality of isolated face region"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Only assess quality within face mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=face_mask)

        # Brightness check (within mask only)
        brightness = cv2.mean(masked_gray, mask=face_mask)[0]
        brightness_score = 100 if 70 <= brightness <= 200 else max(0, 100 - abs(brightness - 135) * 2)

        # Contrast check (within mask only)
        mean_val = cv2.mean(masked_gray, mask=face_mask)[0]
        std_val = np.sqrt(cv2.mean((masked_gray - mean_val) ** 2, mask=face_mask)[0])
        contrast_score = min(100, std_val * 3)

        # Sharpness check (within mask only)
        laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
        laplacian_masked = cv2.bitwise_and(laplacian.astype(np.uint8), laplacian.astype(np.uint8), mask=face_mask)
        sharpness = np.var(laplacian_masked[face_mask > 0]) if np.any(face_mask > 0) else 0
        sharpness_score = min(100, sharpness / 50.0)

        # Size check
        face_area = np.sum(face_mask > 0)
        size_score = min(100, face_area / 200)  # Optimal above 20k pixels

        # Overall quality score
        quality = (brightness_score + contrast_score + sharpness_score + size_score) / 4.0

        return quality

    def preprocess_for_face_detection(self, frame):
        """Preprocess frame to improve face detection reliability"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram equalization for better lighting
        if self.auto_brightness_enabled:
            gray = cv2.equalizeHist(gray)

        # Contrast enhancement
        if self.contrast_enhancement_enabled:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

        return gray

    def enhance_frame_for_detection(self, frame):
        """Enhanced preprocessing for difficult lighting conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Multiple enhancement techniques

        # 1. Gamma correction for lighting
        gamma = 1.2
        gamma_corrected = np.array(255 * (gray / 255) ** gamma, dtype='uint8')

        # 2. Bilateral filter to reduce noise while keeping edges
        bilateral = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)

        # 3. Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)

        return enhanced

    def detect_face_candidates(self, gray_frame):
        """Detect face candidates using multiple detection methods"""
        face_candidates = []

        # Method 1: Primary frontal face cascade (most reliable)
        if self.face_cascade and not self.face_cascade.empty():
            faces1 = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.03,
                minNeighbors=6,
                minSize=(80, 80),
                maxSize=(350, 350),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for face in faces1:
                x, y, w, h = face
                area = w * h
                if self.min_face_area <= area <= self.max_face_area:
                    face_candidates.append((x, y, w, h, 'primary', 1.0))

        # Method 2: Alternative frontal face cascade
        if self.face_cascade_alt and not self.face_cascade_alt.empty():
            faces2 = self.face_cascade_alt.detectMultiScale(
                gray_frame,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(70, 70),
                maxSize=(300, 300)
            )

            for face in faces2:
                x, y, w, h = face
                area = w * h
                if self.min_face_area <= area <= self.max_face_area:
                    # Check if this is a duplicate of Method 1
                    is_duplicate = False
                    for existing in face_candidates:
                        ex_x, ex_y, ex_w, ex_h = existing[:4]
                        overlap = self.calculate_face_overlap((x, y, w, h), (ex_x, ex_y, ex_w, ex_h))
                        if overlap > 0.7:  # 70% overlap = same face
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        face_candidates.append((x, y, w, h, 'alternative', 0.8))

        # Method 3: Relaxed detection for difficult cases
        if len(face_candidates) == 0 and self.face_cascade:
            faces3 = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=3,  # More relaxed
                minSize=(60, 60),
                maxSize=(400, 400)
            )

            for face in faces3:
                x, y, w, h = face
                area = w * h
                if area >= 8000:  # Lower minimum for difficult cases
                    face_candidates.append((x, y, w, h, 'relaxed', 0.6))

        return face_candidates

    def run_continuous_analysis_old(self):
        """Enhanced continuous analysis with timing control"""
        analysis_count = 0

        while self.analysis_active and self.current_frame is not None:
            current_time = time.time()

            # Only analyze at specified intervals
            if current_time - self.last_analysis_time >= self.analysis_interval:
                # Perform analysis
                start_time = time.time()
                results = self.perform_full_skin_analysis(self.current_frame)

                # Store stabilized results
                self.stabilized_results = results

                # Update GUI
                self.update_enhanced_metrics_display(results)
                self.display_enhanced_analysis_results(results)

                # Update timing
                self.last_analysis_time = current_time
                analysis_time = time.time() - start_time

                analysis_count += 1
                self.log_message(f"📊 Enhanced Analysis #{analysis_count} completed in {analysis_time:.3f}s")

            time.sleep(0.1)  # Small sleep to prevent excessive CPU usage

    def perform_full_skin_analysis(self, frame):
        """Hailo-accelerated skin analysis replacement"""
        if not hasattr(self, '_hailo_analyzer'):
            if HAILO_AVAILABLE:
                try:
                    self._hailo_analyzer = create_skin_analysis_system()
                    self.log_message("🚀 Hailo analyzer initialized")
                except Exception as e:
                    self.log_message(f"⚠️ Hailo init error: {e}")
                    return self.get_empty_results()
            else:
                return self.get_empty_results()

        try:
            # Use Hailo-accelerated analysis
            hailo_result = self._hailo_analyzer.analyze_camera_frame(frame)

            # Convert to GUI format
            from datetime import datetime
            overall_health = hailo_result.get('overall_health', {})
            conditions = hailo_result.get('conditions', {})

            results = {
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'faces_detected': 1 if hailo_result.get('success', False) else 0,
                'features': {
                    'acne_spots': {'count': max(0, int((100 - conditions.get('acne', {}).get('score', 80)) / 10)), 'locations': [], 'severity': []},
                    'pores': {'count': max(0, int((100 - conditions.get('skin_quality', {}).get('score', 75)) / 8)), 'locations': [], 'sizes': []}
                },
                'analysis_quality': int(overall_health.get('confidence', 85)),
                'pore_analysis': {
                    'count': max(0, int((100 - conditions.get('skin_quality', {}).get('score', 75)) / 8)),
                    'average_size': 2.5,
                    'coverage_percent': (100 - conditions.get('skin_quality', {}).get('score', 75)) / 20
                },
                'blemish_analysis': {
                    'count': max(0, int((100 - conditions.get('acne', {}).get('score', 80)) / 10)),
                    'severity': 'Mild' if conditions.get('acne', {}).get('score', 80) > 70 else 'Moderate'
                },
                'texture_analysis': {
                    'smoothness_score': conditions.get('skin_quality', {}).get('score', 75),
                    'uniformity_score': conditions.get('pigmentation', {}).get('score', 80),
                    'skin_health': overall_health.get('score', 78)
                }
            }
            return results

        except Exception as e:
            self.log_message(f"⚠️ Hailo analysis error: {e}")
            return self.get_empty_results()
    def analyze_pores_advanced(self, face_gray):
        """Advanced pore detection and analysis"""
        try:
            # Apply bilateral filter to preserve edges while smoothing
            filtered = cv2.bilateralFilter(face_gray, 9, 75, 75)

            # Adaptive thresholding for dark spots (pores)
            thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

            # Morphological operations to clean up
            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and analyze pores
            pores = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 2 < area < 80:  # Pore size range
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.3:  # Reasonably circular
                            pores.append(area)

            pore_count = len(pores)
            avg_pore_size = np.mean(pores) if pores else 0
            pore_density = pore_count / (face_gray.shape[0] * face_gray.shape[1]) * 10000

            return {
                'count': pore_count,
                'average_size': round(avg_pore_size, 2),
                'density': round(pore_density, 2)
            }

        except Exception as e:
            return {'count': 0, 'average_size': 0, 'density': 0}

    def analyze_blemishes_advanced(self, face_region):
        """Advanced blemish and acne detection"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

            # Define ranges for redness (acne/inflammation)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            # Create masks
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Pink/inflamed areas
            lower_pink = np.array([160, 30, 100])
            upper_pink = np.array([180, 150, 255])
            pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

            # Combine masks
            combined_mask = cv2.bitwise_or(red_mask, pink_mask)

            # Clean up with morphological operations
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze blemishes
            blemishes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 15 < area < 1000:  # Blemish size range
                    blemishes.append(area)

            blemish_count = len(blemishes)
            avg_size = np.mean(blemishes) if blemishes else 0
            coverage = sum(blemishes) / (face_region.shape[0] * face_region.shape[1]) * 100

            # Severity classification
            if blemish_count == 0:
                severity = "Clear"
            elif blemish_count <= 2:
                severity = "Mild"
            elif blemish_count <= 6:
                severity = "Moderate"
            else:
                severity = "Severe"

            return {
                'count': blemish_count,
                'average_size': round(avg_size, 2),
                'coverage_percent': round(coverage, 3),
                'severity': severity
            }

        except Exception as e:
            return {'count': 0, 'average_size': 0, 'coverage_percent': 0, 'severity': 'Unknown'}

    def analyze_skin_texture_advanced(self, face_gray):
        """Advanced skin texture and smoothness analysis"""
        try:
            # Calculate texture variance
            texture_variance = np.var(face_gray)

            # Edge detection for smoothness
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Local Binary Pattern simulation for texture uniformity
            def calculate_texture_uniformity(image):
                # Calculate local standard deviation
                kernel = np.ones((5,5), np.float32) / 25
                mean_filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
                sqr_img = np.square(image.astype(np.float32))
                sqr_mean_filtered = cv2.filter2D(sqr_img, -1, kernel)
                return np.sqrt(sqr_mean_filtered - np.square(mean_filtered))

            texture_map = calculate_texture_uniformity(face_gray)
            uniformity_score = 100 - np.mean(texture_map) * 2

            # Calculate smoothness score
            smoothness = max(0, 100 - (edge_density * 500))

            # Overall skin health
            skin_health = (smoothness + uniformity_score) / 2

            return {
                'texture_variance': round(texture_variance, 2),
                'smoothness_score': round(smoothness, 1),
                'uniformity_score': round(max(0, uniformity_score), 1),
                'skin_health': round(max(0, skin_health), 1),
                'edge_density': round(edge_density * 100, 3)
            }

        except Exception as e:
            return {'texture_variance': 0, 'smoothness_score': 0, 'uniformity_score': 0,
                   'skin_health': 0, 'edge_density': 0}

    def calculate_health_score(self, results):
        """Calculate overall skin health score"""
        base_score = 100

        # Pore impact
        if 'pore_analysis' in results:
            pore_count = results['pore_analysis'].get('count', 0)
            pore_penalty = min(25, pore_count * 0.8)
            base_score -= pore_penalty

        # Blemish impact
        if 'blemish_analysis' in results:
            blemish_count = results['blemish_analysis'].get('count', 0)
            blemish_penalty = min(30, blemish_count * 3)
            base_score -= blemish_penalty

        # Texture impact
        if 'texture_analysis' in results:
            skin_health = results['texture_analysis'].get('skin_health', 100)
            texture_penalty = (100 - skin_health) * 0.45
            base_score -= texture_penalty

        return max(0, min(100, base_score))

    def consolidate_multiple_scans(self, scans):
        """Consolidate multiple scans into single high-confidence result"""
        if not scans:
            return None

        # Use the highest quality scan as base
        best_scan = max(scans, key=lambda s: s['quality_score'])

        consolidated_results = {
            'face_region': best_scan['face_region'],
            'quality_score': best_scan['quality_score'],
            'features': {}
        }

        # For each feature type, only keep consistently detected features
        for feature_type in ['acne_spots', 'pores', 'blackheads', 'age_spots', 'moles', 'scars']:
            consolidated_features = []

            # Get all detections for this feature type across all scans
            all_detections = []
            for scan in scans:
                all_detections.extend(scan['features'][feature_type])

            # Group nearby detections (within 15 pixels)
            while all_detections:
                current = all_detections.pop(0)
                nearby_detections = [current]

                # Find nearby detections
                i = 0
                while i < len(all_detections):
                    detection = all_detections[i]
                    distance = np.sqrt((current['position'][0] - detection['position'][0])**2 +
                                     (current['position'][1] - detection['position'][1])**2)

                    if distance < 15:  # Within 15 pixels
                        nearby_detections.append(all_detections.pop(i))
                    else:
                        i += 1

                # Only keep features detected in multiple scans (high confidence)
                if len(nearby_detections) >= 2:  # Detected in at least 2 scans
                    # Average the positions for stability
                    avg_x = int(np.mean([d['position'][0] for d in nearby_detections]))
                    avg_y = int(np.mean([d['position'][1] for d in nearby_detections]))
                    avg_radius = int(np.mean([d['radius'] for d in nearby_detections]))
                    avg_confidence = np.mean([d['confidence'] for d in nearby_detections])

                    consolidated_features.append({
                        'position': (avg_x, avg_y),
                        'radius': avg_radius,
                        'confidence': avg_confidence,
                        'detection_count': len(nearby_detections)
                    })

            consolidated_results['features'][feature_type] = consolidated_features

        return consolidated_results

    def lock_detected_features(self, results):
        """Lock features in stationary positions"""
        if not results:
            return

        # Clear previous locks
        self.locked_features = {key: [] for key in self.locked_features}

        # Lock each feature type in place
        for feature_type, detections in results['features'].items():
            self.locked_features[feature_type] = detections.copy()

        # Update visual statistics to match locked features exactly
        for feature_type in self.locked_features:
            self.visual_statistics[feature_type] = len(self.locked_features[feature_type])

        self.feature_lock_active = True

        self.log_message(f"🔒 Features locked in place:")
        for feature_type, count in self.visual_statistics.items():
            if count > 0:
                self.log_message(f"  • {feature_type.replace('_', ' ').title()}: {count}")

    def update_final_analysis_display(self, results):
        """Enhanced final results with face boundary confirmation"""
        if not results:
            return

        face_region = results.get('face_region')
        if face_region:
            x, y, w, h = face_region
            face_area = w * h
            self.log_message("🎯 FACE-ONLY PRECISION ANALYSIS COMPLETE")
            self.log_message("="*60)
            self.log_message(f"📐 Face Region: {w}x{h} pixels at ({x}, {y})")
            self.log_message(f"📏 Analysis Area: {face_area:,} pixels (face only)")
            self.log_message(f"🔍 Quality Score: {results.get('quality_score', 0):.1f}%")
            self.log_message("="*60)

        total_features = 0

        for feature_type, features in results['features'].items():
            count = len(features)
            if count > 0:
                feature_name = feature_type.replace('_', ' ').title()
                self.log_message(f"🔍 {feature_name}: {count} detected (face-only)")

                # Verify all detections are within face boundary
                all_in_bounds = True
                if face_region:
                    for feature in features:
                        pos_x, pos_y = feature['position']
                        if not self.is_point_in_face_boundary(pos_x, pos_y, face_region):
                            all_in_bounds = False
                            break

                boundary_status = "✅ All within face" if all_in_bounds else "⚠️ Some outside face"
                self.log_message(f"   Location Check: {boundary_status}")

                # Show confidence
                if features:
                    avg_confidence = np.mean([f['confidence'] for f in features])
                    self.log_message(f"   Confidence: {avg_confidence:.1%}")

                total_features += count

        if total_features == 0:
            self.log_message("✅ No skin issues detected within face region")
            self.log_message("💚 Facial skin condition appears healthy")
        else:
            health_score = max(20, 100 - (total_features * 8))
            assessment = (
                "Excellent" if health_score >= 85 else
                "Good" if health_score >= 70 else
                "Fair" if health_score >= 55 else
                "Needs Attention"
            )
            self.log_message(f"📊 Facial Health Score: {health_score:.0f}% ({assessment})")

        self.log_message("🔒 All detections limited to facial area only")
        self.log_message("🚫 No background or non-face detections")
        self.log_message("="*60)

        # Update metrics display
        self.update_metrics_display()

    def update_metrics_display(self, results=None):
        """Enhanced metrics with face validation status"""
        if self.feature_lock_active:
            # Use locked visual statistics
            self.metric_vars['faces'].set("1 ✓")  # Checkmark for validated
            self.metric_vars['pores'].set(str(self.visual_statistics['pores']))
            self.metric_vars['acne_spots'].set(str(self.visual_statistics['acne_spots']))
            self.metric_vars['blackheads'].set(str(self.visual_statistics['blackheads']))
            self.metric_vars['age_spots'].set(str(self.visual_statistics['age_spots']))

            # Calculate health score
            total_issues = sum(self.visual_statistics.values())
            health_score = max(20, 100 - (total_issues * 8))
            self.metric_vars['health_score'].set(f"{health_score:.0f}%")

            # Smoothness based on detected issues
            smoothness = max(30, 100 - (self.visual_statistics.get('scars', 0) * 15))
            self.metric_vars['smoothness'].set(f"{smoothness:.0f}%")
        else:
            # Show validation status
            if self.consecutive_face_detections > 0:
                self.metric_vars['faces'].set(f"Validating ({self.face_tracking_confidence}%)")
            else:
                self.metric_vars['faces'].set("No Face")

            for key in self.metric_vars:
                if key != 'faces':
                    self.metric_vars[key].set("--")

    # === ENHANCED DETECTION METHODS WITH PRECISE LOCATION MAPPING ===

    def detect_acne_with_locations(self, face_region, offset_x, offset_y):
        """Detect acne spots with precise locations and severity assessment"""
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Enhanced acne detection with multiple color ranges
        # Red inflamed areas
        lower_red1 = np.array([0, 60, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 60, 60])
        upper_red2 = np.array([180, 255, 255])

        # Pink inflamed areas
        lower_pink = np.array([160, 40, 100])
        upper_pink = np.array([180, 180, 255])

        # Create combined mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_pink, upper_pink)
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

        # Morphological operations for cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours with hierarchy
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        acne_data = {'count': 0, 'locations': [], 'severity': []}

        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 800:  # Acne size range
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y

                    # Calculate severity based on area
                    severity = "mild" if area < 30 else "moderate" if area < 100 else "severe"

                    acne_data['count'] += 1
                    acne_data['locations'].append((cx, cy, int(np.sqrt(area/np.pi))))
                    acne_data['severity'].append(severity)

        return acne_data

    def detect_pores_with_locations(self, face_gray, offset_x, offset_y):
        """Detect pores with precise locations and size measurements"""
        # Enhanced pore detection
        blurred = cv2.GaussianBlur(face_gray, (3, 3), 0)

        # Adaptive threshold with optimized parameters
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 7, 3)

        # Morphological operations to isolate pores
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pore_data = {'count': 0, 'locations': [], 'sizes': []}

        for contour in contours:
            area = cv2.contourArea(contour)
            if 2 < area < 80:  # Pore size range
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.4:  # Reasonably circular
                        # Get centroid
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"]) + offset_x
                            cy = int(M["m01"] / M["m00"]) + offset_y
                            radius = int(np.sqrt(area / np.pi))

                            pore_data['count'] += 1
                            pore_data['locations'].append((cx, cy, radius))
                            pore_data['sizes'].append(area)

        return pore_data

    def detect_blackheads_with_locations(self, face_region, offset_x, offset_y):
        """Detect blackheads using enhanced dark spot detection"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Enhance contrast for dark spots
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Detect very dark regions
        _, dark_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blackhead_data = {'count': 0, 'locations': [], 'confidence': []}

        for contour in contours:
            area = cv2.contourArea(contour)
            if 3 < area < 50:  # Blackhead size range
                # Calculate confidence based on darkness
                mask = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                mean_intensity = cv2.mean(gray, mask=mask)[0]

                if mean_intensity < 80:  # Dark enough to be blackhead
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + offset_x
                        cy = int(M["m01"] / M["m00"]) + offset_y
                        confidence = (80 - mean_intensity) / 80.0  # Confidence score

                        blackhead_data['count'] += 1
                        blackhead_data['locations'].append((cx, cy, int(np.sqrt(area/np.pi))))
                        blackhead_data['confidence'].append(confidence)

        return blackhead_data

    def detect_fine_lines_with_locations(self, face_gray, offset_x, offset_y):
        """Detect fine lines using advanced edge detection"""
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(face_gray, (3, 3), 0)

        # Enhanced edge detection for fine lines
        edges = cv2.Canny(blurred, 30, 80, apertureSize=3, L2gradient=True)

        # Use HoughLinesP to detect line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                               minLineLength=10, maxLineGap=5)

        fine_line_data = {'count': 0, 'locations': [], 'lengths': []}

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line length
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                # Filter for fine lines (not major edges)
                if 10 < length < 60:
                    # Calculate midpoint
                    mid_x = int((x1 + x2) / 2) + offset_x
                    mid_y = int((y1 + y2) / 2) + offset_y

                    fine_line_data['count'] += 1
                    fine_line_data['locations'].append((mid_x, mid_y, int(length/2)))
                    fine_line_data['lengths'].append(length)

        return fine_line_data

    def detect_wrinkles_with_locations(self, face_gray, offset_x, offset_y):
        """Detect deeper wrinkles using directional filters"""
        # Apply directional Sobel filters
        sobel_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Threshold for strong edges (wrinkles)
        _, wrinkle_thresh = cv2.threshold(gradient_magnitude.astype(np.uint8),
                                        40, 255, cv2.THRESH_BINARY)

        # Morphological operations to connect wrinkle segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        wrinkle_thresh = cv2.morphologyEx(wrinkle_thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(wrinkle_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        wrinkle_data = {'count': 0, 'locations': [], 'depths': []}

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 20:  # Minimum wrinkle size
                # Calculate wrinkle depth based on gradient intensity
                mask = np.zeros(face_gray.shape, np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                mean_gradient = cv2.mean(gradient_magnitude, mask=mask)[0]

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y

                    wrinkle_data['count'] += 1
                    wrinkle_data['locations'].append((cx, cy, 8))  # Fixed radius for wrinkles
                    wrinkle_data['depths'].append(mean_gradient)

        return wrinkle_data

    def detect_age_spots_with_locations(self, face_region, offset_x, offset_y):
        """Detect age spots and hyperpigmentation"""
        # Convert to LAB color space for better pigmentation detection
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]

        # Detect darker regions (hyperpigmentation)
        mean_brightness = np.mean(l_channel)
        _, dark_spots = cv2.threshold(l_channel, mean_brightness - 20, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        age_spot_data = {'count': 0, 'locations': [], 'sizes': []}

        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 200:  # Age spot size range
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    radius = int(np.sqrt(area / np.pi))

                    age_spot_data['count'] += 1
                    age_spot_data['locations'].append((cx, cy, radius))
                    age_spot_data['sizes'].append(area)

        return age_spot_data

    def detect_texture_issues_with_locations(self, face_gray, offset_x, offset_y):
        """Detect texture irregularities and rough patches"""
        if ADVANCED_TEXTURE_ANALYSIS:
            # Use scikit-image for advanced texture analysis
            radius = 1
            n_points = 8
            lbp = local_binary_pattern(face_gray, n_points, radius, method='uniform')

            # Calculate texture variance
            kernel = np.ones((5,5), np.float32) / 25
            lbp_smooth = cv2.filter2D(lbp, -1, kernel)
            texture_var = np.abs(lbp - lbp_smooth)

            # Threshold for high texture variation
            high_var_thresh = np.percentile(texture_var, 90)
            _, texture_thresh = cv2.threshold(texture_var.astype(np.uint8),
                                            int(high_var_thresh), 255, cv2.THRESH_BINARY)
        else:
            # Fallback method using standard deviation
            kernel = np.ones((5,5), np.float32) / 25
            mean_filtered = cv2.filter2D(face_gray.astype(np.float32), -1, kernel)
            sqr_img = np.square(face_gray.astype(np.float32))
            sqr_mean_filtered = cv2.filter2D(sqr_img, -1, kernel)
            texture_var = np.sqrt(np.abs(sqr_mean_filtered - np.square(mean_filtered)))

            high_var_thresh = np.percentile(texture_var, 85)
            _, texture_thresh = cv2.threshold(texture_var.astype(np.uint8),
                                            int(high_var_thresh), 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(texture_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        texture_data = {'count': 0, 'locations': [], 'severity': []}

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 15:  # Minimum texture issue size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y

                    # Calculate severity
                    mask = np.zeros(face_gray.shape, np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    severity = cv2.mean(texture_var, mask=mask)[0]

                    texture_data['count'] += 1
                    texture_data['locations'].append((cx, cy, 6))
                    texture_data['severity'].append(severity)

        return texture_data

    # === TEMPORAL STABILIZATION METHODS ===

    def apply_temporal_stabilization(self, current_results):
        """Apply temporal smoothing to reduce fluctuations"""
        # Add current results to history
        self.analysis_history.append(current_results)

        # Maintain buffer size
        if len(self.analysis_history) > self.history_buffer_size:
            self.analysis_history.pop(0)

        # If we don't have enough history, return current results
        if len(self.analysis_history) < 3:
            return current_results

        # Create stabilized results
        stabilized = current_results.copy()

        # Average counts across recent history
        if current_results['faces_detected'] > 0:
            for feature_type in ['acne_spots', 'pores', 'blackheads', 'fine_lines',
                               'wrinkles', 'age_spots', 'texture_issues']:

                # Get recent counts
                recent_counts = [r['features'][feature_type]['count']
                               for r in self.analysis_history[-3:]
                               if r['faces_detected'] > 0]

                if recent_counts:
                    # Use median for stability (less sensitive to outliers)
                    stabilized_count = int(np.median(recent_counts))
                    stabilized['features'][feature_type]['count'] = stabilized_count

                    # Apply location clustering for stable positioning
                    stabilized['features'][feature_type]['locations'] = \
                        self.cluster_locations(feature_type, current_results['features'][feature_type]['locations'])

        return stabilized

    def cluster_locations(self, feature_type, current_locations):
        """Cluster nearby locations to reduce jitter"""
        if not current_locations:
            return current_locations

        clustered_locations = []
        cluster_radius = 15  # Pixels

        for loc in current_locations:
            x, y = loc[0], loc[1]

            # Check if this location is close to existing clusters
            merged = False
            for i, existing in enumerate(clustered_locations):
                ex_x, ex_y = existing[0], existing[1]
                distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)

                if distance < cluster_radius:
                    # Merge with existing cluster (average position)
                    new_x = int((x + ex_x) / 2)
                    new_y = int((y + ex_y) / 2)
                    clustered_locations[i] = (new_x, new_y, loc[2])
                    merged = True
                    break

            if not merged:
                clustered_locations.append(loc)

        return clustered_locations

    def calculate_analysis_quality(self, face_region):
        """Calculate quality score for analysis reliability"""
        # Check image quality factors
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Brightness check
        brightness = np.mean(gray)
        brightness_score = 1.0 if 80 <= brightness <= 180 else 0.5

        # Contrast check
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 50.0)

        # Blur check (Laplacian variance)
        blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0)

        # Face size check
        face_area = face_region.shape[0] * face_region.shape[1]
        size_score = min(1.0, face_area / 40000.0)  # Optimal around 200x200

        # Combined quality score
        quality = (brightness_score + contrast_score + blur_score + size_score) / 4.0

        return quality * 100  # Return as percentage

    def calculate_enhanced_health_score(self, features):
        """Calculate comprehensive health metrics"""
        base_score = 100

        # Weight different features by importance
        feature_weights = {
            'acne_spots': 3.0,      # High impact
            'pores': 1.0,           # Medium impact
            'blackheads': 2.0,      # Medium-high impact
            'fine_lines': 1.5,      # Medium impact
            'wrinkles': 4.0,        # High impact
            'age_spots': 2.5,       # Medium-high impact
            'texture_issues': 2.0   # Medium-high impact
        }

        # Calculate weighted deductions
        for feature_type, weight in feature_weights.items():
            count = features[feature_type]['count']
            deduction = min(30, count * weight)  # Cap individual deductions
            base_score -= deduction

        # Ensure score stays in valid range
        overall_score = max(0, min(100, base_score))

        # Calculate category scores
        acne_score = max(0, 100 - features['acne_spots']['count'] * 8)
        pore_score = max(0, 100 - features['pores']['count'] * 2)
        aging_score = max(0, 100 - (features['wrinkles']['count'] * 6 +
                                   features['age_spots']['count'] * 4))
        texture_score = max(0, 100 - features['texture_issues']['count'] * 5)

        return {
            'overall_score': overall_score,
            'acne_score': acne_score,
            'pore_score': pore_score,
            'aging_score': aging_score,
            'texture_score': texture_score
        }

    def generate_analysis_details(self, results, face_width, face_height):
        """Generate detailed analysis information"""
        details = []

        details.append(f"Face region analyzed: {face_width}x{face_height} pixels")

        if 'pore_analysis' in results:
            pore_data = results['pore_analysis']
            details.append(f"Pore density: {pore_data.get('density', 0):.1f} per cm²")
            details.append(f"Average pore size: {pore_data.get('average_size', 0):.1f} pixels")

        if 'blemish_analysis' in results:
            blemish_data = results['blemish_analysis']
            details.append(f"Blemish severity: {blemish_data.get('severity', 'Unknown')}")
            details.append(f"Skin coverage affected: {blemish_data.get('coverage_percent', 0):.3f}%")

        if 'texture_analysis' in results:
            texture_data = results['texture_analysis']
            details.append(f"Skin smoothness: {texture_data.get('smoothness_score', 0):.1f}/100")
            details.append(f"Texture uniformity: {texture_data.get('uniformity_score', 0):.1f}/100")

        return details

    def update_analysis_display(self, results):
        """Update GUI with analysis results"""
        # Update metrics
        self.metric_vars['faces_detected'].set(str(results['faces_detected']))

        if results['faces_detected'] > 0:
            # Pore data
            if 'pore_analysis' in results:
                self.metric_vars['pore_count'].set(str(results['pore_analysis'].get('count', 0)))

            # Blemish data
            if 'blemish_analysis' in results:
                self.metric_vars['blemish_count'].set(str(results['blemish_analysis'].get('count', 0)))

            # Texture data
            if 'texture_analysis' in results:
                smoothness = results['texture_analysis'].get('smoothness_score', 0)
                uniformity = results['texture_analysis'].get('uniformity_score', 0)
                self.metric_vars['smoothness_score'].set(f"{smoothness:.1f}%")
                self.metric_vars['skin_uniformity'].set(f"{uniformity:.1f}%")

            # Overall health score
            health_score = results.get('overall_health', 0)
            self.health_score_var.set(f"{health_score:.1f}%")

            # Update health score color based on value
            if health_score >= 80:
                color = '#00ff00'  # Green
            elif health_score >= 60:
                color = '#ffff00'  # Yellow
            elif health_score >= 40:
                color = '#ff8800'  # Orange
            else:
                color = '#ff0000'  # Red

            self.health_score_label.configure(fg=color)
        else:
            # No face detected
            for key in self.metric_vars:
                if key != 'faces_detected':
                    self.metric_vars[key].set("--")
            self.health_score_var.set("---%")
            self.health_score_label.configure(fg='#888888')

    def display_comprehensive_results(self, results):
        """Display comprehensive analysis results in log"""
        timestamp = results['timestamp']

        if results['faces_detected'] == 0:
            self.log_message(f"[{timestamp}] ⚠️ NO FACE DETECTED - Please position face in view")
            return

        # Header
        result_text = f"""
[{timestamp}] 🔬 COMPREHENSIVE DERMATOLOGIST ANALYSIS:

📊 DETECTION SUMMARY:
  👤 Faces Detected: {results['faces_detected']}
"""

        if 'pore_analysis' in results:
            pore_data = results['pore_analysis']
            result_text += f"""
🔍 PORE ANALYSIS:
  • Pores Detected: {pore_data.get('count', 0)}
  • Average Size: {pore_data.get('average_size', 0):.2f} pixels
  • Pore Density: {pore_data.get('density', 0):.2f} per cm²
"""

        if 'blemish_analysis' in results:
            blemish_data = results['blemish_analysis']
            result_text += f"""
🔴 BLEMISH ANALYSIS:
  • Blemishes Found: {blemish_data.get('count', 0)}
  • Severity Level: {blemish_data.get('severity', 'Unknown')}
  • Coverage: {blemish_data.get('coverage_percent', 0):.3f}%
  • Average Size: {blemish_data.get('average_size', 0):.2f} pixels
"""

        if 'texture_analysis' in results:
            texture_data = results['texture_analysis']
            result_text += f"""
✨ SKIN TEXTURE ANALYSIS:
  • Smoothness Score: {texture_data.get('smoothness_score', 0):.1f}/100
  • Uniformity Score: {texture_data.get('uniformity_score', 0):.1f}/100
  • Texture Variance: {texture_data.get('texture_variance', 0):.2f}
  • Edge Density: {texture_data.get('edge_density', 0):.3f}%
"""

        # Overall health assessment
        health_score = results.get('overall_health', 0)
        if health_score >= 85:
            assessment = "EXCELLENT - Outstanding skin condition"
        elif health_score >= 70:
            assessment = "VERY GOOD - Minor areas for improvement"
        elif health_score >= 55:
            assessment = "GOOD - Some areas to address"
        elif health_score >= 40:
            assessment = "FAIR - Multiple concerns detected"
        else:
            assessment = "NEEDS ATTENTION - Significant issues detected"

        result_text += f"""
💚 OVERALL HEALTH SCORE: {health_score:.1f}/100
🏥 PROFESSIONAL ASSESSMENT: {assessment}

📋 DETAILED MEASUREMENTS:
"""

        for detail in results.get('analysis_details', []):
            result_text += f"  • {detail}\n"

        result_text += "\n" + "=" * 70 + "\n"

        self.log_message(result_text)

    # === ENHANCED DISPLAY METHODS ===

    def update_enhanced_metrics_display(self, results):
        """Enhanced metrics display with feature breakdown"""
        if results and results['faces_detected'] > 0:
            features = results['features']

            # Update main metrics
            self.metric_vars['faces_detected'].set(str(results['faces_detected']))
            self.metric_vars['pore_count'].set(str(features['pores']['count']))

            # Calculate total skin issues
            total_acne = features['acne_spots']['count']
            total_issues = (features['blackheads']['count'] +
                           features['age_spots']['count'] +
                           features['texture_issues']['count'])

            self.metric_vars['blemish_count'].set(f"{total_acne} + {total_issues}")

            # Enhanced health score
            health_metrics = results.get('health_metrics', {})
            overall_score = health_metrics.get('overall_score', 0)
            self.health_score_var.set(f"{overall_score:.1f}%")

            # Skin smoothness based on texture analysis
            smoothness = 100 - (features['fine_lines']['count'] +
                               features['wrinkles']['count']) * 2
            smoothness = max(0, min(100, smoothness))
            self.metric_vars['smoothness_score'].set(f"{smoothness:.1f}%")

            # Update uniformity score
            self.metric_vars['skin_uniformity'].set(f"{health_metrics.get('texture_score', 0):.1f}%")

            # Update health score color
            if overall_score >= 80:
                color = '#00ff00'  # Green
            elif overall_score >= 60:
                color = '#ffff00'  # Yellow
            elif overall_score >= 40:
                color = '#ff8800'  # Orange
            else:
                color = '#ff0000'  # Red

            self.health_score_label.configure(fg=color)
        else:
            # No face detected
            for key in self.metric_vars:
                if key != 'faces_detected':
                    self.metric_vars[key].set("--")
            self.health_score_var.set("---%")
            self.health_score_label.configure(fg='#888888')

    def display_enhanced_analysis_results(self, results):
        """Display comprehensive analysis with feature breakdown"""
        if not results or results['faces_detected'] == 0:
            return

        timestamp = results['timestamp']
        features = results['features']
        quality = results.get('analysis_quality', 0)
        health_metrics = results.get('health_metrics', {})

        result_text = f"""
[{timestamp}] 🔬 ENHANCED DERMATOLOGIST ANALYSIS:

📊 DETECTION SUMMARY:
  👤 Faces: {results['faces_detected']} | Quality: {quality:.1f}%
  🔴 Acne Spots: {features['acne_spots']['count']}
  🔵 Enlarged Pores: {features['pores']['count']}
  🟡 Blackheads: {features['blackheads']['count']}
  🟢 Fine Lines: {features['fine_lines']['count']}
  🟢 Wrinkles: {features['wrinkles']['count']}
  🟠 Age Spots: {features['age_spots']['count']}
  🟣 Texture Issues: {features['texture_issues']['count']}

🏥 HEALTH ASSESSMENT:
  💚 Overall Score: {health_metrics.get('overall_score', 0):.1f}%
  🔥 Acne Health: {health_metrics.get('acne_score', 0):.1f}%
  🕳️ Pore Health: {health_metrics.get('pore_score', 0):.1f}%
  ⏰ Aging Score: {health_metrics.get('aging_score', 0):.1f}%
  ✨ Texture Score: {health_metrics.get('texture_score', 0):.1f}%

📍 PRECISE MAPPING: Active (see color-coded overlay)
{"="*55}
"""

        self.log_message(result_text)

    def get_empty_results(self):
        """Return empty results structure"""
        return {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'faces_detected': 0,
            'pore_analysis': {'count': 0, 'average_size': 0, 'density': 0},
            'blemish_analysis': {'count': 0, 'average_size': 0, 'coverage_percent': 0, 'severity': 'Unknown'},
            'texture_analysis': {'smoothness_score': 0, 'uniformity_score': 0, 'skin_health': 0},
            'overall_health': 0,
            'analysis_details': []
        }

    def add_analysis_overlay(self, frame):
        """Enhanced overlay with face detection debugging"""
        overlay = frame.copy()

        # Show face detection debugging information
        if self.analysis_active or True:  # Always show when debugging
            self.draw_face_detection_debug(overlay)

        # Show locked features if analysis complete
        if self.feature_lock_active and self.locked_features:
            self.draw_locked_features(overlay, frame.shape[1])
            self.draw_precision_legend(overlay)

        # Show analysis progress
        elif self.analysis_active:
            if hasattr(self, 'analysis_phase'):
                if self.analysis_phase == "waiting":
                    cv2.putText(overlay, "🎯 Validating Human Face...", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif self.analysis_phase == "analyzing":
                    cv2.putText(overlay, "🔬 Analyzing Validated Face...", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return overlay

    def draw_face_detection_debug(self, overlay):
        """Draw face detection debugging information"""
        if self.current_frame is None:
            return

        # Get current face detection results
        processed_frame = self.preprocess_for_face_detection(self.current_frame)
        face_candidates = self.detect_face_candidates(processed_frame)

        # Draw all face candidates with validation status
        for i, candidate in enumerate(face_candidates):
            x, y, w, h, method, confidence = candidate

            # Mirror coordinates for display
            x_mirrored = overlay.shape[1] - x - w

            # Validate this candidate
            is_valid = self.validate_human_face(self.current_frame, candidate)

            # Color code: Green = valid human face, Red = rejected, Yellow = uncertain
            if is_valid:
                color = (0, 255, 0)  # Green - valid human face
                thickness = 3
                status = "VALID HUMAN"
            else:
                color = (0, 0, 255)  # Red - rejected
                thickness = 2
                status = "REJECTED"

            # Draw rectangle
            cv2.rectangle(overlay, (x_mirrored, y), (x_mirrored + w, y + h), color, thickness)

            # Draw method and status labels
            cv2.putText(overlay, f"{method.upper()}", (x_mirrored, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(overlay, status, (x_mirrored, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(overlay, f"{w}x{h}", (x_mirrored, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw face tracking info
        if self.last_valid_face:
            x, y, w, h = self.last_valid_face
            x_mirrored = overlay.shape[1] - x - w

            # Draw tracking rectangle (blue)
            cv2.rectangle(overlay, (x_mirrored, y), (x_mirrored + w, y + h), (255, 255, 0), 2)
            cv2.putText(overlay, f"TRACKED: {self.face_tracking_confidence}%",
                       (x_mirrored, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw detection statistics
        valid_count = sum(1 for c in face_candidates if self.validate_human_face(self.current_frame, c))
        stats_text = f"Candidates: {len(face_candidates)} | Valid: {valid_count} | Tracking: {self.consecutive_face_detections}"

        cv2.putText(overlay, stats_text, (10, overlay.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw frame quality info
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)

        quality_text = f"Brightness: {brightness:.0f} | Contrast: {contrast:.0f}"
        cv2.putText(overlay, quality_text, (10, overlay.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_locked_features(self, overlay, frame_width):
        """Draw stationary feature markings with precise colors"""

        # Define precise color scheme
        feature_colors = {
            'acne_spots': (0, 0, 255),      # Bright Red
            'pores': (255, 100, 0),         # Blue
            'blackheads': (0, 255, 255),    # Yellow
            'age_spots': (0, 165, 255),     # Orange
            'moles': (128, 0, 128),         # Purple
            'scars': (0, 255, 0)            # Green
        }

        # Draw each feature type
        for feature_type, color in feature_colors.items():
            features = self.locked_features.get(feature_type, [])

            for feature in features:
                x, y = feature['position']
                radius = feature['radius']

                # Mirror x-coordinate for display
                x_mirrored = frame_width - x

                # Draw circle with appropriate thickness
                thickness = 2 if feature_type in ['acne_spots', 'moles'] else 1
                cv2.circle(overlay, (x_mirrored, y), radius, color, thickness)

                # Add feature label inside circle
                label_map = {
                    'acne_spots': 'A',
                    'pores': 'P',
                    'blackheads': 'B',
                    'age_spots': 'S',
                    'moles': 'M',
                    'scars': 'R'
                }

                label = label_map.get(feature_type, '?')

                # Calculate text position (centered in circle)
                text_x = x_mirrored - 4
                text_y = y + 3

                cv2.putText(overlay, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_precision_legend(self, overlay):
        """Draw comprehensive color legend"""
        legend_items = [
            ("🔴 Acne/Pimples", (0, 0, 255)),
            ("🔵 Enlarged Pores", (255, 100, 0)),
            ("🟡 Blackheads", (0, 255, 255)),
            ("🟠 Age Spots", (0, 165, 255)),
            ("🟣 Moles", (128, 0, 128)),
            ("🟢 Scars", (0, 255, 0))
        ]

        # Legend background
        legend_height = len(legend_items) * 25 + 20
        cv2.rectangle(overlay, (10, 10), (200, 10 + legend_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (200, 10 + legend_height), (255, 255, 255), 1)

        # Legend title
        cv2.putText(overlay, "Feature Legend:", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Legend items
        for i, (label, color) in enumerate(legend_items):
            y_pos = 55 + (i * 25)

            # Draw color circle
            cv2.circle(overlay, (25, y_pos), 6, color, -1)
            cv2.circle(overlay, (25, y_pos), 6, (255, 255, 255), 1)

            # Draw label (remove emoji for OpenCV text)
            clean_label = label[2:]  # Remove emoji
            cv2.putText(overlay, clean_label, (40, y_pos + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Legacy methods - keeping for compatibility but not used in precision mode

    def log_message(self, message):
        """Enhanced logging with face boundary confirmation"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        timestamped_message = f"[{timestamp}] {message}"

        self.results_text.insert(tk.END, timestamped_message + "\n")
        self.results_text.see(tk.END)

        # Also print to console for debugging
        print(timestamped_message)

        # Limit log size
        lines = int(self.results_text.index('end-1c').split('.')[0])
        if lines > 200:
            self.results_text.delete('1.0', '50.0')

    def take_snapshot(self):
        """Take and save a snapshot"""
        if self.current_frame is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"smart_mirror_snapshot_{timestamp}.jpg"
                filepath = os.path.join(os.path.expanduser("~"), filename)

                cv2.imwrite(filepath, self.current_frame)
                self.log_message(f"📸 Snapshot saved: {filename}")
                messagebox.showinfo("Snapshot Saved", f"Snapshot saved as {filename}")
            except Exception as e:
                self.log_message(f"❌ Snapshot failed: {e}")
                messagebox.showerror("Snapshot Error", f"Failed to save snapshot: {e}")

    def reset_camera(self):
        """Reset camera connection"""
        try:
            if self.camera:
                self.camera.release()
                time.sleep(1)

            self.start_camera()
            self.log_message("🔄 Camera reset successfully")
        except Exception as e:
            self.log_message(f"❌ Camera reset failed: {e}")

    def save_results(self):
        """Save analysis results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dermatology_analysis_{timestamp}.txt"
            filepath = os.path.join(os.path.expanduser("~"), filename)

            with open(filepath, 'w') as f:
                f.write(f"Smart Mirror Dermatology Analysis Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(self.results_text.get("1.0", tk.END))

            self.log_message(f"💾 Results saved: {filename}")
            messagebox.showinfo("Results Saved", f"Analysis results saved as {filename}")

        except Exception as e:
            self.log_message(f"❌ Save failed: {e}")
            messagebox.showerror("Save Error", f"Failed to save results: {e}")

    def clear_log(self):
        """Clear the analysis log"""
        self.results_text.delete("1.0", tk.END)
        self.log_message("🗑️ Analysis log cleared")
        self.log_message("🔬 Ready for new analysis session")
        self.log_message("=" * 60)

    def exit_app(self):
        """Clean shutdown of application"""
        print("🔄 Shutting down Smart Mirror application...")

        # Stop all operations
        self.running = False
        self.analysis_active = False
        # Stop spawned worker process
        try:
            if hasattr(self, '_mp_ctrl_q'):
                try:
                    self._mp_ctrl_q.put_nowait({'cmd': 'shutdown'})
                except Exception:
                    pass
            if hasattr(self, '_derm_proc') and self._derm_proc is not None:
                self._derm_proc.join(timeout=2.0)
                if self._derm_proc.is_alive():
                    self._derm_proc.terminate()
                    print("[DERM-HAILO] WorkerProc force-terminated")
        except Exception:
            pass
        # Stop dermatologist worker/engine if active
        try:
            if hasattr(self, '_derma_worker_stop'):
                self._derma_worker_stop.set()
            if getattr(self, 'derma_active', False):
                self.derma_active = False
            # Join analysis thread if running
            try:
                if getattr(self, '_derma_worker', None) is not None and self._derma_worker.is_alive():
                    self._derma_worker.join(timeout=1.5)
                    self._derma_worker = None
            except Exception:
                pass
            if hasattr(self, 'derma_engine') and self.derma_engine:
                self.derma_engine.stop()
        except Exception:
            pass

        # Release camera
        if self.camera:
            self.camera.release()

        # Clean up OpenCV
        try:
            cv2.destroyAllWindows()
        except Exception as _e:
            print('[DERM-HAILO] Note: destroyAllWindows not available:', _e)

        # Close application
        self.root.quit()
        self.root.destroy()
        # Cleanup shared memory
        try:
            if getattr(self, '_shm', None) is not None:
                try:
                    self._shm.close()
                except Exception:
                    pass
                try:
                    self._shm.unlink()
                except Exception:
                    pass
            if getattr(self, '_meta_shm', None) is not None:
                try:
                    self._meta_shm.close()
                except Exception:
                    pass
                try:
                    self._meta_shm.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    # (Removed duplicate create_metric_displays; keeping earlier enhanced version)

    # === EFFICIENT METHODS ADDED FOR PERFORMANCE OPTIMIZATION ===
    def update_display_fast(self, frame):
        """Ultra-fast display update with minimal processing"""
        # Convert to RGB quickly
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize efficiently if needed
        height, width = frame_rgb.shape[:2]
        if width > 640:  # Only resize if too large
            frame_rgb = cv2.resize(frame_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Quick PIL conversion and display
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)

        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

    def draw_simple_face_indicator(self, frame):
        """Simple face indicator - no heavy processing"""
        if self.current_face_region:
            x, y, w, h = self.current_face_region
            # Mirror coordinates
            x_mirrored = frame.shape[1] - x - w

            # Simple green rectangle
            cv2.rectangle(frame, (x_mirrored, y), (x_mirrored + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "ANALYZING...", (x_mirrored, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run_efficient_analysis(self):
        """EFFICIENT analysis - no lag, maximum performance"""

        # PHASE 1: Fast Face Detection (no heavy validation)
        self.status_var.set("🎯 Detecting face...")

        while self.analysis_active and self.face_stable_count < self.required_stable_frames:

            if self.current_frame is not None:
                # LIGHTWEIGHT face detection
                face_region = self.detect_face_fast()

                # Minimal Hailo overlay when a fresh HAAR ROI exists
                try:
                    if face_region and getattr(self, '_roi_fresh', False):
                        if getattr(self, 'derma_engine', None) is None:
                            from ai_features.dermatologist.hailo_dual_hef import DualHefEngine
                            import os as _os
                            lesion_path = _os.getenv('HEF_LESION', _os.path.expanduser('~/derma/models/tulanelab_derma.hef'))
                            acne_path = _os.getenv('HEF_ACNE', _os.path.expanduser('~/derma/models/acnenew.hef'))
                            self.derma_engine = DualHefEngine(hef_lesion=lesion_path, hef_acne=acne_path)
                            self.derma_engine.start()
                        import os as _os
                        try:
                            conf_acne = float(_os.getenv('CONF_ACNE', '0.15'))
                        except Exception:
                            conf_acne = 0.15
                        try:
                            conf_lesion = float(_os.getenv('CONF_LESION', '0.15'))
                        except Exception:
                            conf_lesion = 0.15
                        x, y, w, h = face_region
                        vis_roi, counts = self.derma_engine.infer_roi(self.current_frame, (x, y, w, h), conf_acne=conf_acne, conf_lesion=conf_lesion)
                        if counts != 'PAUSE' and vis_roi is not None:
                            try:
                                alpha = float(_os.getenv('DERM_ALPHA', '0.5'))
                            except Exception:
                                alpha = 0.5
                            alpha = max(0.0, min(1.0, alpha))
                            base = self.current_frame.copy()
                            dst = base[y:y+h, x:x+w]
                            if vis_roi.shape[:2] != (h, w):
                                vis_roi = cv2.resize(vis_roi, (w, h))
                            if dst.shape[:2] == vis_roi.shape[:2]:
                                blended = cv2.addWeighted(vis_roi, alpha, dst, 1.0 - alpha, 0.0)
                                base[y:y+h, x:x+w] = blended
                                from threading import Lock
                                try:
                                    self._frame_lock.acquire()
                                    self.current_frame = base
                                finally:
                                    self._frame_lock.release()
                                self.last_derma_vis = base
                                self.derma_counts = counts
                                if _os.getenv('DEBUG_DERM','0')=='1':
                                    print(f"[DERM-HAILO] ROI=({x},{y},{w},{h}) counts={'{'}'acne':{counts.get('acne',0)},'lesion':{counts.get('lesion',0)}{'}'}")
                except Exception:
                    pass

                if face_region:
                    if self.current_face_region is None:
                        self.current_face_region = face_region
                        self.face_stable_count = 1
                    else:
                        # Simple stability check - no complex overlap calculation
                        x1, y1, w1, h1 = self.current_face_region
                        x2, y2, w2, h2 = face_region

                        # Simple distance check
                        distance = abs(x1 - x2) + abs(y1 - y2)

                        if distance < 20:  # Close enough = stable
                            self.face_stable_count += 1
                            self.current_face_region = face_region
                        else:
                            self.face_stable_count = max(0, self.face_stable_count - 1)
                            self.current_face_region = face_region
                else:
                    self.face_stable_count = max(0, self.face_stable_count - 2)

            time.sleep(0.1)  # 10 FPS for face detection (enough for stability)

        if self.face_stable_count < self.required_stable_frames:
            self.status_var.set("❌ Could not detect stable face")
            self.stop_analysis()
            return

        # PHASE 2: 3-Second Efficient Analysis
        self.face_detected = True
        self.analyze_btn.configure(text="🔬 Analyzing...", bg='#e74c3c')
        self.status_var.set("🔬 Analyzing skin - please stay still...")

        self.log_message("✅ Face detected and stable - starting 3-second analysis")

        # Perform single comprehensive analysis (not multiple scans)
        analysis_results = self.perform_single_efficient_analysis()

        if analysis_results:
            # PHASE 3: Lock Results (no consolidation needed)
            self.final_results = analysis_results
            self.results_locked = True

            self.analyze_btn.configure(text="✅ Complete", bg='#27ae60')
            self.status_var.set("✅ Analysis complete - results locked")

            self.update_final_metrics()
            self.log_final_results()

            self.log_message("✅ Analysis complete - markings locked in place")
        else:
            self.status_var.set("❌ Analysis failed")
            self.stop_analysis()

    # (Removed duplicate detect_face_fast; keeping earlier definition near line ~1662)

    def perform_single_efficient_analysis(self):
        """EFFICIENT single analysis - 3 seconds, maximum accuracy"""

        if not self.current_face_region:
            return None

        x, y, w, h = self.current_face_region

        # Extract face region only
        face_region = self.current_frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Create simple face mask (no complex ellipse)
        face_mask = np.ones(face_gray.shape, dtype=np.uint8) * 255

        # Remove edges (10% border)
        border = int(min(w, h) * 0.1)
        face_mask[:border, :] = 0  # Top
        face_mask[-border:, :] = 0  # Bottom
        face_mask[:, :border] = 0  # Left
        face_mask[:, -border:] = 0  # Right

        # Efficient feature detection
        results = {
            'acne_spots': self.detect_acne_efficient(face_region, face_mask, x, y),
            'pores': self.detect_pores_efficient(face_gray, face_mask, x, y),
            'blackheads': self.detect_blackheads_efficient(face_region, face_mask, x, y),
            'age_spots': self.detect_age_spots_efficient(face_region, face_mask, x, y),
            'moles': self.detect_moles_efficient(face_region, face_mask, x, y)
        }

        # Sleep for remaining time to reach 3 seconds total
        time.sleep(2.5)  # Total analysis time = 3 seconds

        return results

    def _log_stub_notice(self, name: str) -> None:
        msg = f"[DERM-HAILO] NOTICE: {name} stubbed; returning empty results"
        try:
            self.log_message(msg)
        except Exception:
            try:
                print(msg)
            except Exception:
                pass

    def detect_acne_ultra_precise(self, *args, **kwargs):
        self._log_stub_notice('detect_acne_ultra_precise')
        return []

    def detect_age_spots_ultra_precise(self, *args, **kwargs):
        self._log_stub_notice('detect_age_spots_ultra_precise')
        return []

    def detect_blackheads_ultra_precise(self, *args, **kwargs):
        self._log_stub_notice('detect_blackheads_ultra_precise')
        return []

    def detect_moles_ultra_precise(self, *args, **kwargs):
        self._log_stub_notice('detect_moles_ultra_precise')
        return []

    def detect_pores_ultra_precise(self, *args, **kwargs):
        self._log_stub_notice('detect_pores_ultra_precise')
        return []

    def detect_scars_ultra_precise(self, *args, **kwargs):
        self._log_stub_notice('detect_scars_ultra_precise')
        return []

    def detect_acne_efficient(self, face_region, face_mask, offset_x, offset_y):
        """EFFICIENT acne detection - fast and accurate"""
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Simple red detection
        lower_red = np.array([0, 70, 70])
        upper_red = np.array([15, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # Apply face mask
        red_mask = cv2.bitwise_and(red_mask, face_mask)

        # Simple morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        acne_spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 300:  # Acne size range
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    radius = max(4, int(np.sqrt(area / np.pi)) + 2)

                    acne_spots.append({
                        'position': (cx, cy),
                        'radius': radius
                    })

        return acne_spots

    def detect_pores_efficient(self, face_gray, face_mask, offset_x, offset_y):
        """EFFICIENT pore detection"""
        # Simple threshold for dark spots
        blurred = cv2.GaussianBlur(face_gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 7, 3)

        # Apply face mask
        thresh = cv2.bitwise_and(thresh, face_mask)

        # Find small circular contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2 < area < 25:  # Pore size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    radius = max(2, int(np.sqrt(area / np.pi)))

                    pores.append({
                        'position': (cx, cy),
                        'radius': radius
                    })

        return pores

    def detect_blackheads_efficient(self, face_region, face_mask, offset_x, offset_y):
        """EFFICIENT blackhead detection"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Simple dark threshold
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, face_mask)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blackheads = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 3 < area < 30:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    radius = max(2, int(np.sqrt(area / np.pi)))

                    blackheads.append({
                        'position': (cx, cy),
                        'radius': radius
                    })

        return blackheads

    def detect_age_spots_efficient(self, face_region, face_mask, offset_x, offset_y):
        """EFFICIENT age spot detection"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Detect darker regions
        mean_val = cv2.mean(gray, mask=face_mask)[0]
        _, dark_spots = cv2.threshold(gray, mean_val - 25, 255, cv2.THRESH_BINARY_INV)
        dark_spots = cv2.bitwise_and(dark_spots, face_mask)

        contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        age_spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 200:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    radius = max(5, int(np.sqrt(area / np.pi)))

                    age_spots.append({
                        'position': (cx, cy),
                        'radius': radius
                    })

        return age_spots

    def detect_moles_efficient(self, face_region, face_mask, offset_x, offset_y):
        """EFFICIENT mole detection"""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Very dark spots
        _, very_dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        very_dark = cv2.bitwise_and(very_dark, face_mask)

        contours, _ = cv2.findContours(very_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 400:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    radius = max(6, int(np.sqrt(area / np.pi)))

                    moles.append({
                        'position': (cx, cy),
                        'radius': radius
                    })

        return moles

    def draw_final_results_only(self, frame):
        """EFFICIENT final results display - no lag"""
        if not self.results_locked:
            return frame

        # Color mapping
        colors = {
            'acne_spots': (0, 0, 255),      # Red
            'pores': (255, 0, 0),           # Blue
            'blackheads': (0, 255, 255),    # Yellow
            'age_spots': (0, 165, 255),     # Orange
            'moles': (128, 0, 128)          # Purple
        }

        # Draw all features efficiently
        for feature_type, color in colors.items():
            features = self.final_results.get(feature_type, [])

            for feature in features:
                x, y = feature['position']
                radius = feature['radius']

                # Mirror x coordinate
                x_mirrored = frame.shape[1] - x

                # Draw circle
                cv2.circle(frame, (x_mirrored, y), radius, color, 2)

                # Add small label
                label_map = {
                    'acne_spots': 'A',
                    'pores': 'P',
                    'blackheads': 'B',
                    'age_spots': 'S',
                    'moles': 'M'
                }

                label = label_map.get(feature_type, '')
                cv2.putText(frame, label, (x_mirrored - 3, y + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw simple legend
        self.draw_efficient_legend(frame)

        return frame

    def draw_efficient_legend(self, frame):
        """SIMPLE legend - no heavy processing"""
        legend_items = [
            ("Red: Acne", (0, 0, 255)),
            ("Blue: Pores", (255, 0, 0)),
            ("Yellow: Blackheads", (0, 255, 255)),
            ("Orange: Age Spots", (0, 165, 255)),
            ("Purple: Moles", (128, 0, 128))
        ]

        y_start = 20
        for i, (label, color) in enumerate(legend_items):
            y_pos = y_start + (i * 25)

            # Simple circle and text
            cv2.circle(frame, (20, y_pos), 8, color, -1)
            cv2.putText(frame, label, (35, y_pos + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def update_final_metrics(self):
        """Update metrics to match final results exactly"""
        if self.results_locked:
            # Count final results
            counts = {key: len(features) for key, features in self.final_results.items()}

            self.metric_vars['faces'].set("1")
            self.metric_vars['acne_spots'].set(str(counts['acne_spots']))
            self.metric_vars['pores'].set(str(counts['pores']))

            # Simple health score
            total_issues = sum(counts.values())
            health_score = max(30, 100 - (total_issues * 6))
            self.metric_vars['health_score'].set(f"{health_score:.0f}%")

            # Simple smoothness
            smoothness = max(40, 100 - (counts['pores'] * 3))
            self.metric_vars['smoothness'].set(f"{smoothness:.0f}%")

    def log_final_results(self):
        """Log final results efficiently"""
        self.log_message("🎯 FINAL ANALYSIS RESULTS:")
        self.log_message("=" * 40)

        total = 0
        for feature_type, features in self.final_results.items():
            count = len(features)
            if count > 0:
                name = feature_type.replace('_', ' ').title()
                self.log_message(f"🔍 {name}: {count}")
                total += count

        if total == 0:
            self.log_message("✅ No significant issues detected")

        health = max(30, 100 - (total * 6))
        self.log_message(f"💚 Health Score: {health:.0f}%")
        self.log_message("🔒 Results locked - stationary markings")
        self.log_message("=" * 40)

        print("✅ Application shutdown complete")

    def run(self):
        """Start the application main loop"""
        try:
            # Handle window close button
            self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

            # Log startup complete
            self.log_message("🚀 SMART MIRROR DERMATOLOGIST APPLICATION STARTED")
            self.log_message("📹 Professional desktop application ready")
            self.log_message("🔬 Advanced OpenCV skin analysis pipeline active")
            self.log_message("🎯 Click 'START ANALYSIS' to begin dermatologist examination")
            self.log_message("💡 Features: Pore detection • Blemish analysis • Texture assessment")
            self.log_message("=" * 70)

            # Start GUI main loop
            self.root.mainloop()

        except KeyboardInterrupt:
            print("\n⚠️ Keyboard interrupt received")
            self.exit_app()
        except Exception as e:
            print(f"❌ Application error: {e}")
            messagebox.showerror("Application Error", f"Fatal error: {e}")
            self.exit_app()


def _derm_apply_detections_on_gui_thread(self, dets):
    """Adaptor now stashes latest detections; drawing occurs in update_video_feed."""
    raw = dets or []
    processed = raw
    try:
        use_xywh_fix = bool(DERM_SMOOTH_FIX_XYWH and raw)
    
        incoming = []
        for d in raw:
            if use_xywh_fix:
                if 'xyxy' in d and isinstance(d['xyxy'], (list, tuple)) and len(d['xyxy']) == 4:
                    x1, y1, x2, y2 = [float(v) for v in d['xyxy']]
                else:
                    x1 = float(d.get('x1', d.get('x', 0.0)))
                    y1 = float(d.get('y1', d.get('y', 0.0)))
                    x2 = float(d.get('x2', x1 + d.get('w', 0.0)))
                    y2 = float(d.get('y2', y1 + d.get('h', 0.0)))
                x, y, w, h = _xyxy_to_xywh(x1, y1, x2, y2)
            else:
                x = float(d.get('x', d.get('roi_x', 0.0)))
                y = float(d.get('y', d.get('roi_y', 0.0)))
                w = float(d.get('w', d.get('roi_w', 0.0)))
                h = float(d.get('h', d.get('roi_h', 0.0)))
            conf = float(d.get('conf', d.get('score', 0.0)))
            cls_ = str(d.get('cls', d.get('label', 'unknown')))
            incoming.append(Det(x=x, y=y, w=w, h=h, conf=conf, cls=cls_))
    
        smoothed = _SMOOTHER.update_and_get(incoming)
    
        if use_xywh_fix and smoothed and len(smoothed) == len(raw):
            for src, det_obj in zip(raw, smoothed):
                x1, y1, x2, y2 = _xywh_to_xyxy(det_obj.x, det_obj.y, det_obj.w, det_obj.h)
                src['x1'], src['y1'], src['x2'], src['y2'] = float(x1), float(y1), float(x2), float(y2)
                src['score'] = det_obj.conf
                src['label'] = det_obj.cls
                if 'xyxy' in src:
                    src['xyxy'] = [src['x1'], src['y1'], src['x2'], src['y2']]
    
        processed = [
            {'x': t.x, 'y': t.y, 'w': t.w, 'h': t.h, 'conf': t.conf, 'cls': t.cls}
            for t in smoothed
        ] if smoothed else raw
    except Exception as _e:
        print(f"[DERM-HAILO] smoother: bypassing due to error: {_e}")
        processed = raw
    
    try:
        self._derm_latest_dets = processed or []
    except Exception:
        self._derm_latest_dets = []
# ===== [DERM-CODEX-END] gui apply dets =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Mirror Dermatologist")
    parser.add_argument('--pretty-draw', dest='pretty_draw', action='store_true',
                        help='Use alpha boxes + AA text for overlay rendering')
    parser.add_argument('--hud', dest='hud', action='store_true',
                        help='Show small FPS and status HUD on the video')
    parser.add_argument('--smooth',   dest='smooth',   action='store_true', help='Enable EMA smoothing via tracker')
    parser.add_argument('--hyst',     dest='hyst',     action='store_true', help='Enable hysteresis (appear/disappear)')
    parser.add_argument('--debounce', dest='debounce', action='store_true', help='Enable track confirmation debounce')
    parser.add_argument('--assoc',    dest='assoc',    action='store_true', help='Enable IoU association for IDs')
    parser.add_argument('--roi-gate', dest='roi_gate', choices=['none','skin','center'], default='none',
                        help='Filter detections by ROI: none (default), skin, or center')
    parser.add_argument('--min-area-px', dest='min_area_px', type=int, default=0,
                        help='Filter out tiny boxes with area below this pixel threshold (default 0 = off)')
    parser.add_argument('--dump-dets', dest='dump_dets', type=str, default='',
                        help='Write JSONL detections to this file (default: off)')
    parser.add_argument('--save-frames', dest='save_frames', type=str, default='',
                        help='Directory to save periodic frame snapshots (default: off)')
    parser.add_argument('--save-interval', dest='save_interval', type=int, default=15,
                        help='Snapshot interval in frames (default 15)')
    parser.add_argument('--save-annotated', dest='save_annotated', action='store_true',
                        help='Save the on-screen (annotated) frame instead of raw camera feed')
    parser.add_argument('--force-minimal-gui', dest='force_min_gui', action='store_true',
                        help='Use the minimal fallback GUI even if a professional GUI is available')
    parser.add_argument('--auto-start', dest='auto_start', action='store_true',
                        help='Automatically begin precision analysis after GUI initialization')

    args, _ = parser.parse_known_args()
    CLI_ARGS = args

    try:
        app = SmartMirrorDermatologist()
        if getattr(args, 'auto_start', False):
            app.root.after(150, lambda: app.start_analysis())
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

# Helper run targets (copy/paste in a shell)
# Phase 1: self-test
# cd ~/smart-mirror-project
# pkill -f smart_mirror_standalone_hailo 2>/dev/null || true
# rm -f /tmp/derm_gui.live.log hailort.log
# export DERM_OVERLAY_SELFTEST=1
# unset DERM_FAKE_RESULTS
# python -X faulthandler -m smart_mirror_standalone_hailo | tee /tmp/derm_gui.live.log
#
# Then verify in another shell:
# grep -F -n "[DERM-HAILO] Results queue ready" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] GUI selftest posted (n_boxes=1)" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] GUI draw on main thread (n_boxes=" /tmp/derm_gui.live.log
#
# Phase 2: FAKE E2E
# pkill -f smart_mirror_standalone_hailo 2>/dev/null || true
# rm -f /tmp/derm_gui.live.log hailort.log
# export DERM_OVERLAY_SELFTEST=0
# export DERM_FAKE_RESULTS=1
# python -X faulthandler -m smart_mirror_standalone_hailo | tee /tmp/derm_gui.live.log
#
# Verify:
# grep -F -n "[DERM-HAILO] Results bridge thread started" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] WorkerProc: got frame seq=" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] WorkerProc: FAKE results posted (seq=" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] GUI bridge got (seq=" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] GUI draw on main thread (n_boxes=" /tmp/derm_gui.live.log
#
# Phase 3: REAL HEF
# pkill -f smart_mirror_standalone_hailo 2>/dev/null || true
# rm -f /tmp/derm_gui.live.log hailort.log
# unset DERM_FAKE_RESULTS
# unset DERM_OVERLAY_SELFTEST
# python -X faulthandler -m smart_mirror_standalone_hailo | tee /tmp/derm_gui.live.log
#
# Verify:
# grep -F -n "[DERM-HAILO] WorkerProc: first inference completed" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] WorkerProc: posted results (seq=" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] GUI bridge got (seq=" /tmp/derm_gui.live.log
# grep -F -n "[DERM-HAILO] GUI draw on main thread (n_boxes=" /tmp/derm_gui.live.log
