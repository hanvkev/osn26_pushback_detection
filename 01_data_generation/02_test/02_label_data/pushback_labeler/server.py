from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import threading
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[3]
STATIC = APP_DIR / "static"
TILE_CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "pushback_labeler" / "tiles"
DEFAULTS = json.loads((APP_DIR / "config.example.json").read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_dict(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def blank_annotation() -> dict[str, Any]:
    return {"status": "not_labelled", "segments": {"pushback": [], "push": [], "pull": []}, "tags": [], "note": "", "updated_at": None, "label_version": 1}


class Runtime:
    def __init__(self) -> None:
        self.source: Path | None = None
        self.output: Path | None = None
        self.settings = json.loads(json.dumps(DEFAULTS))
        self.annotations: dict[str, Any] = {}
        self.index = pd.DataFrame()
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.pickle_frame: pd.DataFrame | None = None
        self.lock = threading.RLock()

    @property
    def columns(self) -> dict[str, str]: return self.settings["columns"]

    def configure(self, source: str, output: str, columns: dict[str, str], metadata: dict[str, str] | None = None) -> None:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if not path.exists(): raise FileNotFoundError(f"Datenquelle nicht gefunden: {path}")

        out = Path(output).expanduser()
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
        saved = out / "settings.json"
        self.settings = merge_dict(json.loads(json.dumps(DEFAULTS)), json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {})
        self.settings["columns"].update({k: v for k, v in columns.items() if v})
        self.settings["dataset"].update({k: str(v).strip() for k, v in (metadata or {}).items() if str(v).strip()})
        self.settings["dataset"].pop("hemisphere", None)
        self.settings["dataset"].pop("tag", None)
        self.source, self.output = path, out
        af = out / "annotations.json"
        self.annotations = json.loads(af.read_text(encoding="utf-8")) if af.exists() else {}
        self.cache.clear(); self.pickle_frame = None
        self.index = self._load_index()
        date_value = self.settings["dataset"].get("date", "")
        if date_value:
            month = pd.Timestamp(date_value).month
            north = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
            self.settings["dataset"]["season"] = north[month]
        atomic_json(saved, self.settings)

    def _pickle(self) -> pd.DataFrame:
        if self.pickle_frame is None:
            with self.source.open("rb") as handle: obj = pickle.load(handle)  # type: ignore[union-attr]
            if isinstance(obj, pd.DataFrame): self.pickle_frame = obj
            elif hasattr(obj, "data") and isinstance(obj.data, pd.DataFrame): self.pickle_frame = obj.data
            else: raise TypeError("Pickle enthält kein Traffic-, Flight- oder DataFrame-Objekt.")
        return self.pickle_frame

    def _load_index(self) -> pd.DataFrame:
        c = self.columns; fid = c["flight_id"]
        requested = list(dict.fromkeys(x for x in [fid, c.get("airport", ""), c.get("season", "")] if x))
        assert self.source is not None
        if self.source.suffix.lower() in {".pkl", ".pickle"}:
            full = self._pickle(); frame = full[[x for x in requested if x in full.columns]]
        elif self.source.suffix.lower() == ".csv":
            frame = pd.read_csv(self.source, usecols=lambda x: x in set(requested))
        else:
            import pyarrow.dataset as ds
            dataset = ds.dataset(str(self.source), format="parquet")
            frame = dataset.to_table(columns=[x for x in requested if x in dataset.schema.names]).to_pandas()
        if fid not in frame: raise KeyError(f"Flight-ID-Spalte '{fid}' fehlt. Vorhanden: {list(frame.columns)}")
        frame[fid] = frame[fid].astype(str)
        return frame.drop_duplicates(fid).reset_index(drop=True)

    def flight(self, flight_id: str) -> pd.DataFrame:
        with self.lock:
            if flight_id in self.cache:
                result = self.cache.pop(flight_id); self.cache[flight_id] = result; return result
        c, source = self.columns, self.source; assert source is not None; fid = c["flight_id"]
        if source.suffix.lower() in {".pkl", ".pickle"}:
            frame = self._pickle().loc[self._pickle()[fid].astype(str) == flight_id].copy()
        elif source.suffix.lower() == ".csv":
            parts = [part for chunk in pd.read_csv(source, chunksize=200_000) if not (part := chunk.loc[chunk[fid].astype(str) == flight_id]).empty]
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        else:
            import pyarrow.dataset as ds
            import pyarrow.types as pat
            dataset = ds.dataset(str(source), format="parquet"); dtype = dataset.schema.field(fid).type; value: Any = flight_id
            if pat.is_integer(dtype): value = int(flight_id)
            elif pat.is_floating(dtype): value = float(flight_id)
            frame = dataset.to_table(filter=ds.field(fid) == value).to_pandas()
        ts = c["timestamp"]; frame[ts] = pd.to_datetime(frame[ts], errors="coerce", utc=True)
        frame = frame.dropna(subset=[ts]).sort_values(ts).reset_index(drop=True); frame["__point_idx"] = np.arange(len(frame))
        with self.lock:
            self.cache[flight_id] = frame
            while len(self.cache) > 2: self.cache.popitem(last=False)
        return frame

    def annotation(self, fid: str) -> dict:
        base = blank_annotation(); saved = self.annotations.get(fid, {})
        base.update({k: v for k, v in saved.items() if k != "segments"})
        for kind in base["segments"]: base["segments"][kind] = saved.get("segments", {}).get(kind, [])
        return base

    def save(self, fid: str, annotation: dict) -> None:
        assert self.output is not None
        annotation["dataset"] = dict(self.settings.get("dataset", {}))
        annotation["updated_at"] = datetime.now(timezone.utc).isoformat(); self.annotations[fid] = annotation
        atomic_json(self.output / "annotations.json", self.annotations)
        rows = [{"flight_id": key, "airport": item.get("dataset", {}).get("airport", ""), "date": item.get("dataset", {}).get("date", ""), "season": item.get("dataset", {}).get("season", ""), "status": item.get("status", "not_labelled"), "pushback_segments": json.dumps(item.get("segments", {}).get("pushback", [])), "push_segments": json.dumps(item.get("segments", {}).get("push", [])), "pull_segments": json.dumps(item.get("segments", {}).get("pull", [])), "tags": json.dumps(item.get("tags", [])), "note": item.get("note", ""), "updated_at": item.get("updated_at"), "label_version": item.get("label_version", 1)} for key, item in self.annotations.items()]
        temp = self.output / "annotations.tmp"; pd.DataFrame(rows).to_csv(temp, index=False); os.replace(temp, self.output / "annotations.csv")


RUNTIME = Runtime()


def approximate_gs(frame: pd.DataFrame) -> np.ndarray:
    c = RUNTIME.columns; lat, lon = np.radians(frame[c["latitude"]]), np.radians(frame[c["longitude"]]); dt = np.diff(frame[c["timestamp"]].astype("int64") / 1e9)
    dlat, dlon = np.diff(lat), np.diff(lon); a = np.sin(dlat/2)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dlon/2)**2
    distance = 6_371_000*2*np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0, 1-a))); speed = np.divide(distance, dt, out=np.full(len(dt), np.nan), where=dt>0)*1.943844
    return pd.Series(np.r_[np.nan, speed]).interpolate(limit_direction="both").to_numpy()


class LoadRequest(BaseModel):
    source: str; output: str; columns: dict[str, str]; metadata: dict[str, str] = Field(default_factory=dict)


app = FastAPI(title="Pushback Labeler")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def root(): return FileResponse(STATIC / "index.html")


@app.get("/tiles/{provider}/{z}/{x}/{y}.png")
async def map_tile(provider: str, z: int, x: int, y: int):
    if not (0 <= z <= 19 and 0 <= x < 2**z and 0 <= y < 2**z):
        raise HTTPException(404, "Ungültige Kartenkachel")
    providers = {
        "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "carto-light": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "carto-dark": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    }
    if provider not in providers: raise HTTPException(404, "Unbekannter Kartenstil")
    target = TILE_CACHE / provider / str(z) / str(x) / f"{y}.png"
    if not target.exists():
        def download() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(
                providers[provider].format(z=z, x=x, y=y),
                headers={"User-Agent": "pushback-labeler/1.0 (local research tool)"},
            )
            with urllib.request.urlopen(request, timeout=15) as source:
                data = source.read()
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(data)
            os.replace(temporary, target)
        try:
            await run_in_threadpool(download)
        except Exception as exc:
            raise HTTPException(502, f"Kartenkachel konnte nicht geladen werden: {exc}") from exc
    return Response(target.read_bytes(), media_type="image/png", headers={"Cache-Control": "public, max-age=2592000, immutable"})


@app.post("/api/load")
def load(req: LoadRequest):
    try: RUNTIME.configure(req.source, req.output, req.columns, req.metadata); return status()
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


@app.get("/api/status")
def status():
    if RUNTIME.index.empty: return {"loaded": False}
    ids = RUNTIME.index[RUNTIME.columns["flight_id"]].astype(str).tolist()
    statuses = {fid: RUNTIME.annotations.get(fid, {}).get("status", "not_labelled") for fid in ids}
    counts = {s: sum(value == s for value in statuses.values()) for s in ["not_labelled", "labelled", "no_pushback", "not_clear"]}
    return {"loaded": True, "flights": ids, "statuses": statuses, "counts": counts, "total": len(ids), "tags": RUNTIME.settings["tags"], "display": RUNTIME.settings["display"], "dataset": RUNTIME.settings.get("dataset", {})}


@app.get("/api/flight/{flight_id}")
def flight(flight_id: str):
    frame = RUNTIME.flight(flight_id); c = RUNTIME.columns
    if frame.empty: raise HTTPException(404, "Flug nicht gefunden")
    gs = pd.to_numeric(frame[c["groundspeed"]], errors="coerce").to_numpy() if c["groundspeed"] in frame else approximate_gs(frame)
    return {"flight_id": flight_id, "timestamp": [x.isoformat() for x in frame[c["timestamp"]]], "latitude": pd.to_numeric(frame[c["latitude"]], errors="coerce").replace({np.nan: None}).tolist(), "longitude": pd.to_numeric(frame[c["longitude"]], errors="coerce").replace({np.nan: None}).tolist(), "gs": pd.Series(gs).replace({np.nan: None}).tolist(), "annotation": RUNTIME.annotation(flight_id)}


@app.put("/api/annotation/{flight_id}")
def save_annotation(flight_id: str, annotation: dict): RUNTIME.save(flight_id, annotation); return {"ok": True, "annotation": annotation}


@app.put("/api/tags")
def save_tags(payload: dict):
    tag = str(payload.get("tag", "")).strip()
    if tag and tag not in RUNTIME.settings["tags"]: RUNTIME.settings["tags"].append(tag); atomic_json(RUNTIME.output / "settings.json", RUNTIME.settings)  # type: ignore[operator]
    return {"tags": RUNTIME.settings["tags"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=9435); args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
