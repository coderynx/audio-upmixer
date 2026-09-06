#!/usr/bin/env python3
"""Small localhost UI for completing a blinded listening-pack ratings CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import stat
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import quote, unquote, urlsplit


KEY_FIELDS = ("listener_id", "round_id", "case_id")
REQUIRED_COLUMNS = (
    *KEY_FIELDS,
    "preference",
    "confidence_1_5",
    "comparison_gain_db",
    "original_level_check",
)


class ReviewStore:
    """Load one pack and safely persist its ratings."""

    def __init__(self, pack_dir: Path) -> None:
        self.pack_dir = pack_dir.resolve()
        self.ratings_path = self.pack_dir / "ratings.csv"
        manifest_path = self.pack_dir / "manifest.json"
        if (
            not self.pack_dir.is_dir()
            or not manifest_path.is_file()
            or not self.ratings_path.is_file()
        ):
            raise ValueError("pack must contain manifest.json and ratings.csv")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.cases: list[dict[str, object]] = []
        self.audio_paths: dict[tuple[str, str], Path] = {}
        seen: set[str] = set()
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("manifest must contain cases")
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("manifest cases must be objects")
            case_id = case.get("case_id")
            if not isinstance(case_id, str):
                raise ValueError("manifest cases need string case_id values")
            case_id = case_id.strip()
            if not case_id or case_id in seen:
                raise ValueError("manifest cases need unique case_id values")
            seen.add(case_id)
            audio = case.get("audio", {})
            if not isinstance(audio, dict):
                raise ValueError(f"invalid audio map for {case_id}")
            safe_audio: dict[str, str] = {}
            for asset, details in audio.items():
                if not isinstance(asset, str) or not asset:
                    raise ValueError(f"invalid audio asset for {case_id}")
                declared = details.get("path") if isinstance(details, dict) else details
                if not isinstance(declared, str) or not declared:
                    raise ValueError(f"invalid audio path for {case_id}/{asset}")
                path = (self.pack_dir / declared).resolve()
                try:
                    path.relative_to(self.pack_dir)
                except ValueError as exc:
                    raise ValueError(f"audio path escapes pack: {declared}") from exc
                asset_name = str(asset)
                self.audio_paths[(case_id, asset_name)] = path
                safe_audio[asset_name] = (
                    f"/audio/{quote(case_id, safe='')}/{quote(asset_name, safe='')}"
                )
            self.cases.append(
                {
                    "case_id": case_id,
                    "category": case.get("category", ""),
                    "item_id": case.get("item_id", ""),
                    "rate_arm": case.get("rate_arm", ""),
                    "rate_hz": case.get("rate_hz", ""),
                    "stem": case.get("stem", ""),
                    "audio": safe_audio,
                }
            )
        self.case_ids = {str(case["case_id"]) for case in self.cases}
        with self.ratings_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("ratings.csv must have a header")
            self.columns = list(reader.fieldnames)
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("ratings.csv must have unique columns")
        missing = set(REQUIRED_COLUMNS) - set(self.columns)
        if missing:
            raise ValueError(
                f"ratings.csv missing columns: {', '.join(sorted(missing))}"
            )
        self.lock = Lock()

    def rows(self) -> list[dict[str, str]]:
        with self.ratings_path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def state(self) -> dict[str, object]:
        return {"columns": self.columns, "cases": self.cases}

    def audio_path(self, case_id: str, asset: str) -> Path:
        path = self.audio_paths.get((case_id, asset))
        if path is None or not path.is_file():
            raise FileNotFoundError(asset)
        return path

    def save(self, payload: object) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        unknown = set(payload) - set(self.columns)
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
        row = {column: str(payload.get(column, "")) for column in self.columns}
        for field in KEY_FIELDS:
            row[field] = row[field].strip()
            if not row[field]:
                raise ValueError(f"{field} is required")
        if row["case_id"] not in self.case_ids:
            raise ValueError("unknown case_id")
        if row.get("preference") not in {"A", "B", "TIE"}:
            raise ValueError("preference must be A, B, or TIE")
        confidence = _integer(row.get("confidence_1_5", ""), "confidence_1_5")
        if not 1 <= confidence <= 5:
            raise ValueError("confidence_1_5 must be 1-5")
        for field in self.columns:
            if field.endswith("_0_3"):
                value = _integer(row[field], field)
                if not 0 <= value <= 3:
                    raise ValueError(f"{field} must be 0-3")
        gain = row.get("comparison_gain_db", "")
        try:
            if not math.isfinite(float(gain)):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("comparison_gain_db must be numeric") from exc
        if not row.get("original_level_check", "").strip():
            raise ValueError("original_level_check is required")

        with self.lock:
            with self.ratings_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if list(reader.fieldnames or []) != self.columns:
                    raise ValueError("ratings.csv header changed")
                rows = list(reader)
            key = tuple(row[field] for field in KEY_FIELDS)
            replaced = False
            for index, existing in enumerate(rows):
                if tuple(existing.get(field, "") for field in KEY_FIELDS) == key:
                    rows[index] = row
                    replaced = True
                    break
            if not replaced:
                rows = [
                    existing
                    for existing in rows
                    if not (
                        existing.get("case_id", "").strip() == row["case_id"]
                        and not existing.get("listener_id", "").strip()
                        and not existing.get("round_id", "").strip()
                    )
                ]
                rows.append(row)
            _atomic_csv_write(self.ratings_path, self.columns, rows)
        return row


def _integer(value: str, field: str) -> int:
    value = value.strip()
    if not value:
        raise ValueError(f"{field} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _atomic_csv_write(
    path: Path, columns: list[str], rows: list[dict[str, str]]
) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = file.name
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _page(store: ReviewStore) -> bytes:
    state = json.dumps(store.state(), separators=(",", ":")).replace("<", "\\u003c")
    return (
        """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Listening review</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}
fieldset{border:1px solid #bbb;margin:1rem 0;padding:1rem} label{display:block;margin:.45rem 0}
input,select,textarea{font:inherit;max-width:100%;box-sizing:border-box} input,textarea{width:18rem}
.audio{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem}
audio{width:100%}.nav{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:.35rem .7rem} #status{min-height:1.5em}
</style>
<body><h1>Listening review</h1>
<div class="nav"><button id="previous">Previous</button><select id="case"></select><button id="next">Next</button></div>
<p id="meta"></p><section class="audio" id="audio"></section><form id="form"></form>
<p><button id="save">Save rating</button> <span id="status" role="status"></span></p>
<script id="review-data" type="application/json">"""
        + state
        + """</script>
<script>
const data=JSON.parse(document.getElementById('review-data').textContent), form=document.getElementById('form'), status=document.getElementById('status'); data.rows=[];
const caseSelect=document.getElementById('case'), audio=document.getElementById('audio'), meta=document.getElementById('meta');
const fields=data.columns.filter(x=>x!=='case_id'); let index=0;
for(const c of data.cases){const o=document.createElement('option');o.value=c.case_id;o.textContent=c.case_id;caseSelect.append(o)}
function control(name){const label=document.createElement('label');label.textContent=name+' ';let el;
if(name==='preference'){el=document.createElement('select');for(const v of ['A','B','TIE']){const o=document.createElement('option');o.value=v;o.textContent=v;el.append(o)}}
else if(name==='confidence_1_5'){el=document.createElement('select');for(let v=1;v<=5;v++){const o=document.createElement('option');o.value=v;o.textContent=v;el.append(o)}}
else if(name.endsWith('_0_3')){el=document.createElement('select');for(let v=0;v<=3;v++){const o=document.createElement('option');o.value=v;o.textContent=v;el.append(o)}}
else if(name==='original_level_check'||name==='notes'){el=document.createElement(name==='notes'?'textarea':'input')}else{el=document.createElement('input');el.type=name==='comparison_gain_db'?'number':'text';if(el.type==='number')el.step='any'}
el.name=name;label.append(el);form.append(label)}
for(const name of fields)control(name);
function row(){return data.rows.find(r=>r.listener_id===form.elements.listener_id.value&&r.round_id===form.elements.round_id.value&&r.case_id===caseSelect.value)}
function render(){const c=data.cases[index], identity={listener_id:form.elements.listener_id?.value||'',round_id:form.elements.round_id?.value||''};caseSelect.value=c.case_id;meta.textContent=[c.category,c.item_id,c.rate_arm,c.rate_hz&&c.rate_hz+' Hz',c.stem].filter(Boolean).join(' · ');audio.replaceChildren();
for(const [name,url] of Object.entries(c.audio)){const label=document.createElement('label');label.textContent=name;const player=document.createElement('audio');player.controls=true;player.preload='metadata';player.src=url;label.append(player);audio.append(label)}
for(const el of form.elements){if(el.name==='case_id'||el.name==='listener_id'||el.name==='round_id')continue;el.value='';}for(const [name,value] of Object.entries(identity))if(form.elements[name])form.elements[name].value=value;const saved=row();if(saved)for(const el of form.elements)if(saved[el.name]!==undefined)el.value=saved[el.name]}
function move(delta){index=(index+delta+data.cases.length)%data.cases.length;render()}
caseSelect.onchange=()=>{index=data.cases.findIndex(c=>c.case_id===caseSelect.value);render()};previous.onclick=()=>move(-1);next.onclick=()=>move(1);
document.getElementById('save').onclick=async()=>{const payload=Object.fromEntries(new FormData(form));payload.case_id=caseSelect.value;status.textContent='Saving…';
try{const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok)throw Error(result.error);data.rows=data.rows.filter(r=>r.listener_id!==result.listener_id||r.round_id!==result.round_id||r.case_id!==result.case_id);data.rows.push(result);status.textContent='Saved'}catch(error){status.textContent=error.message}}
render();
</script></body></html>"""
    ).encode("utf-8")


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def do_GET(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        if route == "/":
            self._send(200, "text/html; charset=utf-8", _page(self.store))
            return
        if route == "/api/state":
            self._send(200, "application/json", _json_bytes(self.store.state()))
            return
        parts = route.split("/")
        if len(parts) == 4 and parts[1] == "audio":
            try:
                path = self.store.audio_path(unquote(parts[2]), unquote(parts[3]))
            except FileNotFoundError:
                self.send_error(404, "audio not found")
                return
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(404, "audio not found")
                return
            self._send(
                200,
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                body,
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/save":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or length > 1_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            row = self.store.save(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, "application/json", _json_bytes({"error": str(exc)}))
            return
        self._send(200, "application/json", _json_bytes(row))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_server(pack_dir: Path, port: int = 8765) -> HTTPServer:
    store = ReviewStore(pack_dir)

    class Handler(ReviewHandler):
        pass

    Handler.store = store
    return HTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_dir", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.pack_dir, args.port)
    print(f"Listening review: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
