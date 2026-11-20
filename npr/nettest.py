"""Connection test for network sources: a fixed sequence of checks, each reported as it finishes.

    address → DNS → TCP reachability → protocol handshake (RTSP/HTTP) → open with FFmpeg → sample frames

``run(url)`` is a generator of step dicts so the dashboard can show progress live. A failed step
ends the run and marks the rest as skipped; every result carries a plain-language hint.
"""
from __future__ import annotations

import base64
import hashlib
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urlsplit

from .appconfig import mask_uri

STEPS = (("address", "Address"), ("dns", "Name lookup"), ("tcp", "Reach the device"),
         ("handshake", "Protocol handshake"), ("open", "Open the stream"), ("frames", "Receive video"))
DEFAULT_PORT = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443, "rtmp": 1935, "rtmps": 443}


def _step(sid: str, status: str, detail: str, ms: float | None = None, hint: str | None = None, **data) -> dict:
    return {"id": sid, "title": dict(STEPS)[sid], "status": status, "detail": mask_uri(detail),
            "ms": None if ms is None else round(ms, 1), "hint": hint, "data": data}


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


# ---- RTSP, spoken by hand so 401 / 404 / wrong-path can be told apart ----------------------
def _rtsp_request(sock, method: str, url: str, cseq: int, auth: str | None = None) -> tuple[int, dict, str]:
    lines = [f"{method} {url} RTSP/1.0", f"CSeq: {cseq}", "User-Agent: npr-nettest"]
    if method == "DESCRIBE":
        lines.append("Accept: application/sdp")
    if auth:
        lines.append(f"Authorization: {auth}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    head, _, body = buf.partition(b"\r\n\r\n")
    rows = head.decode("latin-1").split("\r\n")
    m = re.match(r"RTSP/\d\.\d\s+(\d+)\s*(.*)", rows[0] if rows else "")
    if not m:
        raise ConnectionError("the device did not answer with RTSP — is this the right port?")
    headers: dict = {}
    for row in rows[1:]:
        k, _, v = row.partition(":")
        headers.setdefault(k.strip().lower(), []).append(v.strip())
    want = int((headers.get("content-length") or ["0"])[0] or 0)
    while len(body) < want:
        chunk = sock.recv(4096)
        if not chunk:
            break
        body += chunk
    return int(m.group(1)), headers, body.decode("latin-1", "replace")


def _authorization(challenges: list[str], method: str, url: str, user: str, password: str) -> str | None:
    for ch in challenges:
        if ch.lower().startswith("digest"):
            p = dict(re.findall(r'(\w+)="?([^",]+)"?', ch))
            realm, nonce = p.get("realm", ""), p.get("nonce", "")
            md5 = lambda s: hashlib.md5(s.encode()).hexdigest()  # noqa: E731 (RFC 2069 digest, as cameras use)
            resp = md5(f"{md5(f'{user}:{realm}:{password}')}:{nonce}:{md5(f'{method}:{url}')}")
            return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{url}", response="{resp}"'
    for ch in challenges:
        if ch.lower().startswith("basic"):
            return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    return None


def _rtsp_handshake(u, host: str, port: int) -> dict:
    bare = f"{u.scheme}://{host}:{port}{u.path or '/'}" + (f"?{u.query}" if u.query else "")
    user, password = unquote(u.username or ""), unquote(u.password or "")
    t0 = time.perf_counter()
    sock = socket.create_connection((host, port), timeout=6)
    try:
        if u.scheme == "rtsps":
            sock = ssl._create_unverified_context().wrap_socket(sock, server_hostname=host)
        sock.settimeout(6)
        code, headers, _ = _rtsp_request(sock, "OPTIONS", bare, 1)
        server = (headers.get("server") or [""])[0]
        code, headers, body = _rtsp_request(sock, "DESCRIBE", bare, 2)
        authed = False
        if code == 401 and user:
            auth = _authorization(headers.get("www-authenticate", []), "DESCRIBE", bare, user, password)
            if auth:
                code, headers, body = _rtsp_request(sock, "DESCRIBE", bare, 3, auth)
                authed = True
    finally:
        sock.close()
    ms = _ms(t0)
    who = f" ({server})" if server else ""
    if code == 200:
        tracks = re.findall(r"^m=(\w+)", body, re.M)
        codecs = re.findall(r"^a=rtpmap:\d+\s+([\w.-]+)", body, re.M)
        return _step("handshake", "ok", f"RTSP DESCRIBE accepted{who}: " + (", ".join(
            f"{t} {c}" for t, c in zip(tracks, codecs)) or "stream described"), ms, server=server, code=code)
    if code == 401:
        hint = ("The camera rejected the username or password." if authed else
                "The camera wants a login: add user:password@ to the address.")
        return _step("handshake", "fail", f"401 Unauthorized{who}", ms, hint, code=code)
    if code == 404:
        return _step("handshake", "fail", f"404 Not Found{who} for path {u.path or '/'}", ms,
                     "The device is up and the login was not refused, but it has no stream at this path. Check the "
                     "path in the camera's own app (it changes with stream profile), and whether the camera has "
                     "RTSP turned off, is in privacy/sleep mode, or limits how many viewers may connect.", code=code)
    if code in (453, 503):
        return _step("handshake", "fail", f"{code} — the camera is refusing more viewers{who}", ms,
                     "Close other apps or recorders watching this camera, or raise its session limit.", code=code)
    return _step("handshake", "fail", f"RTSP DESCRIBE answered {code}{who}", ms,
                 "The device answered but would not describe the stream.", code=code)


def _http_handshake(url: str) -> dict:
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "npr-nettest", "Range": "bytes=0-1023"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            code, ctype = r.status, r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        hint = {401: "The server wants a login.", 403: "The server refused access.",
                404: "Nothing is published at this address."}.get(e.code, "The server would not serve this address.")
        return _step("handshake", "fail", f"HTTP {e.code} {e.reason}", _ms(t0), hint, code=e.code)
    return _step("handshake", "ok", f"HTTP {code}, {ctype or 'no content type'}", _ms(t0), code=code)


# ---- the workflow ---------------------------------------------------------------------------
def _sample(url: str, seconds: float):
    """Open with FFmpeg exactly as the app does, then measure what actually arrives."""
    import av

    from .ingest.stream import StreamSource

    t0 = time.perf_counter()
    try:
        container = av.open(url, options=StreamSource.OPTIONS, timeout=10)
    except Exception as e:
        yield _step("open", "fail", f"{type(e).__name__}: {e}", _ms(t0),
                    "FFmpeg could not open the stream even though the device is reachable.")
        return
    try:
        if not container.streams.video:
            yield _step("open", "fail", "the stream has no video track", _ms(t0))
            return
        vs = container.streams.video[0]
        cc = vs.codec_context
        declared = float(vs.average_rate or 0)
        yield _step("open", "ok", f"{cc.name.upper()} {cc.width}×{cc.height}"
                    + (f", declared {declared:.0f} fps" if declared else ""), _ms(t0),
                    codec=cc.name, width=cc.width, height=cc.height, fps=declared)

        t1 = time.perf_counter()
        first = last = None
        n = keys = nbytes = errors = 0
        gaps: list[float] = []
        try:
            for packet in container.demux(vs):
                nbytes += packet.size
                keys += bool(packet.is_keyframe)
                try:
                    frames = packet.decode()
                except av.FFmpegError:
                    errors += 1
                    continue
                now = time.perf_counter()
                for _ in frames:
                    n += 1
                    if first is None:
                        first = now
                    elif last is not None:
                        gaps.append(now - last)
                    last = now
                if (first is not None and now - first >= seconds) or now - t1 > seconds + 12:
                    break
        except Exception as e:
            if not n:
                yield _step("frames", "fail", f"{type(e).__name__}: {e}", _ms(t1),
                            "The stream opened but no picture could be decoded.")
                return
        if not n:
            yield _step("frames", "fail", "no video arrived", _ms(t1),
                        "The stream opened but stayed silent. The camera may be asleep or out of viewer slots.")
            return
        span = max(last - first, 1e-3)
        fps = (n - 1) / span if n > 1 else 0.0
        kbps = nbytes * 8 / 1000 / max(last - t1, 1e-3)
        worst = max(gaps) * 1000 if gaps else 0.0
        mean = sum(gaps) / len(gaps) if gaps else 0.0
        jitter = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 * 1000 if gaps else 0.0
        status, hint = "ok", None
        if errors:
            status, hint = "warn", f"{errors} packet(s) failed to decode — expect glitches (weak Wi-Fi or packet loss)."
        elif declared and fps < declared * 0.7:
            status, hint = "warn", "Frames are arriving well below the declared rate: the link or the camera is struggling."
        elif mean and worst > max(400.0, mean * 8000):
            status, hint = "warn", "Delivery is bursty (long pauses between frames)."
        yield _step("frames", status,
                    f"{n} frames in {span:.1f}s = {fps:.1f} fps · {kbps:.0f} kbit/s · first frame after "
                    f"{(first - t1) * 1000:.0f} ms · worst gap {worst:.0f} ms · jitter {jitter:.0f} ms", _ms(t1), hint,
                    fps=round(fps, 1), kbps=round(kbps), first_frame_ms=round((first - t1) * 1000),
                    worst_gap_ms=round(worst), jitter_ms=round(jitter), keyframes=keys, decode_errors=errors)
    finally:
        container.close()


def run(url: str, seconds: float = 4.0):
    done: set[str] = set()

    def finish(step: dict):
        done.add(step["id"])
        return step

    def skip_rest():
        for sid, _ in STEPS:
            if sid not in done:
                yield _step(sid, "skip", "not run")

    u = urlsplit(url.strip())
    scheme = u.scheme.lower()
    if scheme not in DEFAULT_PORT or not u.hostname:
        yield finish(_step("address", "fail", f"not a network stream address: {url}", None,
                           "Expected rtsp://, rtsps://, http(s):// or rtmp(s):// followed by a host."))
        yield from skip_rest()
        return
    host, port = u.hostname, u.port or DEFAULT_PORT[scheme]
    yield finish(_step("address", "ok", f"{scheme.upper()} · {host}:{port} · path {u.path or '/'} · "
                       + ("login included" if u.username else "no login"), None, host=host, port=port))

    t0 = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addrs = sorted({i[4][0] for i in infos})
    except OSError as e:
        yield finish(_step("dns", "fail", f"cannot resolve {host}: {e.strerror or e}", _ms(t0),
                           "The name does not exist on this network. Try the camera's IP address."))
        yield from skip_rest()
        return
    literal = addrs == [host]
    yield finish(_step("dns", "ok", f"{host} is an IP address" if literal else f"{host} → {', '.join(addrs[:3])}",
                       None if literal else _ms(t0), addresses=addrs))

    times, err = [], None
    for _ in range(4):
        t0 = time.perf_counter()
        try:
            socket.create_connection((addrs[0], port), timeout=4).close()
            times.append(_ms(t0))
        except OSError as e:
            err = e
        time.sleep(0.05)
    if not times:
        refused = isinstance(err, ConnectionRefusedError)
        yield finish(_step("tcp", "fail", f"{addrs[0]}:{port} — {'connection refused' if refused else err}", None,
                           "The device is on the network but nothing is listening on this port (is streaming enabled on it?)." if refused
                           else "No answer: the device is off, on another network/VLAN, or a firewall is in the way."))
        yield from skip_rest()
        return
    lost = 4 - len(times)
    yield finish(_step("tcp", "warn" if lost or max(times) > 150 else "ok",
                       f"port {port} open · connect {min(times):.0f}/{sum(times) / len(times):.0f}/{max(times):.0f} ms "
                       f"(min/avg/max of {len(times)})" + (f" · {lost} of 4 attempts failed" if lost else ""),
                       sum(times) / len(times),
                       "The link to the device is slow or unreliable." if lost or max(times) > 150 else None,
                       min_ms=round(min(times), 1), avg_ms=round(sum(times) / len(times), 1),
                       max_ms=round(max(times), 1), lost=lost))

    if scheme in ("rtsp", "rtsps", "http", "https"):
        try:
            step = _rtsp_handshake(u, addrs[0], port) if scheme.startswith("rtsp") else _http_handshake(url)
        except Exception as e:
            step = _step("handshake", "fail", f"{type(e).__name__}: {e}", None, "The device closed or stalled the conversation.")
        yield finish(step)
        if step["status"] == "fail":
            yield from skip_rest()
            return
    else:
        yield finish(_step("handshake", "skip", "no lightweight handshake for this protocol"))

    for step in _sample(url, seconds):
        yield finish(step)
    yield from skip_rest()


def probe_paths(url: str, paths: list[str]) -> list[dict]:
    """Ask an RTSP device which of several stream paths exist (one DESCRIBE each, no video pulled).

    Returns one row per path: ``{"path", "code", "ok", "detail"}``. Stops early on a login failure,
    since every other path would fail the same way.
    """
    u = urlsplit(url.strip())
    if u.scheme.lower() not in ("rtsp", "rtsps") or not u.hostname:
        raise ValueError("path probing needs an rtsp:// address")
    port = u.port or DEFAULT_PORT[u.scheme.lower()]
    out = []
    for path in paths:
        try:
            step = _rtsp_handshake(u._replace(path=path, query=""), u.hostname, port)
            code = step["data"].get("code")
            out.append({"path": path, "code": code, "ok": step["status"] == "ok", "detail": step["detail"]})
            if code == 401:
                break
        except Exception as e:
            out.append({"path": path, "code": None, "ok": False, "detail": mask_uri(f"{type(e).__name__}: {e}")})
            break
    return out
