#!/usr/bin/env python3
"""
Standalone Ethernet Link Diagnostics for Tenstorrent Devices

Reads ethernet core registers directly via ttexalens — does NOT require
Metal runtime, inspector logs, or a running vLLM process.

Checks per ethernet core:
  - Heartbeat (firmware running?)
  - RX Link Up (physical link active?)
  - Retrain Count (link instability indicator)
  - NOC error counters (parity, header ECC errors)

Usage (inside container with ttexalens installed):
    python scripts/eth_diagnostics.py
    python scripts/eth_diagnostics.py --devices 0,1,2  # specific devices only
    python scripts/eth_diagnostics.py --noc              # include NOC error counters
"""

import argparse
import sys
import time


def install_ttexalens():
    """Try to install ttexalens if not present."""
    try:
        import ttexalens
        return True
    except ImportError:
        pass

    import subprocess
    import os

    # Try install_debugger.sh (tt-metal standard installer)
    tt_metal = os.environ.get("TT_METAL_HOME", "/tt-metal")
    installer = os.path.join(tt_metal, "scripts", "install_debugger.sh")
    if os.path.exists(installer):
        print("Installing ttexalens via install_debugger.sh...")
        result = subprocess.run(["bash", installer], capture_output=True, text=True)
        if result.returncode == 0:
            try:
                import ttexalens
                return True
            except ImportError:
                pass
        print(f"install_debugger.sh failed: {result.stderr[-200:]}")

    print("ERROR: ttexalens not available. Install it first:")
    print("  cd /tt-metal && bash scripts/install_debugger.sh")
    return False


# Wormhole register addresses
WH_HEARTBEAT = 0x1C
WH_RX_LINK_UP = 0x1EC0 + 0x20   # 0x1EE0
WH_RETRAIN_COUNT = 0x1EC0 + 0x28  # 0x1EE8

# Blackhole register addresses
BH_HEARTBEAT = 0x7CC70
BH_PORT_STATUS = 0x7CC04
BH_RX_LINK_UP = 0x7CE04
BH_RETRAIN_COUNT = 0x7CE00

# NOC error registers (both architectures)
NOC0_BASE = 0xFFB20000
NOC1_BASE = 0xFFB30000
NOC_MEM_PARITY_ERR = 0x50
NOC_HEADER_1B_ERR = 0x54
NOC_HEADER_2B_ERR = 0x58


def get_coord(loc):
    """Extract (x, y) from an OnChipCoordinate."""
    coord_info = loc.to("logical")
    if isinstance(coord_info, tuple) and len(coord_info) >= 2:
        if isinstance(coord_info[0], tuple) and len(coord_info[0]) == 2:
            return coord_info[0]
    return None


def check_heartbeat(read_word, loc, addr, context, samples=50):
    """Check if heartbeat register is changing (firmware alive)."""
    prev = read_word(loc, addr, context=context)
    for _ in range(samples):
        val = read_word(loc, addr, context=context)
        if val != prev:
            return True
    return False


def run_diagnostics(device_ids=None, check_noc=False):
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens import read_word_from_device

    print("Initializing ttexalens (no Metal runtime required)...")
    context = init_ttexalens()
    print(f"Found {len(context.devices)} device(s)\n")

    total_cores = 0
    link_up_count = 0
    link_down_count = 0
    retrain_issues = 0
    heartbeat_dead = 0
    noc_errors_total = 0

    for device_id, device in sorted(context.devices.items()):
        if device_ids and device_id not in device_ids:
            continue

        is_wh = device.is_wormhole()
        arch = "Wormhole" if is_wh else "Blackhole" if device.is_blackhole() else "Unknown"
        print(f"=== Device {device_id} ({arch}) ===")

        eth_locations = device.get_block_locations("eth")
        if not eth_locations:
            print("  No ethernet cores found\n")
            continue

        # Table header
        print(f"  {'Core':>8}  {'Heartbeat':>10}  {'RX Link':>8}  {'Retrain':>8}", end="")
        if not is_wh:
            print(f"  {'Port':>8}", end="")
        if check_noc:
            print(f"  {'NOC0 Err':>9}  {'NOC1 Err':>9}", end="")
        print()
        print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}", end="")
        if not is_wh:
            print(f"  {'─'*8}", end="")
        if check_noc:
            print(f"  {'─'*9}  {'─'*9}", end="")
        print()

        for loc in eth_locations:
            coord = get_coord(loc)
            if coord is None:
                continue
            total_cores += 1
            x, y = coord

            try:
                if is_wh:
                    hb = check_heartbeat(read_word_from_device, loc, WH_HEARTBEAT, context)
                    rx_link = read_word_from_device(loc, WH_RX_LINK_UP, context=context)
                    retrain = read_word_from_device(loc, WH_RETRAIN_COUNT, context=context)
                    port_str = None
                else:
                    hb = check_heartbeat(read_word_from_device, loc, BH_HEARTBEAT, context)
                    rx_link = read_word_from_device(loc, BH_RX_LINK_UP, context=context)
                    retrain = read_word_from_device(loc, BH_RETRAIN_COUNT, context=context)
                    port_raw = read_word_from_device(loc, BH_PORT_STATUS, context=context)
                    port_map = {0: "None", 1: "Up", 2: "Down", 3: "Unused"}
                    port_str = port_map.get(port_raw, f"?{port_raw}")

                rx_up = bool(rx_link)
                hb_str = "alive" if hb else "DEAD"
                rx_str = "Up" if rx_up else "DOWN"
                rt_str = str(retrain) if retrain == 0 else f"**{retrain}**"

                if rx_up:
                    link_up_count += 1
                else:
                    link_down_count += 1
                if retrain > 0:
                    retrain_issues += 1
                if not hb:
                    heartbeat_dead += 1

                line = f"  ({x:2},{y:2})  {hb_str:>10}  {rx_str:>8}  {rt_str:>8}"
                if port_str is not None:
                    line += f"  {port_str:>8}"

                if check_noc:
                    noc0_err = 0
                    noc1_err = 0
                    for offset in [NOC_MEM_PARITY_ERR, NOC_HEADER_1B_ERR, NOC_HEADER_2B_ERR]:
                        try:
                            noc0_err += read_word_from_device(loc, NOC0_BASE + offset, context=context)
                            noc1_err += read_word_from_device(loc, NOC1_BASE + offset, context=context)
                        except Exception:
                            pass
                    noc_errors_total += noc0_err + noc1_err
                    n0 = str(noc0_err) if noc0_err == 0 else f"**{noc0_err}**"
                    n1 = str(noc1_err) if noc1_err == 0 else f"**{noc1_err}**"
                    line += f"  {n0:>9}  {n1:>9}"

                print(line)

            except Exception as e:
                print(f"  ({x:2},{y:2})  ERROR: {e}")

        print()

    # Summary
    print("=" * 60)
    print(f"SUMMARY: {total_cores} ethernet cores across {len(context.devices)} devices")
    print(f"  Links up:       {link_up_count}")
    print(f"  Links DOWN:     {link_down_count}")
    print(f"  Retrain issues: {retrain_issues}")
    print(f"  Dead heartbeat: {heartbeat_dead}")
    if check_noc:
        print(f"  NOC errors:     {noc_errors_total}")

    if link_down_count == 0 and retrain_issues == 0 and heartbeat_dead == 0:
        print("\nAll ethernet links healthy!")
        return 0
    else:
        print("\nWARNING: Issues detected — see above for details")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Standalone ethernet link diagnostics")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated device IDs to check (default: all)")
    parser.add_argument("--noc", action="store_true",
                        help="Also check NOC error counters on each ethernet core")
    args = parser.parse_args()

    if not install_ttexalens():
        sys.exit(1)

    device_ids = None
    if args.devices:
        device_ids = [int(x.strip()) for x in args.devices.split(",")]

    sys.exit(run_diagnostics(device_ids=device_ids, check_noc=args.noc))


if __name__ == "__main__":
    main()
