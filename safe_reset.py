#!/usr/bin/env python

import argparse
from models.lock import WaitLock
from os import system


def safe_reset(devices, topology):
    """
    Perform a safe reset by releasing a lock, resetting the system, and acquiring the lock again.
    """
    w = WaitLock()
    system(f'bash -c "source /opt/tt_metal_infra/provisioning/provisioning_env/bin/activate; tt-smi -r {devices}"')
    # if topology:
    #     system(
    #         f'bash -c "source /opt/tt_metal_infra/provisioning/provisioning_env/bin/activate; tt-topology -l {args.topology}"'
    #     )
    w.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a safe reset.")
    parser.add_argument("-d", "--devices", type=str, default="0,1,2,3", help="Devices to reset (default: 0,1,2,3)")
    parser.add_argument(
        "-t", "--topology", type=str, default="mesh", help="Topology to set after reset (default: mesh)"
    )
    args = parser.parse_args()
    safe_reset(args.devices, args.topology)
