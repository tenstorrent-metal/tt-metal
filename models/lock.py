# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time
import pwd


class WaitLock:
    def __init__(self):
        self.lock_file = "/tmp/tenstorrent.lock"
        self.pid = os.getpid()
        self._acquire_lock()

    def _is_lock_valid(self):
        """Check if the lock file exists and if the PID within it corresponds to a running process."""
        if not os.path.exists(self.lock_file):
            return False
        try:
            with open(self.lock_file, "r") as f:
                pid = int(f.read().strip())
            # Check if this PID process exists by sending signal 0 (no signal, just error checking)
            os.kill(pid, 0)
        except (OSError, ValueError):
            # OSError if the process doesn't exist, ValueError if PID is not an integer
            return False
        return True

    def _acquire_lock(self):
        """Acquire the lock, waiting and retrying if necessary."""
        prev_user = None
        while True:
            if not self._is_lock_valid():
                try:
                    # Attempt to acquire the lock
                    with open(self.lock_file, "w") as f:
                        f.write(str(self.pid))
                    # Make the file world-writable in case our process dies
                    os.chmod(self.lock_file, 0o666)
                    # Double-check if the lock was successfully acquired
                    if self._is_lock_valid():
                        break
                except OSError:
                    # If lock acquisition failed due to an OS error, assume another process is acquiring the lock
                    pass
            try:
                user = pwd.getpwuid(os.stat(self.lock_file).st_uid).pw_name
            except OSError:
                pass
            if user != prev_user:
                print(f"Waiting for lock to be released by {user}")
                prev_user = user
            time.sleep(1)

    def release(self):
        """Release the lock."""
        try:
            if self._is_lock_valid():
                os.remove(self.lock_file)
        except OSError:
            pass

    def __del__(self):
        """Delete the lock file upon destruction of the object if this process holds the lock."""
        self.release()


# Example usage
if __name__ == "__main__":
    lock = WaitLock()
    print(f"Lock acquired by PID: {os.getpid()}")
    # Do some work here
    # Lock will be automatically released when the program ends or the lock object is deleted
