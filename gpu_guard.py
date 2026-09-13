# =============================================================================
# GPU GUARD — keep utilisation AND power below the levels that hard-power-off
# this machine.
#
# Symptom (10, 11 Sep): sustained ~98% utilisation for 10-15 min causes an
# abrupt system power-off. This card has a 600 W default limit and idles near
# 90 W; measured bursts during training reach 540 W. Sustained load at that
# level most likely trips the PSU's overcurrent protection.
#
# THE PROPER FIX needs admin, one time, and survives until reboot:
#     nvidia-smi -pm 1
#     nvidia-smi -pl 300         # run from an ADMINISTRATOR shell
# Everything below is the software fallback for when that is not set.
#
# A first attempt throttled once per client round. That was too coarse: work
# came in 1-2 s flat-out bursts hitting 538 W, with idle only between bursts.
# Peak power, not average, is what trips the supply. So:
#
#   1. FINE-GRAINED DUTY CYCLE. tick() is called every optimiser step (a few ms),
#      accumulating a sleep "debt" that is paid once it exceeds ~20 ms — fine
#      enough that the card never runs flat out long enough to reach peak draw,
#      coarse enough to avoid Windows' ~15 ms sleep granularity dominating.
#   2. CLOSED LOOP ON POWER. Power is polled every couple of seconds and the duty
#      cycle is adjusted multiplicatively to hold draw under the cap — cut hard
#      when over, relaxed slowly when comfortably under.
#   3. HARD PAUSE with real cool-down if power or temperature still exceeds the
#      ceiling, which is what protects against the sustained-load failure.
# =============================================================================
import os
import subprocess
import time

_QUERY = ('nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu '
          '--format=csv,noheader,nounits')


def read_gpu():
    """(utilisation %, power W, temp C); (None, None, None) if unreadable."""
    try:
        out = subprocess.check_output(_QUERY, shell=True, stderr=subprocess.DEVNULL,
                                      timeout=10).decode().strip().splitlines()[0]
        u, p, t = (x.strip() for x in out.split(','))
        return float(u), float(p), float(t)
    except Exception:
        return None, None, None


class GpuGuard:
    def __init__(self, target_util=80.0, max_power_w=320.0, max_temp_c=75.0,
                 poll_every_s=2.0, min_duty=0.25, verbose=True,
                 rest_every_s=600.0, rest_for_s=45.0):
        self.duty = max(0.05, min(target_util / 100.0, 1.0))
        self.max_duty = self.duty
        self.min_duty = min_duty
        self.max_power_w = max_power_w
        self.max_temp_c = max_temp_c
        self.poll_every_s = poll_every_s
        self.verbose = verbose
        # 12 Sep: the box hard-powered-off after 227 min WHILE capped at 300 W and
        # 69 C. Peak power was therefore not the trigger -- sustained multi-hour
        # load is. So force a real idle period at a fixed wall-clock cadence,
        # independent of power and temperature readings.
        self.rest_every_s = rest_every_s
        self.rest_for_s = rest_for_s
        self._last_rest = time.time()
        self.rests = 0
        self._last = time.time()
        self._last_poll = 0.0
        self._debt = 0.0
        self.slept_s = 0.0
        self.paused_s = 0.0
        self.pauses = 0
        self.peak_power = 0.0
        self.peak_temp = 0.0
        self.peak_util = 0.0

    def tick(self):
        """Call every optimiser step."""
        now = time.time()
        busy = now - self._last
        if 0 < busy < 60:
            self._debt += busy * (1.0 - self.duty) / self.duty
        if self._debt >= 0.02:                      # pay debt in ~20ms+ chunks
            nap = min(self._debt, 1.0)
            time.sleep(nap)
            self.slept_s += nap
            self._debt -= nap

        if now - self._last_rest >= self.rest_every_s:
            self._last_rest = now
            self.rests += 1
            if self.verbose:
                print('  [gpu_guard] scheduled %.0fs cool-down (%.0f min elapsed)'
                      % (self.rest_for_s, (now - (self._last_rest - self.rest_every_s)) / 60),
                      flush=True)
            time.sleep(self.rest_for_s)
            self.paused_s += self.rest_for_s
            self._last_rest = time.time()

        if now - self._last_poll >= self.poll_every_s:
            self._last_poll = now
            u, p, t = read_gpu()
            if p is not None:
                self.peak_power = max(self.peak_power, p)
                self.peak_temp = max(self.peak_temp, t)
                self.peak_util = max(self.peak_util, u)
                # closed loop: cut duty hard when over budget, relax gently under
                if p > self.max_power_w or t > self.max_temp_c:
                    self.duty = max(self.min_duty, self.duty * 0.75)
                elif p < self.max_power_w * 0.75 and t < self.max_temp_c - 5:
                    self.duty = min(self.max_duty, self.duty * 1.05)
                # last resort: real cool-down
                if p > self.max_power_w * 1.25 or t > self.max_temp_c + 5:
                    self.pauses += 1
                    t0 = time.time()
                    if self.verbose:
                        print('  [gpu_guard] %.0fW %.0fC — cooling down' % (p, t), flush=True)
                    while time.time() - t0 < 120:
                        time.sleep(2.0)
                        _, p2, t2 = read_gpu()
                        if p2 is None or (p2 <= self.max_power_w * 0.6
                                          and t2 <= self.max_temp_c - 8):
                            break
                    self.paused_s += time.time() - t0
        self._last = time.time()

    def summary(self):
        return ('gpu_guard: duty %.0f%%  slept %.0fs  paused %.0fs (%d+%d rests)  '
                'peak %.0f%% %.0fW %.0fC'
                % (self.duty * 100, self.slept_s, self.paused_s, self.pauses,
                   self.rests, self.peak_util, self.peak_power, self.peak_temp))


def power_limit_w():
    """Driver-enforced cap, or None if unreadable."""
    try:
        out = subprocess.check_output(
            'nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits',
            shell=True, stderr=subprocess.DEVNULL, timeout=10).decode()
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


# The software guard is a FALLBACK. When a driver cap is in place the hardware
# already makes peaks impossible, so throttling hard would only waste time; when
# it is absent (e.g. after a reboot -- persistence mode is unsupported on
# Windows, so `nvidia-smi -pl` does NOT survive a restart) we clamp down hard
# and say so loudly.
_LIMIT = power_limit_w()
_CAPPED = _LIMIT is not None and _LIMIT <= 350.0

if _CAPPED:
    # With a driver cap the card SITS AT its cap under load by design, so a
    # software power ceiling near the cap fires continuously and throttles to the
    # floor -- measured: duty collapsed to 25%, sleeping 1114s of a 1560s trial.
    # Hardware already makes overdraw impossible, so disable the power loop here
    # (ceiling above what the cap permits) and police temperature only.
    _defaults = dict(target_util=70.0, max_power_w=_LIMIT * 1.15, max_temp_c=75.0)
else:
    print('=' * 72)
    print('WARNING: no driver power cap (limit = %s W).' % _LIMIT)
    print('  This machine hard-powers-off under sustained full GPU load.')
    print('  Run in an ADMIN shell:   nvidia-smi -pl 300')
    print('  A reboot resets it, so it must be re-applied after every restart.')
    print('  Falling back to aggressive software throttling (slower).')
    print('=' * 72, flush=True)
    _defaults = dict(target_util=60.0, max_power_w=300.0, max_temp_c=72.0)

# Override without editing code: GPU_TARGET_UTIL / GPU_MAX_POWER / GPU_MAX_TEMP
GUARD = GpuGuard(
    target_util=float(os.environ.get('GPU_TARGET_UTIL', _defaults['target_util'])),
    max_power_w=float(os.environ.get('GPU_MAX_POWER', _defaults['max_power_w'])),
    max_temp_c=float(os.environ.get('GPU_MAX_TEMP', _defaults['max_temp_c'])),
    rest_every_s=float(os.environ.get('GPU_REST_EVERY', 600)),
    rest_for_s=float(os.environ.get('GPU_REST_FOR', 45)),
)


if __name__ == '__main__':
    u, p, t = read_gpu()
    print('utilisation %.0f%%   power %.0f W   temp %.0f C' % (u, p, t))
    print('limit:', subprocess.check_output(
        'nvidia-smi --query-gpu=power.limit,power.max_limit --format=csv,noheader',
        shell=True).decode().strip())
