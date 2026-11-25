"""Exact break-even arithmetic. No torch import and no NPU work."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from fractions import Fraction
import math

NUMERIC_RULE = 'durations rounded to integer ns using Decimal(str(ms))*1e6, ROUND_HALF_EVEN; sums/means/crossings use exact Fraction; no epsilon'


def rational(value):
    if value is None:
        raise LookupError('missing value')
    if isinstance(value, Fraction):
        return value
    d = Decimal(str(value))
    if not d.is_finite():
        raise ValueError('nonfinite value')
    return Fraction(d)


def duration_ms(value, positive=False):
    f = rational(value)
    if f < 0 or (positive and f == 0):
        raise ValueError('negative cost or nonpositive call duration')
    ns = int((Decimal(str(float(f)))*Decimal(1000000)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
    if positive and ns <= 0:
        raise ValueError('call duration not positive at ns resolution')
    return Fraction(ns, 1000000)


def solve(A_e, A_c, t_e, t_c, batch=1, warmups=5):
    out = dict(status=None, N_win=None, K_win=None, effective_images=None,
        forward_count=None, including_preparation_images=None, real_crossing_N=None,
        delta_A_ms=None, delta_t_ms=None, last_advantage_N=None, tie_N=None,
        first_eager_strict_N=None, numeric_rule=NUMERIC_RULE)
    try:
        ae, ac, te, tc = [rational(v) for v in [A_e, A_c, t_e, t_c]]
        if min(ae, ac) < 0 or min(te, tc) <= 0 or batch <= 0 or warmups < 0:
            raise ValueError('invalid costs/batch/warmup')
    except LookupError as e:
        out.update(status='insufficient_data', reason=str(e)); return out
    except (ValueError, InvalidOperation, OverflowError) as e:
        out.update(status='invalid', reason=str(e)); return out
    da, dt = ac-ae, te-tc
    out.update(A_e_ms=float(ae), A_c_ms=float(ac), t_e_ms=float(te), t_c_ms=float(tc),
               delta_A_ms=float(da), delta_t_ms=float(dt),
               exact_delta_A=str(da), exact_delta_t=str(dt))
    cross = da/dt if dt else None
    if cross is not None:
        out.update(real_crossing_N=float(cross), exact_real_crossing_N=str(cross))
    if dt > 0:
        n = max(1, math.floor(cross)+1)
        out.update(status='already_beneficial' if n == 1 else 'finite_break_even', N_win=n)
    elif dt == 0:
        out['status'] = 'always_beneficial' if da < 0 else 'always_tied' if da == 0 else 'no_break_even'
        if da < 0:
            out['N_win'] = 1
    elif da-dt < 0:
        last = math.ceil(cross)-1
        out.update(status='initial_advantage_only', N_win=1, last_advantage_N=last,
                   tie_N=int(cross) if cross.denominator == 1 else None,
                   first_eager_strict_N=math.floor(cross)+1,
                   beneficial_integer_interval=[1, last])
    else:
        out['status'] = 'no_break_even'
    n = out['N_win']
    if n is not None:
        out.update(K_win=n+1+warmups, forward_count=n+1+warmups,
            effective_images=batch*n, including_preparation_images=batch*(n+1+warmups),
            T_e_at_N_ms=float(ae+n*te), T_c_at_N_ms=float(ac+n*tc),
            difference_at_N_ms=float(da-n*dt), strict_at_N=da-n*dt < 0,
            predecessor_N=n-1 if n > 1 else None,
            difference_at_predecessor_ms=float(da-(n-1)*dt) if n > 1 else None)
        assert out['strict_at_N']
        if dt > 0 and n > 1:
            assert da-(n-1)*dt >= 0
    return out


def costs(result, warmups, repeats, work_phase='repeat'):
    """Require all requested measurements; never fill missing costs with zero."""
    try:
        if result.get('status') != 'ok' or not result.get('output_valid'):
            raise LookupError('worker/output not normally completed')
        s = duration_ms(result.get('runner_setup_ms'))
        values = {}
        for phase, count in [('first', 1), ('warmup', warmups), (work_phase, repeats)]:
            rows = [x for x in result['timings'] if x['phase'] == phase]
            if len(rows) != count or sorted(x['iteration'] for x in rows) != list(range(1, count+1)):
                raise LookupError('missing/duplicate requested '+phase+' rows')
            if any(x['status'] != 'ok' or not x['timing_valid'] for x in rows):
                raise LookupError('invalid/failed '+phase+' rows')
            values[phase] = [duration_ms(x['elapsed_ms'], positive=True) for x in sorted(rows,key=lambda x:x['iteration'])]
        f = values['first'][0]
        w = sum(values['warmup'], Fraction())
        times = values[work_phase]
        return dict(status='valid', S=s, F=f, W=w, A=s+f+w,
                    t=sum(times, Fraction())/len(times), initial=values['first']+values['warmup'],
                    work=times, warmup_count=warmups)
    except LookupError as e:
        return dict(status='insufficient_data', reason=str(e))
    except (ValueError, InvalidOperation, OverflowError, KeyError) as e:
        return dict(status='invalid', reason=str(e))


def T(c, n):
    if n < 1:
        raise ValueError('N must be >=1')
    return c['A']+n*c['t']


def G(c, k):
    if k < 1:
        raise ValueError('K must be >=1')
    initial = c['initial']
    if k <= len(initial):
        return c['S']+sum(initial[:k], Fraction())
    return c['A']+(k-len(initial))*c['t']
