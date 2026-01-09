from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DecaySchedule:
    """
    A coefficient multiplier schedule:
    - decay exponentially by `rate` per step
    - optionally floor at `min_multiplier`
    - if `times_restart > 0`, jump back to 1.0 when `min_multiplier` is hit and decay again

    If `times_restart=0` (default), this is a simple decay schedule.
    """

    rate: float = 1.0
    min_multiplier: float = 0.0
    times_restart: int = 0

    def __call__(self, step: int) -> float:
        step = max(0, int(step))
        rate = float(self.rate)
        min_multiplier = float(self.min_multiplier)
        times_restart = max(0, int(self.times_restart))

        if rate <= 0.0:
            multiplier = 1.0
        elif (
            times_restart == 0
            or rate >= 1.0
            or min_multiplier <= 0.0
            or min_multiplier >= 1.0
        ):
            multiplier = rate**step
        else:
            # Sawtooth restart without storing state:
            # decay from 1.0 until we hit `min_multiplier`, then jump back to 1.0 and repeat,
            # but only for `times_restart` restarts.
            steps_to_min = int(math.ceil(math.log(min_multiplier) / math.log(rate)))
            steps_to_min = max(0, steps_to_min)
            cycle_len = steps_to_min + 1
            last_cycle_end = (times_restart + 1) * cycle_len - 1

            if step <= last_cycle_end:
                step_in_cycle = step % cycle_len
                multiplier = (
                    min_multiplier
                    if step_in_cycle == steps_to_min
                    else rate**step_in_cycle
                )
            else:
                # After the final restart, keep decaying without any more jumps.
                # This will settle at `min_multiplier` if it is > 0 (due to the floor below).
                multiplier = rate ** (steps_to_min + (step - last_cycle_end))
        # print(f"DecaySchedule step={step} -> multiplier={multiplier}")
        return float(max(float(self.min_multiplier), float(multiplier)))

    def to_dict(self) -> dict:
        return {
            "type": "decay",
            "rate": float(self.rate),
            "min_multiplier": float(self.min_multiplier),
            "times_restart": int(self.times_restart),
        }
