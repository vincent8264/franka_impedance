"""Capture the actuator effort sent during one environment step."""


class AppliedEffortCapture:
    """Retain the effort actually sent by the environment before auto-reset."""

    def __init__(self, env):
        self.env = env
        self.previous = None
        self.applied = None

    def __enter__(self):
        self.previous = self.env.step_recorder
        self.env.step_recorder = self
        return self

    def __exit__(self, *_exc):
        self.env.step_recorder = self.previous

    def capture_effort(self, _command, applied):
        self.applied = applied.detach().clone()

    def capture_step(self, *_args):
        pass
