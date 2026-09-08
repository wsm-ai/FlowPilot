class PassthroughSideEffectExecutor:
    """Explicit test double for tests focused on non-ledger behavior."""

    async def execute(self, *, operation, **kwargs):
        return await operation()


PASSTHROUGH_SIDE_EFFECT_EXECUTOR = PassthroughSideEffectExecutor()
