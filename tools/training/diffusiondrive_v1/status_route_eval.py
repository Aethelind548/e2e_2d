"""Public CLI for the current Route-split Status evaluator.

The implementation and argument contract live in ``status_route_eval_core``.
Historical route-only evaluation is preserved in the pre-refactor source
baseline; its parameters are not part of this CLI.
"""


def main():
    """Delegate without loading models or datasets at import time."""
    from tools.training.diffusiondrive_v1.status_route_eval_core import main as core_main
    return core_main()


if __name__ == '__main__':
    main()
