# motion_correction

Standalone `correct_motion` implementation packaged as a small C++ motion trajectory correction library with Python bindings.

## Installation Guide

### Prerequisites

Ensure you have a C++17 compatible compiler (GCC 7.0+, Clang 5.0+, or MSVC 2017+) and CMake 3.15+. On Windows, install MinGW-w64 or Visual Studio with C++ tools. On Linux, install `build-essential` and `cmake`.

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

### Build & Install

#### Standard Installation
```bash
pip install .
```

#### Development Installation
```bash
pip install -e .
```

### Verify Installation

```python
import motion_correction
print("Installation successful!")
```
You can also run `python run_test.py` for a simple test.

## Tiny3D Studio contact policy

This owned fork deliberately deviates from ARDY commit
`9d3787dfd1a24f3919f4741f5affa3f00d9bd883`: `Animation::CorrectMotion`
no longer runs `DoContactIK`. Studio's native final evaluator is the sole
owner of automatic contact enforcement. The producer preserves the original
`CorrectHipsY`, `CorrectHipsXZ`, `CorrectJointRotations`, and authored
`DoEffectorIK` calculations and their execution order.

The removed automatic pass selected contact intervals, constructed foot/toe
floor targets, applied two-bone and one-bone contact IK, smoothed those
corrections, and optionally applied double-ankle contact repair. Its private
interval, target, reach-falloff helpers and contact-target buffers are removed.
Authored effector masks, target poses, and their IK remain independent of this
removed automatic pass. The external Python and Studio WASM ABI still accept
the original contacts tensor, threshold, and double-ankle argument; callers
must pass the original contacts without zeroing or replacing them.

In the pinned source, `Utility.cpp:222-267` declares `contacts` and
`contactThreshold` parameters for `CorrectHipsY`, but its body does not read
either. Its root-height margins depend only on `fullBodyMask`. The old
`CorrectMotion` comment claiming contact-dependent height preservation does
not describe that implementation. Removing the unused internal parameters and
automatic contact-target construction does not alter that height calculation.
This observation is specific to the pinned commit, not a claim about later
upstream versions or permission to zero the externally supplied contacts.

Studio's browser path is `createCore40MotionCorrectionWasm().correct` →
`core40_correct_motion` → `Animation::CorrectMotion`. Tiny3D's
`apps/tiny3d/scripts/export_ardy_core40_browser.py` generates the C ABI wrapper
and compiles these C++ sources; its wrapper must use the updated internal
`CorrectMotion` signature and omit its former automatic target construction.
The native Python path is `_apply_source_motion_correction` →
`ardy.postprocess.correct_motion` → `motion_postprocess.correct_motion` →
`BindingsPython.cpp` → the same `Animation::CorrectMotion`. Both paths require
a rebuilt correction binary with matching source identity. Existing released
binaries retain the old automatic contact behavior until rebuilt and admitted.
