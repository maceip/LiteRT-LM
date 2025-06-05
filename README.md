# LiteRT LM
GitHub repository for Google's open-source high-performance runtime for
on-device LLM.

## How to build and run

`bazel` is used as the build system. Install `bazel` via
[Bazelisk](https://github.com/bazelbuild/bazelisk) or follow the
[instructions](https://bazel.build/install) per platform.

### Linux

`clang` is used to build LiteRT LM on linux. Build `litert_lm_main`, a CLI
executable and run models on CPU.

```
bazel build //runtime/engine:litert_lm_main

bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=<model .litertlm or .task file>
```

### Android

Install NDK r28b or newer from https://developer.android.com/ndk/downloads.
To run on CPU, only `litert_lm_main` is required.

```
export ANDROID_NDK_HOME=<ndk dir>

bazel build --config=android_arm64 //runtime/engine:litert_lm_main

adb push bazel-bin/runtime/engine/litert_lm_main /data/local/tmp

adb shell /data/local/tmp/litert_lm_main \
    --backend=cpu \
    --model_path=<model .litertlm or .task file>
```

To run on GPU, `libLiteRtGpuAccelerator.so` and `libLiteRtTopKOpenClSampler.so`
are required.

```
adb push prebuilt/android_arm64/*.so /data/local/tmp

adb shell LD_LIBRARY_PATH=/data/local/tmp \
    /data/local/tmp/litert_lm_main \
    --backend=gpu \
    --model_path=<model .litertlm or .task file>
```

To run on Qualcomm NPU, `libLiteRtDispatch_Qualcomm.so` in the maven package
from [TBD]() and Qualcomm's runtime shared libraries downloaded from [TBD]()
are required. Model file should be compiled with LiteRt compiler with Qualcomm's
plugin.

```
adb push libLiteRtDispatch_Qualcomm.so /data/local/tmp
adb push <qualcomm runtime shlibs> /data/local/tmp

adb shell LD_LIBRARY_PATH=/data/local/tmp \
    /data/local/tmp/litert_lm_main \
    --backend=qnn \
    --model_path=<model .tflite file>
```

### MacOS

Xcode command line tools include clang. Run `xcode-select --install` if not
installed before.

```
bazel build //runtime/engine:litert_lm_main

bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=<model .litertlm or .task file>
```

If bazel can't figure out the right version of MacOS SDK, append
`--macos_sdk_version=<sdk version>` in `bazel build`.
