# LiteRT LM
GitHub repository for Google's open-source high-performance runtime for
on-device LLM.

## How to build and run

### Linux

`clang` is used to build LiteRT LM on linux. Build `litert_lm_main`, a CLI
executable and run models on CPU.

```
bazel build //runtime/engine:litert_lm_main

bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=<model task file>
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
    --model_path=<model task file>
```

To run on GPU, `libLiteRtGpuAccelerator.so` is required. Download the maven
package from [TBD]()

```
adb push libLiteRtGpuAccelerator.so /data/local/tmp

adb shell LD_LIBRARY_PATH=/data/local/tmp \
    /data/local/tmp/litert_lm_main \
    --backend=gpu \
    --model_path=<model task file>
```
