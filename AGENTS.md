# AGENTS.md

## Cursor Cloud specific instructions

### Overview

LiteRT-LM is a C++ inference framework for on-device LLMs with Python/Kotlin/C bindings. The primary build system is **Bazel 7.6.1** (managed via Bazelisk). See `docs/getting-started/build-and-run.md` for the full build-from-source guide.

### Build

- **Build command**: `bazel build --config=linux_x86_64 //runtime/engine:litert_lm_main`
- **Build all runtime targets**: `bazel build --config=linux_x86_64 //runtime/...`
- Always include `--config=linux_x86_64` on Linux x86_64 to pick up the correct platform options (AVX2, etc.).
- The build requires `clang`/`clang++` (enforced by `.bazelrc`). clang-20 is installed at `/usr/bin/clang-20`.
- `libstdc++-14-dev` must be installed for clang to find C++ standard library headers (clang auto-selects GCC 14's headers).

### Test

- **Run all runtime tests**: `bazel test --config=linux_x86_64 --test_output=errors //runtime/...`
- E2E sanity tests (`tools/test/`) require a `.litertlm` model file and a built binary; run with: `pytest tools/test/ --model-path=<path> --build-system=bazel`
- E2E tests require a HuggingFace token (`HF_TOKEN` env var) to download gated models.

### Lint

No lint tooling is configured in this repository (no `.clang-format`, no CI lint step, no buildifier).

### Python CLI / SDK

- `pip install -r requirements.txt` installs CLI dependencies (typer, rich, huggingface-hub, etc.).
- `pip install litert-lm` installs the prebuilt Python SDK and CLI (`litert-lm` command).
- Ensure `~/.local/bin` is on `PATH` for pip-installed scripts.

### Known issues on main (as of April 2026)

- `runtime/conversation/conversation.cc` has invalid C++ at line 336 (`const auto& S = ConversationConfig::MemoryStrategy;` — cannot alias an enum class type). This blocks building `litert_lm_main` and targets depending on `//runtime/conversation:conversation`. All other runtime targets compile and test successfully.
- `runtime/util/model_type_utils_test` has a pre-existing failure in `GetDefaultJinjaPromptTemplateGemma4`.
