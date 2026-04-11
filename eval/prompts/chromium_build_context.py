"""Long-context prompts for the Chromium build agentic evaluation.

These prompts simulate realistic agentic software engineering tasks around
compiling Chromium, at varying context lengths (short, medium, long).
"""

SYSTEM_PROMPT = (
    "You are an expert build engineer assistant. You help developers compile "
    "large C++ projects from source. You have access to a Linux terminal and "
    "can run shell commands, edit files, and diagnose build errors. "
    "Respond with precise, actionable steps. When diagnosing errors, identify "
    "the root cause before suggesting fixes."
)

# ---------------------------------------------------------------------------
# Harness 1: Agentic Step Planning
# ---------------------------------------------------------------------------

PLANNING_PROMPT_SHORT = """\
I need to compile Chromium from source on a fresh Ubuntu 24.04 machine with
32 cores, 128GB RAM, and 200GB free disk. Give me the complete ordered list
of steps from a blank machine to a working chrome binary. Include:
- All prerequisite packages
- depot_tools setup
- gclient sync
- gn gen with appropriate args
- autoninja invocation
- Any post-build verification

Number each step and note dependencies between steps.
"""

PLANNING_PROMPT_MEDIUM = """\
I need to compile Chromium from source on a fresh Ubuntu 24.04 machine.

Machine specs:
- CPU: AMD EPYC 7R13 (32 vCPUs)
- RAM: 128 GiB
- Disk: 1.2TB EBS gp3, 200GB free
- OS: Ubuntu 24.04 LTS, kernel 6.17
- Network: 10 Gbps
- No GPU (CPU-only build)

I want a component build for development (not release), with the following:
- symbol_level=1 for faster builds
- is_component_build=true
- use_remoteexec=false (no RBE)
- blink_symbol_level=0
- No NaCl, no PDF compositor
- ccache enabled if possible

Additionally, I need to handle:
1. Setting up a swap file (disk is large enough)
2. Configuring ccache with a 50GB cache
3. Monitoring memory during the build (ninja can OOM with -j32 on 128GB)
4. Collecting build timing data

Provide a complete, ordered, dependency-aware plan. For each step, indicate:
- The exact command(s)
- Expected duration
- Disk space consumed
- Whether it can be parallelized with other steps
- What to check to verify the step succeeded
"""

PLANNING_PROMPT_LONG = """\
I need to set up a complete Chromium development environment on two machines:

Machine A (Build Server):
- AMD EPYC 7R13, 32 vCPUs, 128 GiB RAM
- Ubuntu 24.04 LTS, kernel 6.17
- 1.2TB EBS gp3 (25GB free - CRITICAL: must free space first!)
- No GPU

Machine B (Test Device):
- Raspberry Pi 5, 8GB RAM
- Ubuntu 24.04 arm64
- 256GB SD card
- Connected to Machine A via SSH

Goals:
1. On Machine A: compile Chromium for both x86_64 (local testing) and
   aarch64 (cross-compile for Pi)
2. Deploy the aarch64 build to Machine B
3. Run the Chromium test suite on both machines
4. Collect performance metrics (build time, binary size, test results)

Constraints:
- Machine A disk is nearly full, must clean up first
- Must not break existing builds in /home/cory/chromium-src
- Need ccache for iterative development
- Should use component build for x86_64, release build for aarch64
- Must handle potential OOM during linking (gold linker vs lld)
- Need to set up distcc or icecc between the two machines is a bonus

The following directories exist on Machine A:
/home/cory/chromium-src/ (symlink to /home/cory/cef, currently empty dir)
/home/cory/depot_tools/ (already cloned)
/home/cory/build/ (old build artifacts, 400GB, can be cleaned)

Prior failed build log (last 50 lines):
```
[4523/52891] CXX obj/third_party/blink/renderer/core/core/css_property.o
[4524/52891] CXX obj/third_party/blink/renderer/core/core/css_value.o
FAILED: obj/third_party/blink/renderer/core/core/dom_element.o
../../third_party/llvm-build/Release+Asserts/bin/clang++ -MMD -MF ...
In file included from ../../third_party/blink/renderer/core/dom/element.cc:7:
../../third_party/blink/renderer/core/dom/element.h:42:10: fatal error:
    'third_party/blink/renderer/core/dom/element_data.h' file not found
#include "third_party/blink/renderer/core/dom/element_data.h"
         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1 error generated.
[4525/52891] CXX obj/third_party/blink/renderer/core/core/css_style.o
[4526/52891] CXX obj/third_party/blink/renderer/core/core/node.o
ninja: build stopped: subcommand failed.
Build ended at 2026-04-08 14:32:15 (elapsed: 1h23m)
gn gen args:
  target_os = "linux"
  target_cpu = "x64"
  is_debug = true
  is_component_build = true
  symbol_level = 2
  blink_symbol_level = 2
  use_remoteexec = false
  enable_nacl = false
  is_official_build = false
```

gclient status:
```
[chromium/src] Status: synced to r128.0.6613.0
[chromium/src/third_party/blink] Status: DIRTY - uncommitted changes
  M third_party/blink/renderer/core/dom/element.h
  M third_party/blink/renderer/core/dom/element.cc
```

System state:
```
$ df -h /
Filesystem  Size  Used  Avail  Use%  Mounted on
/dev/root   1.2T  1.2T  25G    98%   /
$ free -h
              total   used   free   shared  buff/cache  available
Mem:          123Gi   8.5Gi  18Gi   3.3Mi   97Gi        115Gi
```

Provide a comprehensive, prioritized recovery and build plan that:
1. Diagnoses and fixes the disk space issue
2. Fixes the failed build
3. Sets up both x86_64 and aarch64 build configurations
4. Deploys to Machine B
5. Runs validation tests

For each step, specify: command, expected output, estimated time,
disk impact, and rollback strategy if it fails.
"""

PLANNING_REFERENCE_STEPS = [
    "Install prerequisites (git, python3, lsb-release, sudo, curl, etc.)",
    "Clone/update depot_tools and add to PATH",
    "Configure gclient for chromium checkout",
    "Run gclient sync to fetch source (~40GB)",
    "Install build dependencies via install-build-deps.sh",
    "Configure build with gn gen out/Default",
    "Build with autoninja -C out/Default chrome",
    "Verify build by running out/Default/chrome --version",
]

# ---------------------------------------------------------------------------
# Harness 2: Error Diagnosis
# ---------------------------------------------------------------------------

ERROR_DIAGNOSIS_CONTEXT = """\
I'm building Chromium on Ubuntu 24.04 and hit multiple errors. Here is the
full context. Please analyze everything and tell me the root cause and fix.

== System Info ==
$ uname -a
Linux ip-172-31-27-231 6.17.0-1009-aws x86_64 GNU/Linux
$ cat /etc/os-release | head -4
NAME="Ubuntu"
VERSION="24.04 LTS (Noble Numbat)"
ID=ubuntu
VERSION_ID="24.04"
$ gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
$ python3 --version
Python 3.12.3
$ df -h /
Filesystem  Size  Used  Avail  Use%  Mounted on
/dev/root   1.2T  1.1T  100G   92%   /

== GN Args (out/Debug/args.gn) ==
target_os = "linux"
target_cpu = "x64"
is_debug = true
is_component_build = true
symbol_level = 2
blink_symbol_level = 2
use_remoteexec = false
enable_nacl = false
use_sysroot = true
proprietary_codecs = false
ffmpeg_branding = "Chromium"

== Build Command ==
$ autoninja -C out/Debug chrome 2>&1 | tail -100

[12045/52891] CXX obj/base/base/message_loop.o
[12046/52891] CXX obj/base/base/run_loop.o
[12047/52891] CXX obj/base/base/task_runner.o
...
[31002/52891] LINK ./libcontent.so
FAILED: libcontent.so libcontent.so.TOC
python3 "../../build/toolchain/gcc_solink_wrapper.py" --readelf="readelf" ...
/usr/bin/ld: final link requires too much memory (12582912 bytes); recompile
    with -fno-PIC or use -no-keep-memory
collect2: error: ld returned 1 exit status

[31003/52891] CXX obj/content/browser/content/navigation_controller.o
[31004/52891] CXX obj/content/browser/content/render_frame_host.o
...
[31050/52891] LINK ./libchrome.so
FAILED: libchrome.so libchrome.so.TOC
/usr/bin/ld: cannot find -lstdc++: No such file or directory
/usr/bin/ld: cannot find -lgcc_s: No such file or directory
collect2: error: ld returned 1 exit status

ninja: build stopped: subcommand failed.
Errors: 2 | Warnings: 347 | Elapsed: 2h14m

== Linker Details ==
$ ld --version
GNU ld (GNU Binutils for Ubuntu) 2.42
$ which lld
/usr/bin/lld-18
$ file out/Debug/obj/base/base/message_loop.o
out/Debug/obj/base/base/message_loop.o: ELF 64-bit LSB relocatable, x86-64

== Memory at failure ==
$ dmesg | tail -20
[14523.456] Out of memory: Killed process 89234 (ld)
[14523.457] oom_kill_process: 1 (ld), adj 0, rss 11234567
[14523.458] Memory cgroup out of memory.
[14523.459] Kill process 89234 (ld) score 950 or sacrifice child

$ free -h
              total   used   free   shared  buff/cache  available
Mem:          123Gi   120Gi  1.2Gi  3.3Mi   2.5Gi       3.0Gi

== Additional Files ==
$ cat out/Debug/toolchain.ninja | grep -A2 "rule solink"
rule solink
  command = python3 "../../build/toolchain/gcc_solink_wrapper.py" ...
  description = SOLINK $out

$ cat build/config/compiler/BUILD.gn | grep -B2 -A5 "use_lld"
  if (use_lld) {
    ldflags += [ "-fuse-ld=lld" ]
  } else {
    # Fall back to gold or bfd
    ldflags += [ "-fuse-ld=gold" ]
  }
"""

ERROR_DIAGNOSIS_REFERENCE = {
    "root_causes": [
        "GNU ld (bfd) running out of memory during linking of large shared libraries",
        "use_lld not enabled in GN args, falling back to memory-hungry GNU ld",
        "symbol_level=2 producing excessively large object files",
        "Missing libstdc++ dev package or broken sysroot",
    ],
    "fixes": [
        "Add use_lld=true to args.gn to use LLVM lld linker (much lower memory)",
        "Reduce symbol_level from 2 to 1 or 0",
        "Install libstdc++-13-dev if missing",
        "Add -no-keep-memory to ldflags as workaround if staying with GNU ld",
        "Reduce ninja parallelism (-j8) to limit concurrent linker processes",
    ],
    "key_evidence": [
        "dmesg OOM kill of ld process",
        "ld error 'final link requires too much memory'",
        "lld-18 is installed but not being used (use_lld not in args.gn)",
        "symbol_level=2 with is_component_build=true causes many large .so links",
    ],
}

# ---------------------------------------------------------------------------
# Harness 3: Multi-Turn Tool Use
# ---------------------------------------------------------------------------

TOOL_USE_SCENARIO = {
    "description": "Fix a Chromium build failure and complete the build",
    "initial_state": {
        "cwd": "/home/user/chromium/src",
        "depot_tools_in_path": True,
        "gn_args_file": "out/Default/args.gn",
        "build_status": "failed",
        "error": "undefined reference to `blink::Element::setAttribute'",
        "disk_free_gb": 50,
        "ram_total_gb": 64,
        "ram_used_gb": 12,
    },
    "available_tools": [
        {"name": "run_command", "description": "Execute a shell command"},
        {"name": "read_file", "description": "Read contents of a file"},
        {"name": "write_file", "description": "Write/overwrite a file"},
        {"name": "search_code", "description": "Search source code (like grep)"},
    ],
    "turns": [
        {
            "turn": 1,
            "user_message": (
                "The Chromium build failed with 'undefined reference to "
                "`blink::Element::setAttribute''. The build was using "
                "out/Default with is_component_build=true. What should I do?"
            ),
            "expected_actions": [
                "search_code for setAttribute in blink/renderer/core/dom/",
                "read_file out/Default/args.gn to check build config",
                "run_command to check if the symbol exists in object files",
            ],
            "reference_response_contains": [
                "setAttribute",
                "linker",
                "component build",
                "symbol visibility",
            ],
        },
        {
            "turn": 2,
            "user_message": (
                "I ran `nm -C obj/third_party/blink/renderer/core/core/"
                "element.o | grep setAttribute` and got:\n"
                "0000000000001234 T blink::Element::setAttribute(...)\n"
                "The symbol exists. But the link step for libcontent.so fails."
            ),
            "expected_actions": [
                "check GN deps between content and blink/core targets",
                "run gn desc out/Default //content/browser:browser deps",
                "look for missing COMPONENT_EXPORT macros",
            ],
            "reference_response_contains": [
                "deps",
                "visibility",
                "COMPONENT_EXPORT",
                "gn desc",
            ],
        },
        {
            "turn": 3,
            "user_message": (
                "Found it - the blink::Element::setAttribute was recently "
                "moved but the BUILD.gn deps weren't updated. I added the "
                "dep. Now `autoninja -C out/Default chrome` is running. "
                "It's at [42000/52891] and memory is at 95%. Should I worry?"
            ),
            "expected_actions": [
                "monitor memory usage",
                "suggest reducing ninja parallelism if memory is critical",
                "check if swap is configured",
            ],
            "reference_response_contains": [
                "memory",
                "ninja",
                "-j",
                "swap",
                "OOM",
            ],
        },
    ],
    "max_turns": 10,
    "success_criteria": "Build completes successfully or correct fix identified",
}

# Token count approximations for each prompt tier
PROMPT_TOKEN_ESTIMATES = {
    "planning_short": 150,
    "planning_medium": 450,
    "planning_long": 2200,
    "error_diagnosis": 3500,
    "tool_use_per_turn": 200,
}
