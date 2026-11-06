# NDK toolchain file fragment. Usage, from an Android NDK r26+ install:
#
#   cmake -S . -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-28 \
#     -DANDROID_STL=c++_static
#
# Link k3_android.c into the upstream engine and keep -fopenmp.
# arm64-v8a implies NEON. Do not pass -mfpu. Do not enable fast-math:
# the upstream kernels are bit-stable only because they refuse it.
#
# Termux (no NDK required), on the phone:
#
#   pkg install clang make libomp
#   cd native && make test
#   # then build upstream kimi-k3-in-c with the same compiler
#
# Copy k3_android.c into that tree and call k3_android_plan_from() before
# the preset allocator. After every buffered trunk or expert pread, call
# k3_android_drop_cache(fd, off, len) when the plan's drop_cache flag is set.
# Pin with sched_setaffinity to the indexes from k3_android_pick_big()
# on /sys/devices/system/cpu/cpuN/cpu_capacity. A core counts as big at
# 70% of the maximum, so a 1+5+2 mid cluster is kept and the littles are not.
