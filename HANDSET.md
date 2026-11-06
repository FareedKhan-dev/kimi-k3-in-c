# Run it: Codespace, then the phone

This fork is [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
plus `native/`, the arm64 Android layer. Apache-2.0, same as the engine.
Weights stay Moonshot’s. A Codespace and a phone can both prove the engine.
Neither one holds the 1.67 TB checkpoint.

What each machine is for:

| Machine | What it proves | What it cannot do |
|---|---|---|
| Codespace | The engine matches its fixtures. A Range GET matches `pread`. | Store or generate from the real checkpoint. The disk is too small. |
| Phone, Termux, foreground | The same two checks, on the CPU you will actually use. | An APK. The app-sandbox cgroup is smaller than the 2.57 GiB floor. |
| A disk with 1.7 TB free | The download, the pack, and the first real token. | This is not the Codespace and not the phone’s internal flash. |

## 1. Codespace

On the repo page: **Code → Codespaces → Create codespace on main**.

Or open [a new codespace on this repo](https://codespaces.new/cyeezy08/kimi-k3-in-c).

The container installs a compiler, then runs, in order:

```sh
make -C native test     # the handset plan
make -C native stream   # three Range GETs, byte-identical to the file
make -j                 # the engine
make test               # weightless gates. No checkpoint, no network.
```

Leave it. The last line you want is:

```
ALL WEIGHTLESS TESTS PASSED
```

and, before that:

```
k3_android: all checks passed
k3_stream: range pread matched
```

If you open a terminal before that finishes, run the same four commands yourself.
Nothing here downloads the model. Do not point `download-model.sh` at the
Codespace disk.

## 2. Phone

Install [Termux from F-Droid](https://f-droid.org/packages/com.termux/), not the
Play Store build. Open it, leave the screen on, and run:

```sh
pkg update
pkg install git clang make python libomp
git clone https://github.com/cyeezy08/kimi-k3-in-c
cd kimi-k3-in-c
make -C native test
make -C native stream
make -j
make test
```

Same four results as the Codespace. `make test` is the engine. It does not
need the checkpoint and it does not produce “Paris”.

## 3. The checkpoint, on a machine that has the room

Not the Codespace. Not the phone. A desktop, a server, or a rented disk.

```sh
git clone https://github.com/cyeezy08/kimi-k3-in-c
cd kimi-k3-in-c
# token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...
./scripts/download-model.sh ~/k3model
./scripts/pack-trunk.sh ~/k3model ~/k3trunk
```

That is the official `moonshotai/Kimi-K3` checkpoint, 1.56 TB, plus the 109 GB
packed trunk. A partial download still runs and emits the wrong token. Wait
until both scripts finish.

Skip Google Drive. The Drive API allows 1 TB per project per day, and one
token already moves about 0.21 TB.

## 4. Serve the bytes with Range

On that same machine, any HTTP server that answers `206` is enough. The one
in this repo binds localhost only, which is what the test uses. For the phone
you want a server on the LAN or a VPS, with TLS ended there, not on the phone.

nginx in front of the two directories is the plain version. The phone then
sees one host. `k3_stream_pread` sends:

```
GET /k3/trunk.bin HTTP/1.1
Range: bytes=START-END
```

and refuses anything except `206` with a `Content-Length` equal to the span.
A `200` would be the whole object.

A TypeScript WebSocket proxy is optional, and only after this works. Put it
next to the files. On 8 MiB slices it removes the per-range round trip and
saves minutes. On 1 MiB Telegram parts it is the difference between two hours
and about seventy minutes. It does not shrink the checkpoint.

## 5. One real token

`bin/k3` in this fork still reads files with `pread`. The stream reader is
proven beside it, not spliced into the forward pass. Two ways to get a token:

**USB disk.** Copy `k3model` and `k3trunk` to a 2 TB USB-C SSD, plug it into
the phone, and in Termux:

```sh
termux-setup-storage
./bin/k3 /storage/<usb>/k3model \
    --trunk /storage/<usb>/k3trunk \
    --preset ultra --ultra-low-memory \
    --threads <big-cores> \
    --tok /storage/<usb>/k3model \
    --prompt "The capital of France is" --gen 1 --incremental
```

Stock Android mounts that disk as exFAT, so `O_DIRECT` fails. That is expected.
After each buffered read, call `k3_android_drop_cache` when the plan says so.
Build with the Termux clang. Do not pass `-ffast-math`.

**Object store.** Replace the `pread` calls in `src/io/k3_st.c` and
`src/io/k3_trunk.c` with `k3_stream_pread`, rebuild, and point the host at the
gateway from step 4. Offsets do not change. If the bytes match, the first
token is `17374`, the text ` Paris`. If they do not, stop. The engine will
still print a token, and it will be the wrong one.

Call `k3_android_plan_from` before you allocate. The laptop preset peaks at
8.24 GB and the plan refuses it. Ultra stays on: the embedding and `lm_head`
are streamed, not resident.

Leave Termux in the foreground. An APK is the wrong host.
