# Host-side checks. The handset binary itself is built on device
# (Termux clang, or the NDK). These targets prove the plan and that a
# Range GET can stand in for pread.
CC ?= gcc
CFLAGS ?= -std=c99 -Wall -Wextra -Werror -O2 -D_GNU_SOURCE

.PHONY: test stream oracle clean

test: k3_android.c test_android.c k3_android.h
	$(CC) $(CFLAGS) -o test_android k3_android.c test_android.c -lm
	./test_android

stream: k3_stream.c test_stream.c k3_stream.h range_server.py
	$(CC) $(CFLAGS) -o test_stream k3_stream.c test_stream.c
	mkdir -p /tmp/k3http
	python3 -c 'open("/tmp/k3http/blob.bin","wb").write(bytes([(i*7+3) & 255 for i in range(200000)]))'
	python3 range_server.py /tmp/k3http 18765 >/tmp/k3http.log 2>&1 & echo $$! > /tmp/k3http.pid
	sleep 0.3
	./test_stream 127.0.0.1 18765 /blob.bin; ec=$$?; kill $$(cat /tmp/k3http.pid) || true; exit $$ec

oracle: k3_mxfp4_ref.c
	$(CC) $(CFLAGS) -o mxfp4_ref k3_mxfp4_ref.c
	./mxfp4_ref

clean:
	rm -f test_android test_stream mxfp4_ref
