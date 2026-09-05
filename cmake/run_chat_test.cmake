# Runs the XTML chat-template check (test_chat) when the released vocabulary is available, and prints a
# line ctest recognises as a skip when it is not.
#
# tiktoken.model has 163,584 entries and ships with the checkpoint, not with this
# repository, so this gate cannot run on a clean checkout. It is registered anyway: a
# check that vanishes from the list when it cannot run reads as "one fewer test to care
# about", and ctest's count silently stops matching what `make test` runs.
#
# K3_TOK_FILES is read here, at TEST time rather than configure time, so
# `K3_TOK_FILES=/path/to/k3model ctest` works without reconfiguring the build.

set(TOK_FILES "$ENV{K3_TOK_FILES}")

if(TOK_FILES STREQUAL "")
  message("SKIPPED: K3_TOK_FILES is not set. The vocabulary ships with the checkpoint;"
          " set K3_TOK_FILES to a model directory to run this gate.")
  return()
endif()

if(NOT EXISTS "${TOK_FILES}/tiktoken.model")
  message("SKIPPED: no tiktoken.model in ${TOK_FILES}")
  return()
endif()

execute_process(COMMAND "${TEST_CHAT}" "${TOK_FILES}"
                RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "XTML chat-template check failed with status ${rc}")
endif()
