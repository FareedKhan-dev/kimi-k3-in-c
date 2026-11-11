# Contract for the K3 tokenizer loader against the synthetic vocabularies in
# tests/fixtures/tok (tools/make_tok_fixture.py). Same expectations as the
# "tokenizer loader" block in `make test`: the good pair encodes through rank BPE
# and the byte-level keys, and every corrupt pair is refused for its own reason.
if(NOT DEFINED TEST_TOK OR NOT DEFINED FIXTURES)
  message(FATAL_ERROR "TEST_TOK and FIXTURES were not provided")
endif()

execute_process(COMMAND "${TEST_TOK}" "${FIXTURES}/tok/good" encode "hello hello"
                OUTPUT_VARIABLE out ERROR_QUIET RESULT_VARIABLE rc)
string(STRIP "${out}" out)
if(NOT rc EQUAL 0 OR NOT out STREQUAL "259,258,32,259,258")
  message(FATAL_ERROR "good vocabulary encoded 'hello hello' to [${out}] (rc ${rc}),"
                      " expected [259,258,32,259,258]")
endif()

foreach(case
    "dup_rank|appears twice"
    "missing_rank|rank 100 is missing"
    "junk_rank|bad rank field"
    "dup_bytes|encode the same bytes"
    "added_collides|already taken by a rank"
    "truncated|looks truncated")
  string(REPLACE "|" ";" parts "${case}")
  list(GET parts 0 dir)
  list(GET parts 1 why)
  execute_process(COMMAND "${TEST_TOK}" "${FIXTURES}/tok/${dir}" encode x
                  OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(rc EQUAL 0)
    message(FATAL_ERROR "${dir}: the corrupt vocabulary loaded")
  endif()
  if(NOT err MATCHES "${why}")
    message(FATAL_ERROR "${dir}: refused, but not because of '${why}': ${err}")
  endif()
endforeach()
