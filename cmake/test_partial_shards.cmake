# Contract for tools/layers_to_shards.py, mirroring the "partial download
# contract" block in `make test`. Same fixture, same expectations: an exact
# shard set for --layers 1, the split-layer shard present and the far-future
# shard absent for --layers 2, and exit 2 for every malformed input.
if(NOT DEFINED PYTHON OR NOT DEFINED FIXTURES)
  message(FATAL_ERROR "PYTHON and FIXTURES were not provided")
endif()

function(map_layers out_var rc_var)
  execute_process(
    COMMAND "${PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/../tools/layers_to_shards.py" ${ARGN}
    OUTPUT_VARIABLE map_out
    ERROR_QUIET
    RESULT_VARIABLE map_rc)
  # execute_process keeps the trailing newline; command substitution in the
  # Makefile block strips it. Strip it here so both harnesses compare the same
  # string.
  string(REGEX REPLACE "\n$" "" map_out "${map_out}")
  set(${out_var} "${map_out}" PARENT_SCOPE)
  set(${rc_var} "${map_rc}" PARENT_SCOPE)
endfunction()

map_layers(out1 rc1 "${FIXTURES}/partial_index.json" --layers 1)
if(NOT rc1 EQUAL 0)
  message(FATAL_ERROR "--layers 1 returned ${rc1}, expected 0")
endif()
set(exp1 "model-00001-of-00004.safetensors\nmodel-00002-of-00004.safetensors")
if(NOT out1 STREQUAL exp1)
  message(FATAL_ERROR "--layers 1 mapped to [${out1}], expected [${exp1}]")
endif()

map_layers(out2 rc2 "${FIXTURES}/partial_index.json" --layers 2)
if(NOT rc2 EQUAL 0)
  message(FATAL_ERROR "--layers 2 returned ${rc2}, expected 0")
endif()
if(NOT out2 MATCHES "model-00003-of-00004\\.safetensors")
  message(FATAL_ERROR "--layers 2 missed the shard holding half of layer 1: [${out2}]")
endif()
if(out2 MATCHES "model-00004-of-00004\\.safetensors")
  message(FATAL_ERROR "--layers 2 pulled the far-future shard: [${out2}]")
endif()

foreach(bad 0 9 abc 1junk)
  map_layers(bad_out bad_rc "${FIXTURES}/partial_index.json" --layers "${bad}")
  if(NOT bad_rc EQUAL 2)
    message(FATAL_ERROR "--layers ${bad} returned ${bad_rc}, expected 2")
  endif()
endforeach()

map_layers(nomap_out nomap_rc "${FIXTURES}/partial_index_nomap.json" --layers 1)
if(NOT nomap_rc EQUAL 2)
  message(FATAL_ERROR "index without a weight_map returned ${nomap_rc}, expected 2")
endif()
