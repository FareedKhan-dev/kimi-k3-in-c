if(NOT DEFINED CLI)
  message(FATAL_ERROR "CLI was not provided")
endif()

execute_process(
  COMMAND "${CLI}" fake --ids 1 --preset ultra
  RESULT_VARIABLE no_trunk_rc
  OUTPUT_QUIET ERROR_QUIET)
if(NOT no_trunk_rc EQUAL 2)
  message(FATAL_ERROR "ultra mode without --trunk returned ${no_trunk_rc}, expected 2")
endif()

execute_process(
  COMMAND "${CLI}" fake --ids 1 --preset ultra --trunk fake --spec 2
  RESULT_VARIABLE spec_rc
  OUTPUT_QUIET ERROR_QUIET)
if(NOT spec_rc EQUAL 2)
  message(FATAL_ERROR "ultra mode with --spec returned ${spec_rc}, expected 2")
endif()
