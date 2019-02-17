set(LLVM_LINK_COMPONENTS
  SPIRVLib
  Analysis
  BitReader
  BitWriter
  Core
  ScalarOpts
  Support
  TransformUtils
)

add_llvm_tool(llvm-spirv
  llvm-spirv.cpp

# Sadly, linking this tool with the dylib causes issues with globals in
# they dylib. Not sure exactly what is going on, but just not using it at
# least works
  DISABLE_LLVM_LINK_LLVM_DYLIB
)

if (BUILD_EXTERNAL)
  target_link_libraries(llvm-spirv PRIVATE LLVMSPIRVLib)
endif()

target_include_directories(llvm-spirv
  PRIVATE
    ${LLVM_INCLUDE_DIR}
    ${LLVM_SPIRV_INCLUDE_DIRS}
)
