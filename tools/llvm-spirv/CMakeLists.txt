set(LLVM_LINK_COMPONENTS
  SPIRVLib
  Analysis
  BitReader
  BitWriter
  Core
  Support
  TransformUtils
)

add_llvm_tool(llvm-spirv
  llvm-spirv.cpp
)

if (BUILD_EXTERNAL)
  target_link_libraries(llvm-spirv PRIVATE LLVMSPIRVLib)
endif()

target_include_directories(llvm-spirv
  PRIVATE
    ${LLVM_INCLUDE_DIR}
    ${LLVM_SPIRV_INCLUDE_DIRS}
)
