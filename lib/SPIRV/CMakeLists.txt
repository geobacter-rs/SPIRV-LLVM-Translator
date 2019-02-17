
if(NOT LLVM_LINK_LLVM_DYLIB)
# llvm-spirv is linked with DISABLE_LLVM_LINK_LLVM_DYLIB, which will
# be ignored if we specify the deps here.
  set(LLVM_LINK_COMPONENTS
    Analysis
    BitReader
    BitWriter
    Core
    ScalarOpts
    Support
    TransformUtils
    )
endif()

add_llvm_library(LLVMSPIRVLib
  LLVMToSPIRVDbgTran.cpp
  Mangler/FunctionDescriptor.cpp
  Mangler/Mangler.cpp
  Mangler/ManglingUtils.cpp
  Mangler/ParameterType.cpp
  OCL20To12.cpp
  OCL20ToSPIRV.cpp
  OCL21ToSPIRV.cpp
  OCLTypeToSPIRV.cpp
  OCLUtil.cpp
  SPIRVCanonicalization.cpp
  SPIRVLowerBool.cpp
  SPIRVLowerConstExpr.cpp
  SPIRVLowerMemmove.cpp
  SPIRVLowerOCLBlocks.cpp
  SPIRVLowerSPIRBlocks.cpp
  SPIRVReader.cpp
  SPIRVRegularizeLLVM.cpp
  SPIRVToLLVMDbgTran.cpp
  SPIRVToOCL20.cpp
  SPIRVUtil.cpp
  SPIRVWriter.cpp
  SPIRVWriterPass.cpp
  TransOCLMD.cpp
  libSPIRV/SPIRVBasicBlock.cpp
  libSPIRV/SPIRVDebug.cpp
  libSPIRV/SPIRVDecorate.cpp
  libSPIRV/SPIRVEntry.cpp
  libSPIRV/SPIRVFunction.cpp
  libSPIRV/SPIRVInstruction.cpp
  libSPIRV/SPIRVModule.cpp
  libSPIRV/SPIRVStream.cpp
  libSPIRV/SPIRVType.cpp
  libSPIRV/SPIRVValue.cpp

  DEPENDS
    intrinsics_gen
)

target_include_directories(LLVMSPIRVLib
  PRIVATE
    ${LLVM_INCLUDE_DIR}
    ${LLVM_SPIRV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libSPIRV
    ${CMAKE_CURRENT_SOURCE_DIR}/Mangler
)
