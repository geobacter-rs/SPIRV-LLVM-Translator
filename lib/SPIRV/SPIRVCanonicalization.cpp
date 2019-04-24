//===- SPIRVCanonicalization.cpp - Clean up general IR ------- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// Cleanup IR. In particular, some optimization produce some wacky ptrtoint
// inttoptr chains and addrspace casts.
// Rust also sometimes creates addrspace casts of null.
//
//


#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "SPIRVMDBuilder.h"
#include "SPIRVMDWalker.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

  class SPIRVCanonicalization : public FunctionPass {
  public:
    SPIRVCanonicalization() : FunctionPass(ID) {
      initializeSPIRVCanonicalizationPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function& F) override;
    bool doInitialization(Module& M) override;

    static char ID;

  private:

    void resetState() {
      for(auto* Use : Delete) {
        Use->eraseFromParent();
      }
      Delete.clear();

      Changed = false;
    }

    Value* stripNullPtrAddrSpaceCasts(Value* C) {
      if (auto* Expr = dyn_cast<ConstantExpr>(C)) {
        if (Expr->getOpcode() == Instruction::AddrSpaceCast) {
          if (isa<ConstantPointerNull>(Expr->getOperand(0))) {
            C = ConstantPointerNull::get(cast<PointerType>(C->getType()));
          }
        }
      } else if (auto* Cast = dyn_cast<AddrSpaceCastInst>(C)) {
        auto* Src = Cast->getOperand(0);
        if(isa<ConstantPointerNull>(Src)) {
          C = ConstantPointerNull::get(cast<PointerType>(C->getType()));
          Cast->replaceAllUsesWith(C);
        }
      }

      return C;
    }

    void runOnIntToPtr(IntToPtrInst* I);
    void runOnAddrSpaceCast(AddrSpaceCastInst* AS);
    void runOnPhi(PHINode* Phi);

    bool runOnFunction_(Function& F);

    Module *M = nullptr;
    LLVMContext *Ctx = nullptr;
    bool Changed = false;

    SmallVector<Instruction*, 32> Delete;
    SmallVector<std::pair<Instruction*, Instruction*>, 32> NewInsts;
  };

  char SPIRVCanonicalization::ID = 0;

  bool SPIRVCanonicalization::runOnFunction(Function& F) {
    bool Changed = false;

    resetState();

    while(runOnFunction_(F)) {
      Changed = true;
      resetState();
    }

    return Changed;
  }

  bool SPIRVCanonicalization::doInitialization(Module& M) {
    this->M = &M;
    this->Ctx = &M.getContext();
    return Pass::doInitialization(M);
  }

  bool SPIRVCanonicalization::runOnFunction_(Function& F) {
    for(auto BBIt = F.begin(); BBIt != F.end(); ++BBIt) {
      BasicBlock& BB = *BBIt;
      for(auto IIt = BB.begin(); IIt != BB.end(); ++IIt) {
        Instruction* I = &*IIt;

        if(auto* IntToPtr = dyn_cast<IntToPtrInst>(I)) {
          runOnIntToPtr(IntToPtr);
          continue;
        } else if(auto* AS = dyn_cast<AddrSpaceCastInst>(I)) {
          runOnAddrSpaceCast(AS);
          continue;
        } else if(auto* II = dyn_cast<IntrinsicInst>(I)) {
          if(II->getIntrinsicID() == Intrinsic::assume) {
            II->dropAllReferences();
            Delete.emplace_back(II);
            Changed = true;
            continue;
          }
        }

        // remove addrspace casts of null:
        for(unsigned OpIt = 0; OpIt != I->getNumOperands(); ++OpIt) {
          auto* Operand = I->getOperand(0);
          auto* NewOperand = stripNullPtrAddrSpaceCasts(Operand);
          if(Operand != NewOperand) {
            I->setOperand(OpIt, NewOperand);
            Changed = true;
          }
        }
      }

      for(auto& New : NewInsts) {
        New.second->insertAfter(New.first);
      }
      NewInsts.clear();
    }

    return this->Changed;
  }

  void SPIRVCanonicalization::runOnIntToPtr(IntToPtrInst *I) {
    if(auto* PtrToInt = dyn_cast<PtrToIntInst>(I->getOperand(0))) {
      auto* ITy = I->getType();
      auto* Src = PtrToInt->getOperand(0);
      auto* SrcTy = Src->getType();
      if(ITy == SrcTy) {
        // we have an identity ptrtoint -> inttoptr chain.
        I->replaceAllUsesWith(Src);
        Delete.push_back(I);
        Changed = true;
      } else if(ITy->getPointerElementType() == SrcTy->getPointerElementType()) {
        // the types are only different in the address space. Insert a cast for this.
        auto* Cast = new AddrSpaceCastInst(Src, ITy);
        NewInsts.push_back({ I, Cast, });
        I->replaceAllUsesWith(Cast);
        Delete.push_back(I);
        Changed = true;
      }
    }
  }

  void SPIRVCanonicalization::runOnAddrSpaceCast(AddrSpaceCastInst *AS) {
    if(auto* SrcAS = dyn_cast<AddrSpaceCastInst>(AS->getOperand(0))) {
      auto* ASTy = AS->getType();
      auto* SrcSrc = SrcAS->getOperand(0);
      auto* SrcSrcTy = SrcSrc->getType();

      if(ASTy == SrcSrcTy) {
        AS->replaceAllUsesWith(SrcSrc);
        Delete.push_back(AS);
        Changed = true;
      }
    }
  }

  void SPIRVCanonicalization::runOnPhi(PHINode *Phi) {
    auto IsCandidate = [this] (Value* V) {
      this->stripNullPtrAddrSpaceCasts(V);
    };
  }

}

INITIALIZE_PASS(SPIRVCanonicalization, "spirv-canonicalization",
                "Canonicalize some undesirable patterns for SPIRV",
                false, false)

FunctionPass* llvm::createSPIRVCanonicalization() {
  return new SPIRVCanonicalization();
}