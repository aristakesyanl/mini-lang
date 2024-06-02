#ifndef MLIR_TUTORIAL_MINI_LANG_MAKETILEDIMPLEMENTATION_H_
#define MLIR_TUTORIAL_MINI_LANG_MAKETILEDIMPLEMENTATION_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mini_lang {

#define GEN_PASS_DEF_MAKETILEDIMPLEMENTATION
#include "Mini/MakeTiledImplementation.h.inc"

} 
} 
#endif 
