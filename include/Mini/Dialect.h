#ifndef MLIR_TUTORIAL_MINI_LANG_DIALECT_H_
#define MLIR_TUTORIAL_MINI_LANG_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Mini/ShapeInferenceInterface.h"
#include "mlir/Interfaces/TilingInterface.h"


#include "Mini/Dialect.h.inc"


#define GET_OP_CLASSES
#include "Mini/Ops.h.inc"

#endif 
