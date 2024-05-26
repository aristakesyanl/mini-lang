#ifndef MLIR_MINI_DIALECT_H_
#define MLIR_MINI_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the mini
/// dialect.
#include "Mini/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// mini operations.
#define GET_OP_CLASSES
#include "Mini/Ops.h.inc"

#endif 
