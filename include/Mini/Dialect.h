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

#include <optional>


#include "Mini/Dialect.h.inc"
#include "mlir/Dialect/Tensor/IR/TensorOps.h.inc"

const int64_t tile_size = 8;
const int64_t rank = 2;
namespace mlir{
    namespace mini_lang{
        std::optional<DenseElementsAttr> getOperandAttribute(Value val);
        struct SliceParameters{
            ArrayRef<OpFoldResult> offsets;
            ArrayRef<OpFoldResult> sizes;
            ArrayRef<OpFoldResult> slides;
        };

        SliceParameters getSliceParameters(mlir::OpBuilder &builder, int64_t index);
    }
}

#define GET_OP_CLASSES
#include "Mini/Ops.h.inc"

#endif 
