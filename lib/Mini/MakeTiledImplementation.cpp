#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "Mini/Dialect.h"
#include "Mini/Passes.h"
#include "Mini/MakeTiledImplementation.h"
#include "Mini/ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

#define DEBUG_TYPE "make-tile"


using namespace mlir;
using namespace mlir::mini_lang;
using namespace mlir::tensor;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "Mini/MakeTiledImplementation.cpp.inc"


struct SliceParameters{
    ArrayRef<OpFoldResult> offsets;
    ArrayRef<OpFoldResult> sizes;
    ArrayRef<OpFoldResult> slides;
};

SliceParameters getSliceParameters(PatternRewriter &rewriter, int64_t index){
  SliceParameters param;
  ArrayRef<int64_t> off{index * tile_size, index * tile_size};
  ArrayRef<int64_t> sz{tile_size, tile_size};
  ArrayRef<int64_t> sld{1, 1};

  SmallVector<OpFoldResult> vecOff = getAsIndexOpFoldResult(rewriter.getContext(), off);
  SmallVector<OpFoldResult> vecSz = getAsIndexOpFoldResult(rewriter.getContext(), sz);
  SmallVector<OpFoldResult> vecSld = getAsIndexOpFoldResult(rewriter.getContext(), sld);

  param.offsets = ArrayRef(vecOff);
  param.sizes = ArrayRef(vecSz);
  param.slides = ArrayRef(vecSld);

  return param;

}

struct MakeTiledImplementationRewrite : public OpRewritePattern<AddOp> {
  MakeTiledImplementationRewrite(mlir::MLIRContext *context) : OpRewritePattern<AddOp>(context, /*benefit=*/1) {}


  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    auto tensorLHS = dyn_cast<RankedTensorType>(lhs.getType());
    auto tensorRHS = dyn_cast<RankedTensorType>(rhs.getType());
    int64_t tile_count = 1;

    if(tensorLHS && tensorRHS){
      ArrayRef<int64_t> dimLHS{tensorLHS.getShape().begin(), tensorLHS.getShape().end()};
      ArrayRef<int64_t> dimRHS{tensorLHS.getShape().begin(), tensorLHS.getShape().end()}; 

      if(dimLHS != dimRHS){
        return failure();
      } else{
        //if tensor dimensions match try tiling them
        for(auto dim : dimLHS){
          //check that dimenstions are multiples of tile_size
          if(dim % tile_size != 0){
            //if tensor dimensions are not multiples of tile_size done tile
            return failure();
          } 
          tile_count *= dim / tile_size;
        }
      }

      if(tile_count > 1){
        std::vector<Value> addResults;

        for(int64_t i = 0; i < tile_count; i++){
          SliceParameters param = getSliceParameters(rewriter, i);

          // Value extracted = rewriter.create<tensor::ExtractSliceOp>(
          //                           target.getLoc(), target.getDest(), target.getMixedOffsets(),
          //                           target.getMixedSizes(), target.getMixedStrides());

          mlir::Type elementType = rewriter.getF64Type();
          auto destTensorType = RankedTensorType::get(ArrayRef<int64_t>{tile_size,tile_size}, elementType);

          auto tileLhs = rewriter.create<tensor::ExtractSliceOp>(op.getLoc(), destTensorType,
                                                                lhs, param.offsets, param.sizes, param.slides);

          addResults.push_back(tileLhs);

          auto tileRhs = rewriter.create<tensor::ExtractSliceOp>(op.getLoc(), destTensorType,
                                                                rhs, param.offsets, param.sizes, param.slides);

          addResults.push_back(tileRhs);
          
          addResults.push_back(rewriter.create<AddOp>(op.getLoc(), tileLhs, tileRhs));
        }

        ValueRange vals(addResults);

        rewriter.replaceOp(op, vals);
        return success();

      }
    }

    return success();

  }
};


struct MakeTiledImplementation : mlir::mini_lang::impl::MakeTiledImplementationBase<MakeTiledImplementation> {
  using MakeTiledImplementationBase::MakeTiledImplementationBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MakeTiledImplementationRewrite>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};


