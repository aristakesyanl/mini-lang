#ifndef MAKE_TILED_IMPLEMENTATION_TD_
#define MAKE_TILED_IMPLEMENTATION_TD_

include "mlir/Pass/PassBase.td"

def MakeTiledImplementation : Pass<"make-tiled-implementation"> {
  let summary = "Make tiled implementation of operation using the pattern rewrite engine";
  let description = [{
   Make tiled implementation of operation using the pattern rewrite engine
  }];
}


#endif  

