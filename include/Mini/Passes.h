#ifndef MINI_LANG_PASSES_H
#define MINI_LANG_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace mini_lang {
std::unique_ptr<Pass> createShapeInferencePass();
} 
} 

#endif
