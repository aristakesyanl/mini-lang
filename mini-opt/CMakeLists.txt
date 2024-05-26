get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRMini
        )
add_llvm_executable(mini-opt mini-opt.cpp)

llvm_update_compile_flags(mini-opt)
target_link_libraries(mini-opt PRIVATE ${LIBS})
