add_mlir_dialect_library(MLIRMini
        Dialect.cpp
        AST.cpp
        MLIRGen.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Mini

        DEPENDS
        MLIRMiniOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)