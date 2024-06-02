add_mlir_dialect_library(MLIRMini
        Dialect.cpp
        AST.cpp
        MLIRGen.cpp
        ShapeInferencePass.cpp
        ToyCombine.cpp
        MakeTiledImplementation.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Mini

        DEPENDS
        MLIRMiniOpsIncGen
        ShapeInferenceInterfaceIncGen
        ToyCombineIncGen
        MakeTiledImplementationIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)