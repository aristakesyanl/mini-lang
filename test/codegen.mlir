module {
  mini.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = mini.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = mini.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = mini.matmul %0, %1 : tensor<*xf64>
    mini.return %2 : tensor<*xf64>
  }
  mini.func @main() {
    %0 = mini.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = mini.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = mini.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %3 = mini.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
    %4 = mini.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %5 = mini.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    mini.print %5 : tensor<*xf64>
    mini.return
  }
}
