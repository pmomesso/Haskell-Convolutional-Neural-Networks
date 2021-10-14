module Layers where

import qualified Data.Matrix as M
import qualified Data.Vector as V

type RealMatrix = M.Matrix Float
type Kernel = [ RealMatrix ]
type Bias = V.Vector Float
type Activation = Float -> Float
type ActivationDerivative = Float -> Float

data TensorialLayer = ConvolutionalLayer Kernel | MaxPoolingLayer Int Int
data DenseLayer = DenseLayer Activation ActivationDerivative RealMatrix Bias | SoftmaxLayer RealMatrix Bias

type DenseNetwork = [ DenseLayer ]
type ConvolutionalNetwork = [ TensorialLayer ]

data NeuralNetwork = NeuralNetwork ConvolutionalNetwork DenseNetwork

windowIndices :: Int -> Int -> M.Matrix a -> [(Int, Int)]
windowIndices numRows numCols rm = [(i, j) | i <- [1..(M.nrows rm - numRows + 1)], j <- [1..(M.ncols rm - numCols + 1)]]

window :: M.Matrix a -> Int -> (Int, Int) -> M.Matrix a
window mat kerDim (row, col) = M.submatrix row (row + kerDim - 1) col (col + kerDim - 1) mat

windows :: M.Matrix a -> Int -> [M.Matrix a]
windows mat kerDim = map windowStartingAtIndex indices
    where windowStartingAtIndex = window mat kerDim
          indices = windowIndices kerDim kerDim mat

elemwiseMult :: Num a => M.Matrix a -> M.Matrix a -> M.Matrix a
elemwiseMult = M.elementwise (*)

applyKernel :: Num a => M.Matrix a -> M.Matrix a -> a
applyKernel m1 m2 = sum $ elemwiseMult m1 m2

{- Opcion 1
convolve :: Num a => M.Matrix a -> M.Matrix a -> M.Matrix a
convolve ker mat = let kerDim = M.nrows ker in 
                   let rows = M.nrows mat in
                   let cols = M.ncols mat in
                   M.fromList (rows - kerDim + 1) (cols - kerDim + 1) $ map (applyKernel ker) (windows mat (M.nrows ker))
-}

{- Opcion 2 -}
convolve :: Num a => M.Matrix a -> M.Matrix a -> M.Matrix a
convolve ker mat = let kerDim = M.nrows ker in
                   let rows = M.nrows mat in
                   let cols = M.ncols mat in
                   M.matrix (rows - kerDim + 1) (cols - kerDim + 1) (applyKernel ker . window mat kerDim)