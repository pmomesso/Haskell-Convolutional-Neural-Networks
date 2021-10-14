module Layers where

import qualified Data.Matrix as M
import qualified Data.Vector as V

type RealMatrix = M.Matrix Float
type Kernel = V.Vector (M.Matrix Float)
type KernelTensor = V.Vector Kernel
type Bias = Float
type BiasVector = V.Vector Bias
type Activation = Float -> Float
type ActivationDerivative = Float -> Float

type Image = V.Vector RealMatrix

data TensorialLayer = ConvolutionalLayer KernelTensor BiasVector Activation ActivationDerivative | MaxPoolingLayer Int
data DenseLayer = DenseLayer RealMatrix BiasVector Activation ActivationDerivative | SoftmaxLayer RealMatrix BiasVector

type DenseNetwork = [ DenseLayer ]
type ConvolutionalNetwork = [ TensorialLayer ]

data NeuralNetwork = NeuralNetwork ConvolutionalNetwork DenseNetwork | JustDenseNetwork DenseNetwork

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

convolveByChannel :: Num a => V.Vector (M.Matrix a) -> V.Vector (M.Matrix a) -> V.Vector (M.Matrix a)
convolveByChannel = V.zipWith convolve

sumMatricesWithDim :: Num a => Int -> Int -> V.Vector (M.Matrix a) -> M.Matrix a
sumMatricesWithDim rows cols
  = foldr (+) (M.matrix rows cols (const 0))

{- Precond: forall m, n in mats: nrows m = nrows n and ncols m = nrcols n -}
sumMatrices :: Num a => V.Vector (M.Matrix a) -> M.Matrix a
sumMatrices mats = let rows = M.nrows $ V.head mats in
                   let cols = M.ncols $ V.head mats in
                       sumMatricesWithDim rows cols mats

kernelExcitation :: Num a => V.Vector (M.Matrix a) -> V.Vector (M.Matrix a) -> a ->  M.Matrix a
kernelExcitation image kernel bias = (bias+) <$> sumMatrices (convolveByChannel kernel image)

convLayerExcitation :: Num a => V.Vector (V.Vector (M.Matrix a)) -> V.Vector a -> V.Vector (M.Matrix a) -> V.Vector (M.Matrix a)
convLayerExcitation kernelTensor biasVector image = V.zipWith (kernelExcitation image) kernelTensor biasVector

tensorialExcitation :: TensorialLayer -> Image -> Image
tensorialExcitation (ConvolutionalLayer kernelTensor biasVector _ _) = convLayerExcitation kernelTensor biasVector
tensorialExcitation _ = error "Unimplemented!"

tensorialActivation :: TensorialLayer -> Image -> Image
tensorialActivation (ConvolutionalLayer kernelTensor biasVector act der) image = let excitationState = tensorialExcitation (ConvolutionalLayer kernelTensor biasVector act der) image in
                                                                               fmap (fmap act) excitationState
tensorialActivation _ image = error "Unimplemented!"

denseExcitation :: DenseLayer -> V.Vector Float -> V.Vector Float
denseExcitation (DenseLayer mat bias _ _) x = let xColVector = M.colVector x in
                                              let biasColVector = M.colVector bias in
                                              M.getCol 1 $ mat * xColVector + biasColVector
denseExcitation (SoftmaxLayer mat bias) x = let xColVector = M.colVector x in
                                            let biasColVector = M.colVector bias in
                                            M.getCol 1 $ mat * xColVector + biasColVector

denseActivation :: DenseLayer -> V.Vector Float -> V.Vector Float
denseActivation (DenseLayer mat bias act der) x = let excitation = denseExcitation (DenseLayer mat bias act der) x in
                                                  fmap act excitation
denseActivation (SoftmaxLayer mat bias) x = let excitation = denseExcitation (SoftmaxLayer mat bias) x in
                                            softmax excitation

softmax :: V.Vector Float -> V.Vector Float
softmax vector = let exps = fmap exp vector in
                 let s = sum exps in
                 fmap (/s) exps

forwardDenseNetwork :: DenseNetwork -> V.Vector Float -> V.Vector Float
forwardDenseNetwork denseNetwork x = foldl denseActivation' x denseNetwork
                                     where denseActivation' = flip denseActivation