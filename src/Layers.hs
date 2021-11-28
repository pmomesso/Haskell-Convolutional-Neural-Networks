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

data TensorialLayer = ConvolutionalLayer KernelTensor BiasVector Activation ActivationDerivative | MaxPoolingLayer Int Int
data DenseLayer = DenseLayer RealMatrix BiasVector Activation ActivationDerivative | SoftmaxLayer RealMatrix BiasVector

type DenseNetwork = [ DenseLayer ]
type TensorialNetwork = [ TensorialLayer ]

data NeuralNetwork = ConvolutionalNetwork TensorialNetwork DenseNetwork

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
kernelExcitation image kernel bias = (bias+) <$> V.sum (convolveByChannel kernel image)

convLayerExcitation :: Num a => V.Vector (V.Vector (M.Matrix a)) -> V.Vector a -> V.Vector (M.Matrix a) -> V.Vector (M.Matrix a)
convLayerExcitation kernelTensor biasVector image = V.zipWith (kernelExcitation image) kernelTensor biasVector

tensorialExcitation :: TensorialLayer -> Image -> Image
tensorialExcitation (ConvolutionalLayer kernelTensor biasVector _ _) = convLayerExcitation kernelTensor biasVector
tensorialExcitation maxPoolingLayer = id

tensorialActivation :: TensorialLayer -> Image -> Image
tensorialActivation (ConvolutionalLayer kernelTensor biasVector act der) tensor = let excitationState = tensorialExcitation (ConvolutionalLayer kernelTensor biasVector act der) tensor in
                                                                               fmap (fmap act) excitationState
tensorialActivation (MaxPoolingLayer supportRows supportCols) tensor = let rows = M.nrows $ V.head tensor in
                                                                       let cols = M.ncols $ V.head tensor in
                                                                       fmap (\channel -> M.matrix (rows `quot` supportRows) (cols `quot` supportCols) (\(row, col) -> maximum $ M.toList (M.submatrix (2*(row-1)+1) (2*row) (2*(col-1)+1) (2*col) channel))) tensor

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


forwardConvNetwork :: TensorialNetwork -> Image -> Image
forwardConvNetwork convNet image = foldl tensorialActivation' image convNet
                                   where tensorialActivation' = flip tensorialActivation

forwardDenseNetwork :: DenseNetwork -> V.Vector Float -> V.Vector Float
forwardDenseNetwork denseNetwork x = foldl denseActivation' x denseNetwork
                                     where denseActivation' = flip denseActivation

flatten :: M.Matrix a -> V.Vector a
flatten mat = V.fromList $ M.toList mat

flattenMatrices :: V.Vector (M.Matrix a) -> V.Vector a
flattenMatrices mats = V.concat (V.toList $ fmap flatten mats)

forwardNeuralNetwork :: NeuralNetwork -> Image -> V.Vector Float
forwardNeuralNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) image = let finalTensor = forwardConvNetwork tensorialNetwork image in
                                                                                  let flattenedTensor = flattenMatrices finalTensor in
                                                                                  forwardDenseNetwork denseNetwork flattenedTensor

setSubMatrix :: M.Matrix a -> M.Matrix a -> Int -> Int -> M.Matrix a
setSubMatrix target block i j = let dRows = M.nrows block in
                                let dCols = M.ncols block in
                                let indices = [ (i', j') | i' <- [1..dRows], j' <- [1..dCols] ] in
                                let f mat (row, col) = M.setElem (M.getElem row col block) (row + i - 1, col + j - 1) mat in
                                foldl f target indices

indicatorFunction :: Eq a => a -> a -> Float
indicatorFunction elem x = if x == elem then 1.0 else 0.0

indicatorBlock :: Eq a => Int -> Int -> M.Matrix a -> M.Matrix a -> (Int, Int) -> M.Matrix Float
indicatorBlock supportRows supportCols output mat (i, j) = fmap (indicatorFunction $ M.getElem i j output) (M.submatrix ((i - 1)*supportRows + 1) ((i - 1)*supportRows + supportRows) ((j - 1)*supportCols + 1) ((j - 1)*supportCols + supportCols) mat)

backwardPoolingLayerSingleChannel :: Int -> Int -> RealMatrix -> RealMatrix -> RealMatrix -> RealMatrix
backwardPoolingLayerSingleChannel supportRows supportCols dE_dO output input =
                                                                        let blockIndices = [ (supportRows * (i-1) + 1, supportCols * (j-1) + 1) | i <- [1..(M.nrows dE_dO)], j <- [1..(M.ncols dE_dO)] ] in
                                                                        let f mat (i,j) = setSubMatrix mat (M.scaleMatrix (M.getElem ((i-1) `quot` supportRows + 1) ((j-1) `quot` supportCols + 1) dE_dO) (indicatorBlock supportRows supportCols output mat ((i-1) `quot` supportRows + 1, (j-1) `quot` supportCols + 1))) i j in
                                                                        foldl f input blockIndices

backwardPoolingLayerMultiChannel :: Int -> Int -> Image -> Image -> Image -> Image
backwardPoolingLayerMultiChannel supportRows supportCols = V.zipWith3 (backwardPoolingLayerSingleChannel supportRows supportCols)

{- TODO: dE_dI, dE_dK, dE_dBias -}
diffKernelSingleChannel :: Int -> Int -> RealMatrix -> RealMatrix -> RealMatrix
diffKernelSingleChannel kRows kCols dE_dH input =
                                                let indices = [ (i,j) | i <- [1..(M.nrows dE_dH)], j <- [1..(M.ncols dE_dH)] ] in
                                                let f dE_dK (i, j) = dE_dK + M.scaleMatrix (M.getElem i j dE_dH) (M.submatrix i (i + kRows - 1) j (j + kCols - 1) input) in
                                                foldl f (M.matrix kRows kCols (const 0)) indices

diffKernelMultiChannel :: Int -> Int -> Image -> RealMatrix -> Kernel
diffKernelMultiChannel kRows kCols inputs dE_dH = fmap (diffKernelSingleChannel kRows kCols dE_dH) inputs

diffKernelTensor :: Int -> Int -> Image -> Image -> KernelTensor
diffKernelTensor kRows kCols inputs = fmap (diffKernelMultiChannel kRows kCols inputs)

diffBiasSingleChannel dE_dH = sum $ M.toList dE_dH

diffBias = V.map diffBiasSingleChannel

diffInputSingleChannel dE_dH kernel = let indices = [ (i, j) | i <- [1..(M.nrows dE_dH)], j <- [1..(M.ncols dE_dH)] ] in
                                      let kRows = M.nrows kernel in
                                      let kCols = M.ncols kernel in
                                      let f mat (i, j) = setSubMatrix mat (M.submatrix i (i + kRows - 1) j (j + kCols - 1) mat + M.scaleMatrix (M.getElem i j dE_dH) kernel) i j in
                                      foldl f (M.matrix (M.nrows dE_dH + kRows - 1) (M.ncols dE_dH + kCols - 1) (const 0)) indices

diffInputMultiChannel :: RealMatrix -> Image -> Kernel
diffInputMultiChannel dE_dH = fmap (diffInputSingleChannel dE_dH)

diffInput :: Image -> KernelTensor -> KernelTensor
diffInput = V.zipWith diffInputMultiChannel

{- TODO: dE_dH functions for tensorial layers -}
deltasConvSingleChannel activationDerivative excitationChannel = elemwiseMult (fmap activationDerivative excitationChannel)

deltasConvMultiChannel activationDerivative = V.zipWith (deltasConvSingleChannel activationDerivative)

{- TODO -}
-- backwardTensorialLayer :: TensorialLayer -> Image -> Image -> Image
-- backwardTensorialLayer (MaxPoolingLayer supportRows supportCols) dE_dO input = foldl