module CommonTypes where

import qualified Data.Matrix as M
import qualified Data.Vector as V

applyPair f (x, y) = (f x, f y)

{- Types for representing datapoints -}
type RealMatrix = M.Matrix Float

type Kernel = V.Vector RealMatrix
type KernelTensor = V.Vector Kernel

type Bias = Float
type BiasVector = V.Vector Bias

type Activation = Float -> Float
type ActivationDerivative = Float -> Float

type Image = V.Vector RealMatrix

{- 
Type for representing layer forwarding state

    All type instances are of the form "Constructor Input Excitation"
-}
data LayerState = EmptyState
                 | MaxPoolingLayerState Image Image 
                 | ConvolutionalLayerState Image Image 
                 | DenseLayerState (V.Vector Float) (V.Vector Float) 
                 | SoftmaxLayerState (V.Vector Float) (V.Vector Float) 

extractOutputs :: LayerState -> V.Vector Float
extractOutputs (DenseLayerState input output) = output
extractOutputs (SoftmaxLayerState input output) = output
extractOutputs _ = error "Not implemented!"

{- 
Type for representing layer backpropagation state 

    The result of backpropagation operations is always twofold: 
        - The derivative of the error function w.r.t the layer's weights. This is needed for weight adjustment
        - The derivative of the error function w.r.t the layer's inputs. This is needed for further backpropagation
-}
data BackpropagationResult = EmptyBPResultDense (V.Vector Float) 
                            | EmptyBPResultTensorial Image 
                            | DenseLayerBPResult (V.Vector Float) (V.Vector Float) 
                            | TensorialLayerBPResult Image Image

{- 
Type for representing training actions to be applied to layers 

    The actions to be applied during Gradient Descent always involve adjusting the layer's weight matrix, either in the form of an actual matrix or a kernel, and adjusting the layer's bias vector
-}
data GradientAction = TensorialLayerAction KernelTensor BiasVector 
                     | DenseLayerAction RealMatrix RealMatrix

{- Common matrix and/or vector utilities and class declarations -}
elemwiseMult :: RealMatrix -> RealMatrix -> RealMatrix
elemwiseMult = M.elementwise (*)

shiftLeft v = V.generate (V.length v) (\index -> v V.! ((index - 1) `mod` V.length v))

toVector realMatrix = V.fromList $ M.toList realMatrix

applyKernel :: RealMatrix -> RealMatrix -> Float
applyKernel m1 m2 = sum $ elemwiseMult m1 m2

sumMatricesWithDim :: Int -> Int -> V.Vector (M.Matrix Float) -> M.Matrix Float
sumMatricesWithDim rows cols
  = foldr (+) (M.matrix rows cols (const 0))

{- Precond: forall m, n in mats: nrows m = nrows n and ncols m = nrcols n -}
sumMatrices :: V.Vector (M.Matrix Float) -> M.Matrix Float
sumMatrices mats = let rows = M.nrows $ V.head mats in
                   let cols = M.ncols $ V.head mats in
                    sumMatricesWithDim rows cols mats

flatten :: M.Matrix a -> V.Vector a
flatten mat = V.fromList $ M.toList mat

flattenMatrices :: V.Vector (M.Matrix a) -> V.Vector a
flattenMatrices mats = V.concat (V.toList $ fmap flatten mats)

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

partitionInRec :: Int -> [a] -> [[a]]
partitionInRec perPartition [] = []
partitionInRec perPartition list = take perPartition list : partitionInRec perPartition (drop perPartition list)

partitionIn :: Int -> [a] -> [[a]]
partitionIn n list = let numValues = length list in
                     let perPartition = numValues `quot` n in
                     partitionInRec perPartition list

deflattenToSameDimensionsOf :: Image -> V.Vector Float -> Image
deflattenToSameDimensionsOf image vector = let asLists = partitionIn (V.length image) (V.toList vector) in
                                           let rows = M.nrows $ V.head image in
                                           let cols = M.ncols $ V.head image in
                                           V.fromList $ fmap (M.fromList rows cols) asLists

scaleKernelTensor :: Float -> KernelTensor -> KernelTensor
scaleKernelTensor eta = fmap (fmap $ M.scaleMatrix eta)

sumKernelTensors :: KernelTensor -> KernelTensor -> KernelTensor
sumKernelTensors = V.zipWith (V.zipWith (+))

scaleVector :: Float -> V.Vector Float -> V.Vector Float
scaleVector eta = fmap (*eta)

sumTensors :: KernelTensor -> KernelTensor -> KernelTensor
sumTensors = V.zipWith (V.zipWith (+))

instance Num a => Num (V.Vector a) where
  (+) = V.zipWith (+)