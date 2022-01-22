module TensorialLayer where
import CommonTypes

import qualified Data.Vector as V
import qualified Data.Matrix as M

data TensorialLayer = ConvolutionalLayer KernelTensor BiasVector Activation ActivationDerivative | MaxPoolingLayer Int Int

windowIndices :: Int -> Int -> M.Matrix a -> [(Int, Int)]
windowIndices numRows numCols rm = [(i, j) | i <- [1..(M.nrows rm - numRows + 1)], j <- [1..(M.ncols rm - numCols + 1)]]

window :: M.Matrix a -> Int -> (Int, Int) -> M.Matrix a
window mat kerDim (row, col) = M.submatrix row (row + kerDim - 1) col (col + kerDim - 1) mat

windows :: M.Matrix a -> Int -> [M.Matrix a]
windows mat kerDim = map windowStartingAtIndex indices
    where windowStartingAtIndex = window mat kerDim
          indices = windowIndices kerDim kerDim mat

{- Convolution operation definition -}
convolve :: RealMatrix  -> RealMatrix -> RealMatrix
convolve ker mat = let kerDim = M.nrows ker in
                   let rows = M.nrows mat in
                   let cols = M.ncols mat in
                   M.matrix (rows - kerDim + 1) (cols - kerDim + 1) (applyKernel ker . window mat kerDim)

convolveByChannel :: Kernel -> Image -> Image
convolveByChannel = V.zipWith convolve

kernelExcitation :: Kernel -> Image -> Bias -> RealMatrix
kernelExcitation image kernel bias = (+bias) <$> sumMatrices (convolveByChannel kernel image)

convLayerExcitation :: KernelTensor -> BiasVector -> Image -> Image
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

backwardPoolingLayerSingleChannel :: Int -> Int -> RealMatrix -> RealMatrix -> RealMatrix -> RealMatrix
backwardPoolingLayerSingleChannel supportRows supportCols dE_dO output input =
                                                                        let blockIndices = [ (supportRows * (i-1) + 1, supportCols * (j-1) + 1) | i <- [1..(M.nrows dE_dO)], j <- [1..(M.ncols dE_dO)] ] in
                                                                        let f mat (i,j) = setSubMatrix mat (M.scaleMatrix (M.getElem ((i-1) `quot` supportRows + 1) ((j-1) `quot` supportCols + 1) dE_dO) (indicatorBlock supportRows supportCols output mat ((i-1) `quot` supportRows + 1, (j-1) `quot` supportCols + 1))) i j in
                                                                        foldl f input blockIndices

backwardPoolingLayerMultiChannel :: Int -> Int -> Image -> Image -> Image -> Image
backwardPoolingLayerMultiChannel supportRows supportCols = V.zipWith3 (backwardPoolingLayerSingleChannel supportRows supportCols)

diffKernelSingleChannel :: Int -> Int -> RealMatrix -> RealMatrix -> RealMatrix
diffKernelSingleChannel kRows kCols dE_dH input =
                                                let indices = [ (i,j) | i <- [1..(M.nrows dE_dH)], j <- [1..(M.ncols dE_dH)] ] in
                                                let f dE_dK (i, j) = dE_dK + M.scaleMatrix (M.getElem i j dE_dH) (M.submatrix i (i + kRows - 1) j (j + kCols - 1) input) in
                                                foldl f (M.matrix kRows kCols (const 0)) indices

diffKernelMultiChannel :: Int -> Int -> Image -> RealMatrix -> Kernel
diffKernelMultiChannel kRows kCols inputs dE_dH = fmap (diffKernelSingleChannel kRows kCols dE_dH) inputs

diffKernelTensor :: Int -> Int -> Image -> Image -> KernelTensor
diffKernelTensor kRows kCols inputs = fmap (diffKernelMultiChannel kRows kCols inputs)

diffKernelBiasSingleChannel :: RealMatrix -> Float
diffKernelBiasSingleChannel dE_dH = sum $ M.toList dE_dH

diffKernelBias :: Image -> BiasVector
diffKernelBias = V.map diffKernelBiasSingleChannel

diffInputSingleChannel :: RealMatrix -> RealMatrix -> RealMatrix
diffInputSingleChannel dE_dH kernel = let indices = [ (i, j) | i <- [1..(M.nrows dE_dH)], j <- [1..(M.ncols dE_dH)] ] in
                                      let kRows = M.nrows kernel in
                                      let kCols = M.ncols kernel in
                                      let f mat (i, j) = setSubMatrix mat (M.submatrix i (i + kRows - 1) j (j + kCols - 1) mat + M.scaleMatrix (M.getElem i j dE_dH) kernel) i j in
                                      foldl f (M.matrix (M.nrows dE_dH + kRows - 1) (M.ncols dE_dH + kCols - 1) (const 0)) indices

diffInputMultiChannel :: RealMatrix -> Kernel -> Image
diffInputMultiChannel dE_dH = fmap (diffInputSingleChannel dE_dH)

diffInput :: Image -> KernelTensor -> Image
diffInput dE_dHChannels kernelTensor = V.sum (V.zipWith diffInputMultiChannel dE_dHChannels kernelTensor)

diffTensorialLayer :: TensorialLayer -> Image -> Image -> KernelTensor
diffTensorialLayer (ConvolutionalLayer kernelTensor _ _ _) input deltas = diffKernelTensor (M.nrows $ (V.head . V.head) kernelTensor) (M.ncols $ (V.head . V.head) kernelTensor) input deltas
diffTensorialLayer MaxPoolingLayer {} input deltas = V.empty

diffBiasTensorialLayer :: TensorialLayer -> V.Vector (M.Matrix Float) -> V.Vector Float
diffBiasTensorialLayer ConvolutionalLayer {} deltas = diffKernelBias deltas
diffBiasTensorialLayer MaxPoolingLayer {} deltas = V.empty

deltasConvSingleChannel :: (Float -> Float) -> RealMatrix -> RealMatrix -> RealMatrix
deltasConvSingleChannel activationDerivative excitationChannel = elemwiseMult (fmap activationDerivative excitationChannel)

deltasConvMultiChannel :: (Float -> Float) -> Image -> Image -> Image
deltasConvMultiChannel activationDerivative = V.zipWith (deltasConvSingleChannel activationDerivative)

backwardTensorialLayer :: TensorialLayer -> Image -> Image -> Image -> (Image, Image)
backwardTensorialLayer (MaxPoolingLayer supportRows supportCols) inputChannels outputChannels dE_dOChannels = let dE_dO = backwardPoolingLayerMultiChannel supportRows supportCols dE_dOChannels outputChannels inputChannels in
                                                                                                              (dE_dO, dE_dO)
backwardTensorialLayer (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputChannels excitationChannels dE_dOChannels =
                                                                                      let deltas = deltasConvMultiChannel activationDerivative excitationChannels dE_dOChannels in
                                                                                      (diffInput deltas kernelTensor, deltas)

forwardTensorialLayer (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputs =
                                                let excitationState = tensorialExcitation (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputs in
                                                let activationState = tensorialActivation (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputs in
                                                (ConvolutionalLayerState inputs excitationState, activationState)
forwardTensorialLayer (MaxPoolingLayer supportRows supportCols) inputs =
                                                let activation = tensorialActivation (MaxPoolingLayer supportRows supportCols) inputs in
                                                (MaxPoolingLayerState inputs activation, activation)

applyTensorialAction :: TensorialLayer -> GradientAction -> TensorialLayer
applyTensorialAction (ConvolutionalLayer kernelTensor biasVector act der) (TensorialLayerAction dK dB) = ConvolutionalLayer (sumTensors kernelTensor dK) (biasVector + dB) act der
applyTensorialAction (MaxPoolingLayer rows cols) _ = MaxPoolingLayer rows cols
applyTensorialAction layer _ = error "Bad layer action pairing"

diffTensorialLayerWithState :: TensorialLayer -> LayerState -> BackpropagationResult -> (KernelTensor, V.Vector Float)
diffTensorialLayerWithState tensorialLayer (ConvolutionalLayerState input _) (TensorialLayerBPResult _ dE_dH) = (diffTensorialLayer tensorialLayer input dE_dH, diffBiasTensorialLayer tensorialLayer dE_dH)
diffTensorialLayerWithState tensorialLayer (MaxPoolingLayerState input _) (TensorialLayerBPResult _ dE_dH) = (diffTensorialLayer tensorialLayer input dE_dH, diffBiasTensorialLayer tensorialLayer dE_dH)
diffTensorialLayerWithState x y z = error "Bad pairing of layer state and input"

{- Utility function -}
resultingDimension :: TensorialLayer -> (Int, Int, Int) -> (Int, Int, Int)
resultingDimension (ConvolutionalLayer tensor _ _ _) (channels, rows, cols) =
            let numFilters = V.length tensor in
            let numRows = M.nrows $ (V.head . V.head) tensor in
            let numCols = M.ncols $ (V.head . V.head) tensor in
            (numFilters, rows - numRows + 1, cols - numCols + 1)

resultingDimension (MaxPoolingLayer suppRows suppCols) (channels, rows, cols) =
            (channels, rows `quot` suppRows, cols `quot` suppCols)