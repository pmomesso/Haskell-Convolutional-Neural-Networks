module Layers where

{- Most important functions: forwardNetworkWithState, nextNetwork -}

import qualified Data.Matrix as M
import qualified Data.Vector as V

import Data.Bifunctor ( Bifunctor(second) )

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

{- TODO: declare instance of Image for summing -}

instance Num a => Num (V.Vector a) where
  (+) = V.zipWith (+)

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

diffKernelBiasSingleChannel :: RealMatrix -> Float
diffKernelBiasSingleChannel dE_dH = sum $ M.toList dE_dH

diffKernelBias :: V.Vector (M.Matrix Float) -> V.Vector Float
diffKernelBias = V.map diffKernelBiasSingleChannel

diffInputSingleChannel :: RealMatrix -> RealMatrix -> RealMatrix
diffInputSingleChannel dE_dH kernel = let indices = [ (i, j) | i <- [1..(M.nrows dE_dH)], j <- [1..(M.ncols dE_dH)] ] in
                                      let kRows = M.nrows kernel in
                                      let kCols = M.ncols kernel in
                                      let f mat (i, j) = setSubMatrix mat (M.submatrix i (i + kRows - 1) j (j + kCols - 1) mat + M.scaleMatrix (M.getElem i j dE_dH) kernel) i j in
                                      foldl f (M.matrix (M.nrows dE_dH + kRows - 1) (M.ncols dE_dH + kCols - 1) (const 0)) indices

diffInputMultiChannel :: RealMatrix -> Kernel -> Image
diffInputMultiChannel dE_dH = fmap (diffInputSingleChannel dE_dH)

{- TODO: fix semantics. Should sum accross multiple channels of the resulting dE_dI's -}
diffInput :: Image -> KernelTensor -> Image
diffInput dE_dHChannels kernelTensor = V.sum (V.zipWith diffInputMultiChannel dE_dHChannels kernelTensor)

{- TODO: dE_dH functions for tensorial layers -}
deltasConvSingleChannel :: (Float -> Float) -> RealMatrix -> RealMatrix -> RealMatrix
deltasConvSingleChannel activationDerivative excitationChannel = elemwiseMult (fmap activationDerivative excitationChannel)

deltasConvMultiChannel :: (Float -> Float) -> Image -> Image -> Image
deltasConvMultiChannel activationDerivative = V.zipWith (deltasConvSingleChannel activationDerivative)

{- TODO -}
backwardTensorialLayer :: TensorialLayer -> Image -> Image -> Image -> (Image, Image)
backwardTensorialLayer (MaxPoolingLayer supportRows supportCols) inputChannels outputChannels dE_dOChannels = let dE_dO = backwardPoolingLayerMultiChannel supportRows supportCols dE_dOChannels outputChannels inputChannels in
                                                                                                              (dE_dO, dE_dO)
backwardTensorialLayer (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputChannels excitationChannels dE_dOChannels =
                                                                                      let deltas = deltasConvMultiChannel activationDerivative excitationChannels dE_dOChannels in
                                                                                      (diffInput deltas kernelTensor, deltas)

{- TODO: dE_dH, dE_dW, dE_dB, dE_dI functions for dense layers -}
softmaxDeltaTerms exps s dE_dO termIndex index = if termIndex == index
                                                 then ((exps V.! termIndex)*s - (exps V.! termIndex)^2)/s^2
                                                 else -(exps V.! termIndex)*(exps V.! index)/s^2

softmaxDeltaTerm exps s dE_dO termIndex = sum $ map (softmaxDeltaTerms exps s dE_dO termIndex) [1..(V.length exps)]

softmaxDeltas excitationVec dE_dO = let exps = fmap exp excitationVec in
                                    let s = sum exps in
                                    V.generate (V.length excitationVec) (softmaxDeltaTerm exps s dE_dO)

backwardDenseLayer :: DenseLayer -> V.Vector Float -> V.Vector Float -> V.Vector Float -> (RealMatrix, V.Vector Float)
backwardDenseLayer (DenseLayer weights bias activation activationDerivative) inputVector excitationVec dE_dO =
                                                                                      let deltasVector = activationDerivative <$> excitationVec in
                                                                                      let deltasColVector = elemwiseMult (M.colVector dE_dO) (M.colVector deltasVector) in
                                                                                      (M.transpose weights * deltasColVector, deltasVector)

backwardDenseLayer (SoftmaxLayer weights bias) inputVector excitationVec dE_dO =
                                                                      let deltasVector = softmaxDeltas excitationVec dE_dO in
                                                                      let deltasColVector = M.colVector $ softmaxDeltas excitationVec dE_dO in
                                                                      (M.transpose weights * deltasColVector, deltasVector)

diffDenseLayer :: Num a => V.Vector a -> V.Vector a -> M.Matrix a
diffDenseLayer inputs deltas = M.colVector inputs * M.transpose (M.colVector deltas)

diffDenseBias :: V.Vector a -> M.Matrix a
diffDenseBias = M.colVector

data LayerState = MaxPoolingLayerState Image Image | ConvolutionalLayerState Image Image | DenseLayerState (V.Vector Float) (V.Vector Float) | SoftmaxLayerState (V.Vector Float) (V.Vector Float) | EmptyState

forwardDenseLayer (DenseLayer weights bias activation activationDerivative) inputs = let excitationState = denseExcitation (DenseLayer weights bias activation activationDerivative) inputs in
                                                                                let activationState = denseActivation (DenseLayer weights bias activation activationDerivative) inputs in
                                                                                (DenseLayerState inputs excitationState, activationState)
forwardDenseLayer (SoftmaxLayer weights bias) inputs = let excitationState = denseExcitation (SoftmaxLayer weights bias) inputs in
                                                  let activationState = denseActivation (SoftmaxLayer weights bias) inputs in
                                                  (SoftmaxLayerState inputs excitationState, activationState)

forwardTensorialLayer (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputs =
                                                let excitationState = tensorialExcitation (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputs in
                                                let activationState = tensorialActivation (ConvolutionalLayer kernelTensor biasVector activation activationDerivative) inputs in
                                                (ConvolutionalLayerState inputs excitationState, activationState)
forwardTensorialLayer (MaxPoolingLayer supportRows supportCols) inputs =
                                                let activation = tensorialActivation (MaxPoolingLayer supportRows supportCols) inputs in
                                                (MaxPoolingLayerState inputs activation, activation)

forwardTensorialNetworkWithStates :: TensorialNetwork -> Image -> [(LayerState, Image)]
forwardTensorialNetworkWithStates tensorialNetwork image = scanl f (EmptyState, image) tensorialNetwork
                          where f prevLayerState tensorialLayer = let prevActivation = snd prevLayerState in
                                                                  forwardTensorialLayer tensorialLayer prevActivation

forwardDenseNetworkWithState :: DenseNetwork -> V.Vector Float -> [(LayerState, V.Vector Float)]
forwardDenseNetworkWithState denseNetwork input = scanl f (EmptyState, input) denseNetwork
                          where f prevLayerState denseLayer = let prevActivation = snd prevLayerState in
                                                              forwardDenseLayer denseLayer prevActivation

forwardNetworkWithState :: NeuralNetwork -> V.Vector RealMatrix -> ([LayerState], [LayerState])
forwardNetworkWithState (ConvolutionalNetwork tensorialNetwork denseNetwork) image = let tensorialStates = forwardTensorialNetworkWithStates tensorialNetwork image in
                                                                                      let tensorAsVector = (flattenMatrices . snd . last) tensorialStates in
                                                                                      let denseStates = forwardDenseNetworkWithState denseNetwork tensorAsVector in
                                                                                      (fmap fst tensorialStates, fmap fst denseStates)

data BackpropagationResult = EmptyBPResultDense (V.Vector Float) | EmptyBPResultTensorial Image | DenseLayerBPResult (V.Vector Float) (V.Vector Float) | TensorialLayerBPResult Image Image

toVector realMatrix = V.fromList $ M.toList realMatrix

{- TODO: backward dense network -}
backwardDenseNetwork :: DenseNetwork -> [LayerState] -> V.Vector Float -> [BackpropagationResult]
backwardDenseNetwork denseNetwork layerStates dE_dO = let layersPairedWithStates = zip denseNetwork layerStates in
                                                      scanr backpropagationStepDense (EmptyBPResultDense dE_dO) layersPairedWithStates

backpropagationStepDense :: (DenseLayer, LayerState) -> BackpropagationResult -> BackpropagationResult
backpropagationStepDense (denseLayer, denseLayerState) (DenseLayerBPResult curr_dE_dO _) =
                                                                case denseLayerState of
                                                                DenseLayerState input exc -> let (dE_dI, dE_dH) = backwardDenseLayer denseLayer input exc curr_dE_dO in
                                                                                             DenseLayerBPResult (toVector dE_dI) dE_dH
                                                                SoftmaxLayerState input exc -> let (dE_dI, dE_dH) = backwardDenseLayer denseLayer input exc curr_dE_dO in
                                                                                             DenseLayerBPResult (toVector dE_dI) dE_dH
                                                                _ -> error "Bad pairing of layer with state"

backpropagationStepDense _ _ = error "Bad pairing of layer with state"

{- TODO: backward tensorial network -}
backwardTensorialNetwork :: [TensorialLayer] -> [LayerState] -> Image -> [BackpropagationResult]
backwardTensorialNetwork tensorialNetwork layerStates dE_dO = let layersPairedWithStates = zip tensorialNetwork layerStates in
                                                              scanr backpropagationStepTensorial (EmptyBPResultTensorial dE_dO) layersPairedWithStates

backpropagationStepTensorial :: (TensorialLayer, LayerState) -> BackpropagationResult -> BackpropagationResult
backpropagationStepTensorial (tensorialLayer, tensorialLayerState) (TensorialLayerBPResult curr_dE_dO _) =
                                                                case tensorialLayerState of
                                                                  ConvolutionalLayerState input exc -> let (dE_dI, dE_dH) = backwardTensorialLayer tensorialLayer input exc curr_dE_dO in
                                                                                            TensorialLayerBPResult dE_dI dE_dH
                                                                  MaxPoolingLayerState input output -> let (dE_dI, _) = backwardTensorialLayer tensorialLayer input output curr_dE_dO in
                                                                                            TensorialLayerBPResult dE_dI dE_dI
                                                                  _ -> error "Bad pairing of layer with state"

backpropagationStepTensorial _ _ = error "Bad pairing of layer with state"

{- TODO: backward network -}
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

extractLayerExcitation :: LayerState -> Image
extractLayerExcitation (ConvolutionalLayerState _ exc) = exc
extractLayerExcitation _ = error "not implemented"

backpropagationNetwork :: NeuralNetwork -> [LayerState] -> [LayerState] -> V.Vector Float -> ([BackpropagationResult], [BackpropagationResult])
backpropagationNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialLayerStates denseLayerStates dE_dO =
                                                                                        let denseBPResults = backwardDenseNetwork denseNetwork denseLayerStates dE_dO in
                                                                                        let (DenseLayerBPResult dE_dI _) = last denseBPResults in
                                                                                        let next_dE_dO = deflattenToSameDimensionsOf (extractLayerExcitation $ last tensorialLayerStates) dE_dI in
                                                                                        let tensorialBPResults = backwardTensorialNetwork tensorialNetwork tensorialLayerStates next_dE_dO in
                                                                                        (tensorialBPResults, denseBPResults)

{- TODO: dE_dW on whole networks -}
-- data GradientAction = NoAction | ConvolutionalLayerAction Image (V.Vector Float) | DenseLayerAction RealMatrix (V.Vector Float) 

diffTensorialLayer :: TensorialLayer -> Image -> Image -> KernelTensor
diffTensorialLayer (ConvolutionalLayer kernelTensor _ _ _) input deltas = diffKernelTensor (M.nrows $ (V.head . V.head) kernelTensor) (M.ncols $ (V.head . V.head) kernelTensor) input deltas
diffTensorialLayer MaxPoolingLayer {} input deltas = V.empty

diffBiasTensorialLayer :: TensorialLayer -> V.Vector (M.Matrix Float) -> V.Vector Float
diffBiasTensorialLayer ConvolutionalLayer {} deltas = diffKernelBias deltas
diffBiasTensorialLayer MaxPoolingLayer {} deltas = V.empty

diffDenseLayer2 :: DenseLayer -> V.Vector Float -> V.Vector Float -> RealMatrix
diffDenseLayer2 DenseLayer {} input delta = diffDenseLayer input delta
diffDenseLayer2 SoftmaxLayer {} input delta = diffDenseLayer input delta

diffDenseBias2 :: DenseLayer -> V.Vector Float -> RealMatrix
diffDenseBias2 DenseLayer {} delta = diffDenseBias delta
diffDenseBias2 SoftmaxLayer {} delta = diffDenseBias delta

diffDenseLayerWithState :: DenseLayer -> LayerState -> BackpropagationResult -> (RealMatrix, RealMatrix)
diffDenseLayerWithState denseLayer (DenseLayerState input _) (DenseLayerBPResult _ dE_dH) = (diffDenseLayer2 denseLayer input dE_dH, diffDenseBias2 denseLayer dE_dH)
diffDenseLayerWithState _ _ _ = error "Bad pairing of layer state and input"

diffTensorialLayerWithState :: TensorialLayer -> LayerState -> BackpropagationResult -> (KernelTensor, V.Vector Float)
diffTensorialLayerWithState tensorialLayer (ConvolutionalLayerState input _) (TensorialLayerBPResult _ dE_dH) = (diffTensorialLayer tensorialLayer input dE_dH, diffBiasTensorialLayer tensorialLayer dE_dH)
diffTensorialLayerWithState tensorialLayer (MaxPoolingLayerState input _) (TensorialLayerBPResult _ dE_dH) = (diffTensorialLayer tensorialLayer input dE_dH, diffBiasTensorialLayer tensorialLayer dE_dH)
diffTensorialLayerWithState _ _ _ = error "Bad pairing of layer state and input"

diffNetwork :: NeuralNetwork -> [LayerState] -> [LayerState] -> [BackpropagationResult] -> [BackpropagationResult] -> ([(KernelTensor, V.Vector Float)], [(RealMatrix, RealMatrix)])
diffNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialLayerStates denseLayerStates tensorialBPResults denseBPResults =
                                                                                                            let tensorialLayerDiffs = zipWith3 diffTensorialLayerWithState tensorialNetwork tensorialLayerStates tensorialBPResults  in
                                                                                                            let denseLayerDiffs = zipWith3 diffDenseLayerWithState denseNetwork denseLayerStates denseBPResults in
                                                                                                            (tensorialLayerDiffs, denseLayerDiffs)

data GradientAction = TensorialLayerAction KernelTensor (V.Vector Float) | DenseLayerAction RealMatrix RealMatrix

scaleKernelTensor :: Float -> KernelTensor -> KernelTensor
scaleKernelTensor eta = fmap (fmap $ M.scaleMatrix eta)

sumKernelTensors :: KernelTensor -> KernelTensor -> KernelTensor
sumKernelTensors = V.zipWith (V.zipWith (+))

scaleVector :: Float -> V.Vector Float -> V.Vector Float
scaleVector eta = fmap (*eta)

scaleTensorialActions :: Float -> ([KernelTensor], [V.Vector Float]) -> ([KernelTensor], [V.Vector Float])
scaleTensorialActions eta (dE_dKList, dE_dBList) = (fmap (scaleKernelTensor eta) dE_dKList , fmap (scaleVector eta) dE_dBList)

applyPair f (x, y) = (f x, f y)

scaleDenseActions :: Float -> ([RealMatrix], [RealMatrix]) -> ([RealMatrix], [RealMatrix])
scaleDenseActions eta = applyPair (fmap $ M.scaleMatrix eta)

toTensorialActions :: Float -> ([KernelTensor], [V.Vector Float]) -> [GradientAction]
toTensorialActions eta (dE_dKList, dE_dBList) = let (dE_dKList', dE_dBList') = scaleTensorialActions eta (dE_dKList, dE_dBList) in
                                            zipWith TensorialLayerAction dE_dKList' dE_dBList'
toDenseActions :: Float -> ([RealMatrix], [RealMatrix]) -> [GradientAction]
toDenseActions eta (dE_dWList, dE_dBList) = let (dE_dWList', dE_dBList') = scaleDenseActions eta (dE_dWList, dE_dBList) in
                                            zipWith DenseLayerAction dE_dWList' dE_dBList'

sumTensors :: KernelTensor -> KernelTensor -> KernelTensor
sumTensors = V.zipWith (V.zipWith (+))

applyDenseAction :: DenseLayer -> GradientAction -> DenseLayer
applyDenseAction (DenseLayer weights bias act der) (DenseLayerAction dW dB) = DenseLayer (weights + dW) (bias + toVector dB) act der
applyDenseAction (SoftmaxLayer weights bias) (DenseLayerAction dW dB) = SoftmaxLayer (weights + dW) (bias + toVector dB)
applyDenseAction layer _ = error "Bad layer action pairing"

applyTensorialAction :: TensorialLayer -> GradientAction -> TensorialLayer
applyTensorialAction (ConvolutionalLayer kernelTensor biasVector act der) (TensorialLayerAction dK dB) = ConvolutionalLayer (sumTensors kernelTensor dK) (biasVector + dB) act der
applyTensorialAction (MaxPoolingLayer rows cols) _ = MaxPoolingLayer rows cols
applyTensorialAction layer _ = error "Bad layer action pairing"

applyActions :: NeuralNetwork -> [GradientAction] -> [GradientAction] -> NeuralNetwork
applyActions (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialActions denseActions = let nextTensorialNetwork = zipWith applyTensorialAction tensorialNetwork tensorialActions in
                                                                                                  let nextDenseNetwork = zipWith applyDenseAction denseNetwork denseActions in
                                                                                                  ConvolutionalNetwork nextTensorialNetwork nextDenseNetwork



nextNetwork :: Float -> NeuralNetwork -> [LayerState] -> [LayerState] -> V.Vector Float -> NeuralNetwork
nextNetwork eta (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialStates denseStates dE_dO =
                                                                            let (tensorialBPResults, denseBPResults) = backpropagationNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialStates denseStates dE_dO in
                                                                            let (intermediateTActions, intermediateDActions) = diffNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialStates denseStates tensorialBPResults denseBPResults in
                                                                            let (tensorialActions, denseActions) = (toTensorialActions eta $ unzip intermediateTActions, toDenseActions eta $ unzip intermediateDActions) in
                                                                            applyActions (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialActions denseActions