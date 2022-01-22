module Network where

import qualified Data.Vector as V
import qualified Data.Matrix as M
import Dataset
import CommonTypes
import DenseLayer
import TensorialLayer
    ( TensorialLayer,
      resultingDimension,
      tensorialActivation,
      backwardTensorialLayer,
      forwardTensorialLayer,
      applyTensorialAction,
      diffTensorialLayerWithState )

type DenseNetwork = [ DenseLayer ]
type TensorialNetwork = [ TensorialLayer ]
data NeuralNetwork = ConvolutionalNetwork TensorialNetwork DenseNetwork

extractTensorial (ConvolutionalNetwork tn _) = tn
extractDense (ConvolutionalNetwork _ dn) = dn

backpropagationStepDense :: (DenseLayer, LayerState) -> BackpropagationResult -> BackpropagationResult
backpropagationStepDense (denseLayer, denseLayerState) (DenseLayerBPResult curr_dE_dO _) =
                                                                case denseLayerState of
                                                                DenseLayerState input exc -> let (dE_dI, dE_dH) = backwardDenseLayer denseLayer input exc curr_dE_dO in
                                                                                             DenseLayerBPResult (toVector dE_dI) dE_dH
                                                                SoftmaxLayerState input exc -> let (dE_dI, dE_dH) = backwardDenseLayer denseLayer input exc curr_dE_dO in
                                                                                             DenseLayerBPResult (toVector dE_dI) dE_dH
                                                                _ -> error "Bad pairing of layer with state"

backpropagationStepDense (denseLayer, denseLayerState) (EmptyBPResultDense curr_dE_dO) = case denseLayerState of
                                                                DenseLayerState input exc -> let (dE_dI, dE_dH) = backwardDenseLayer denseLayer input exc curr_dE_dO in
                                                                                             DenseLayerBPResult (toVector dE_dI) dE_dH
                                                                SoftmaxLayerState input exc -> let (dE_dI, dE_dH) = backwardDenseLayer denseLayer input exc curr_dE_dO in
                                                                                             DenseLayerBPResult (toVector dE_dI) dE_dH
                                                                _ -> error "Bad pairing of layer with state"

backpropagationStepDense _ _ = error "Bad pairing of layer with state"

scaleTensorialActions :: Float -> ([KernelTensor], [V.Vector Float]) -> ([KernelTensor], [V.Vector Float])
scaleTensorialActions eta (dE_dKList, dE_dBList) = (fmap (scaleKernelTensor eta) dE_dKList , fmap (scaleVector eta) dE_dBList)

scaleDenseActions :: Float -> ([RealMatrix], [RealMatrix]) -> ([RealMatrix], [RealMatrix])
scaleDenseActions eta = applyPair (fmap $ M.scaleMatrix eta)

toTensorialActions :: Float -> ([KernelTensor], [V.Vector Float]) -> [GradientAction]
toTensorialActions eta (dE_dKList, dE_dBList) = let (dE_dKList', dE_dBList') = scaleTensorialActions eta (dE_dKList, dE_dBList) in
                                            zipWith TensorialLayerAction dE_dKList' dE_dBList'
toDenseActions :: Float -> ([RealMatrix], [RealMatrix]) -> [GradientAction]
toDenseActions eta (dE_dWList, dE_dBList) = let (dE_dWList', dE_dBList') = scaleDenseActions eta (dE_dWList, dE_dBList) in
                                            zipWith DenseLayerAction dE_dWList' dE_dBList'

backpropagationStepTensorial :: (TensorialLayer, LayerState) -> BackpropagationResult -> BackpropagationResult
backpropagationStepTensorial (tensorialLayer, tensorialLayerState) (TensorialLayerBPResult curr_dE_dO _) =
                                                                case tensorialLayerState of
                                                                  ConvolutionalLayerState input exc -> let (dE_dI, dE_dH) = backwardTensorialLayer tensorialLayer input exc curr_dE_dO in
                                                                                            TensorialLayerBPResult dE_dI dE_dH
                                                                  MaxPoolingLayerState input output -> let (dE_dI, _) = backwardTensorialLayer tensorialLayer input output curr_dE_dO in
                                                                                            TensorialLayerBPResult dE_dI dE_dI
                                                                  _ -> error "Bad pairing of layer with state"
backpropagationStepTensorial (tensorialLayer, tensorialLayerState) (EmptyBPResultTensorial curr_dE_dO) =
                                                                  case tensorialLayerState of
                                                                  ConvolutionalLayerState input exc -> let (dE_dI, dE_dH) = backwardTensorialLayer tensorialLayer input exc curr_dE_dO in
                                                                                            TensorialLayerBPResult dE_dI dE_dH
                                                                  MaxPoolingLayerState input output -> let (dE_dI, _) = backwardTensorialLayer tensorialLayer input output curr_dE_dO in
                                                                                            TensorialLayerBPResult dE_dI dE_dI
                                                                  _ -> error "Bad pairing of layer with state"
backpropagationStepTensorial _ _ = error "Bad pairing of layer with state"

resultingDimensionTensorialNetwork :: TensorialNetwork -> (Int, Int, Int) -> (Int, Int, Int)
resultingDimensionTensorialNetwork network (channels, rows, cols) = foldl (flip resultingDimension) (channels, rows, cols) network

forwardConvNetwork :: TensorialNetwork -> Image -> Image
forwardConvNetwork convNet image = foldl tensorialActivation' image convNet
                                   where tensorialActivation' = flip tensorialActivation

forwardDenseNetwork :: DenseNetwork -> V.Vector Float -> V.Vector Float
forwardDenseNetwork denseNetwork x = foldl denseActivation' x denseNetwork
                                     where denseActivation' = flip denseActivation

forwardNeuralNetwork :: NeuralNetwork -> Image -> V.Vector Float
forwardNeuralNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) image = let finalTensor = forwardConvNetwork tensorialNetwork image in
                                                                                  let flattenedTensor = flattenMatrices finalTensor in
                                                                                  forwardDenseNetwork denseNetwork flattenedTensor

backwardTensorialNetwork :: [TensorialLayer] -> [LayerState] -> Image -> [BackpropagationResult]
backwardTensorialNetwork tensorialNetwork layerStates dE_dO = let layersPairedWithStates = zip tensorialNetwork layerStates in
                                                              init $ scanr backpropagationStepTensorial (EmptyBPResultTensorial dE_dO) layersPairedWithStates

backwardDenseNetwork :: DenseNetwork -> [LayerState] -> V.Vector Float -> [BackpropagationResult]
backwardDenseNetwork denseNetwork layerStates dE_dO = let layersPairedWithStates = zip denseNetwork layerStates in
                                                      init $ scanr backpropagationStepDense (EmptyBPResultDense dE_dO) layersPairedWithStates

forwardTensorialNetworkWithStates :: TensorialNetwork -> Image -> [(LayerState, Image)]
forwardTensorialNetworkWithStates tensorialNetwork image = tail $ scanl f (EmptyState, image) tensorialNetwork
                          where f prevLayerState tensorialLayer = let prevActivation = snd prevLayerState in
                                                                  forwardTensorialLayer tensorialLayer prevActivation

forwardDenseNetworkWithState :: DenseNetwork -> V.Vector Float -> [(LayerState, V.Vector Float)]
forwardDenseNetworkWithState denseNetwork input = tail $ scanl f (EmptyState, input) denseNetwork
                          where f prevLayerState denseLayer = let prevActivation = snd prevLayerState in
                                                              forwardDenseLayer denseLayer prevActivation

forwardNetworkWithState :: NeuralNetwork -> Image -> ([LayerState], [LayerState], V.Vector Float)
forwardNetworkWithState (ConvolutionalNetwork tensorialNetwork denseNetwork) image = let tensorialStates = forwardTensorialNetworkWithStates tensorialNetwork image in
                                                                                      let tensorAsVector = (flattenMatrices . snd . last) tensorialStates in
                                                                                      let denseStates = forwardDenseNetworkWithState denseNetwork tensorAsVector in
                                                                                      let probabilityVector = (snd . last) denseStates in
                                                                                      (fmap fst tensorialStates, fmap fst denseStates, probabilityVector)

extractLayerExcitation :: LayerState -> Image
extractLayerExcitation (ConvolutionalLayerState _ exc) = exc
extractLayerExcitation (MaxPoolingLayerState input output) = output
extractLayerExcitation _ = error "not implemented"

backpropagationNetwork :: NeuralNetwork -> [LayerState] -> [LayerState] -> V.Vector Float -> ([BackpropagationResult], [BackpropagationResult])
backpropagationNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialLayerStates denseLayerStates dE_dO =
                                                                                        let denseBPResults = backwardDenseNetwork denseNetwork denseLayerStates dE_dO in
                                                                                        let (DenseLayerBPResult dE_dI _) = head denseBPResults in
                                                                                        let next_dE_dO = deflattenToSameDimensionsOf (extractLayerExcitation $ last tensorialLayerStates) dE_dI in
                                                                                        let tensorialBPResults = backwardTensorialNetwork tensorialNetwork tensorialLayerStates next_dE_dO in
                                                                                        (tensorialBPResults, denseBPResults)

applyActions :: NeuralNetwork -> [GradientAction] -> [GradientAction] -> NeuralNetwork
applyActions (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialActions denseActions = let nextTensorialNetwork = zipWith applyTensorialAction tensorialNetwork tensorialActions in
                                                                                                  let nextDenseNetwork = zipWith applyDenseAction denseNetwork denseActions in
                                                                                                  ConvolutionalNetwork nextTensorialNetwork nextDenseNetwork

diffNetwork :: NeuralNetwork -> [LayerState] -> [LayerState] -> [BackpropagationResult] -> [BackpropagationResult] -> ([(KernelTensor, V.Vector Float)], [(RealMatrix, RealMatrix)])
diffNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialLayerStates denseLayerStates tensorialBPResults denseBPResults =
                                                                                                            let tensorialLayerDiffs = zipWith3 diffTensorialLayerWithState tensorialNetwork tensorialLayerStates tensorialBPResults  in
                                                                                                            let denseLayerDiffs = zipWith3 diffDenseLayerWithState denseNetwork denseLayerStates denseBPResults in
                                                                                                            (tensorialLayerDiffs, denseLayerDiffs)

nextNetwork :: Float -> NeuralNetwork -> [LayerState] -> [LayerState] -> V.Vector Float -> NeuralNetwork
nextNetwork eta (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialStates denseStates dE_dO =
                                                                            let (tensorialBPResults, denseBPResults) = backpropagationNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialStates denseStates dE_dO in
                                                                            let (intermediateTActions, intermediateDActions) = diffNetwork (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialStates denseStates tensorialBPResults denseBPResults in
                                                                            let (tensorialActions, denseActions) = (toTensorialActions eta $ unzip intermediateTActions, toDenseActions eta $ unzip intermediateDActions) in
                                                                            applyActions (ConvolutionalNetwork tensorialNetwork denseNetwork) tensorialActions denseActions

trainClassificationNetwork :: NeuralNetwork -> Float -> CategoricalDataset Image -> (Category -> V.Vector Float -> Float) -> (Category -> V.Vector Float -> V.Vector Float) -> [(Float, NeuralNetwork)]
trainClassificationNetwork network eta dataset errorFunction dErrorFunction =
                              let (tensorialStates, denseStates, probabilityVector) = forwardNetworkWithState network (extractInput $ head dataset) in
                              let errorValue = errorFunction (extractCat $ head dataset) probabilityVector in
                              let n = scanl (trainingStep eta errorFunction dErrorFunction) (errorValue, network) (tail dataset) in
                              n

trainingStep :: Float -> (Category -> V.Vector Float -> Float) -> (Category -> V.Vector Float -> V.Vector Float) -> (Float, NeuralNetwork) -> CategoricalDataPoint Image -> (Float, NeuralNetwork)
trainingStep eta errorFunction dErrorFunction neuralNetwork (CategoricalDataPoint img correctCat) =
                                let (tensorialStates, denseStates, probabilityVector) = forwardNetworkWithState (snd neuralNetwork) img in
                                let errorValue = errorFunction correctCat probabilityVector in
                                let dE_dO = dErrorFunction correctCat probabilityVector in
                                (errorValue, nextNetwork eta (snd neuralNetwork) tensorialStates denseStates dE_dO)