module DenseLayer where

import CommonTypes

import qualified Data.Vector as V
import qualified Data.Matrix as M
import Softmax

data DenseLayer = DenseLayer RealMatrix BiasVector Activation ActivationDerivative 
                 | SoftmaxLayer RealMatrix BiasVector

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

forwardDenseLayer :: DenseLayer -> V.Vector Float -> (LayerState, V.Vector Float)
forwardDenseLayer (DenseLayer weights bias activation activationDerivative) inputs = let excitationState = denseExcitation (DenseLayer weights bias activation activationDerivative) inputs in
                                                                                let activationState = denseActivation (DenseLayer weights bias activation activationDerivative) inputs in
                                                                                (DenseLayerState inputs excitationState, activationState)
forwardDenseLayer (SoftmaxLayer weights bias) inputs = let excitationState = denseExcitation (SoftmaxLayer weights bias) inputs in
                                                  let activationState = denseActivation (SoftmaxLayer weights bias) inputs in
                                                  (SoftmaxLayerState inputs excitationState, activationState)

backwardDenseLayer :: DenseLayer -> V.Vector Float -> V.Vector Float -> V.Vector Float -> (RealMatrix, V.Vector Float)
backwardDenseLayer (DenseLayer weights bias activation activationDerivative) inputVector excitationVec dE_dO =
                                                                                      let deltasVector = activationDerivative <$> excitationVec in
                                                                                      let deltasColVector = elemwiseMult (M.colVector dE_dO) (M.colVector deltasVector) in
                                                                                      (M.transpose weights * deltasColVector, M.getCol 1 deltasColVector)

backwardDenseLayer (SoftmaxLayer weights bias) inputVector excitationVec dE_dO =
                                                                      let deltasVector = softmaxDeltas excitationVec dE_dO in
                                                                      let deltasColVector = M.colVector deltasVector in
                                                                      (M.transpose weights * deltasColVector, deltasVector)

diffDenseLayer :: V.Vector Float -> V.Vector Float -> M.Matrix Float
diffDenseLayer inputs deltas = M.colVector deltas * M.transpose (M.colVector inputs)

diffDenseBias = M.colVector

diffDenseLayer2 :: DenseLayer -> V.Vector Float -> V.Vector Float -> RealMatrix
diffDenseLayer2 DenseLayer {} input delta = diffDenseLayer input delta
diffDenseLayer2 SoftmaxLayer {} input delta = diffDenseLayer input delta

diffDenseBias2 :: DenseLayer -> V.Vector Float -> RealMatrix
diffDenseBias2 DenseLayer {} delta = diffDenseBias delta
diffDenseBias2 SoftmaxLayer {} delta = diffDenseBias delta

diffDenseLayerWithState :: DenseLayer -> LayerState -> BackpropagationResult -> (RealMatrix, RealMatrix)
diffDenseLayerWithState denseLayer (DenseLayerState input _) (DenseLayerBPResult _ dE_dH) = (diffDenseLayer2 denseLayer input dE_dH, diffDenseBias2 denseLayer dE_dH)
diffDenseLayerWithState denseLayer (SoftmaxLayerState input _) (DenseLayerBPResult _ dE_dH) = (diffDenseLayer2 denseLayer input dE_dH, diffDenseBias2 denseLayer dE_dH)
diffDenseLayerWithState x y z = error "Bad pairing of layer state and input"

applyDenseAction :: DenseLayer -> GradientAction -> DenseLayer
applyDenseAction (DenseLayer weights bias act der) (DenseLayerAction dW dB) = DenseLayer (weights + dW) (bias + toVector dB) act der
applyDenseAction (SoftmaxLayer weights bias) (DenseLayerAction dW dB) = SoftmaxLayer (weights + dW) (bias + toVector dB)
applyDenseAction layer _ = error "Bad layer action pairing"