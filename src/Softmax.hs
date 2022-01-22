module Softmax where
import qualified Data.Vector as V

import CommonTypes

softmax :: V.Vector Float -> V.Vector Float
softmax vector = let exps = fmap exp vector in
                 let s = sum exps in
                 fmap (/s) exps

softmaxDeltaTerms exps s dE_dO termIndex index = if termIndex == index
                                                 then (((exps V.! termIndex)*s - (exps V.! termIndex)*(exps V.! termIndex))/(s^2)) * (dE_dO V.! termIndex)
                                                 else (-1)*(((exps V.! termIndex)*(exps V.! index))/(s^2)) * (dE_dO V.! index)

softmaxDeltaTerm :: V.Vector Float -> Float -> V.Vector Float -> Int -> Float
softmaxDeltaTerm exps s dE_dO termIndex = sum $ map (softmaxDeltaTerms exps s dE_dO termIndex) [0..(V.length exps-1)]

softmaxDeltas :: V.Vector Float -> V.Vector Float -> V.Vector Float
softmaxDeltas excitationVec dE_dO = let exps = fmap exp excitationVec in
                                    let s = sum exps in
                                    shiftLeft $ V.generate (V.length excitationVec) (softmaxDeltaTerm exps s dE_dO)