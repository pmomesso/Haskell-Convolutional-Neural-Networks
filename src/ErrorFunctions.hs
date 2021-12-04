module ErrorFunctions where

import qualified Data.Vector as V

crossEntropy :: Int -> V.Vector Float -> Float
crossEntropy correctClass probabilityVector = -log (probabilityVector V.! correctClass)

dCrossEntropy :: Int -> V.Vector Float -> V.Vector Float
dCrossEntropy correctClass probabilityVector = V.generate (V.length probabilityVector) (\index -> if index == correctClass then -1 / (probabilityVector V.! correctClass) else 0::Float)