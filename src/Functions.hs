module Functions where

import qualified Data.Vector as V

crossEntropy :: Int -> V.Vector Float -> Float
crossEntropy correctClass probabilityVector = -log (probabilityVector V.! (correctClass-1))

dCrossEntropy :: Int -> V.Vector Float -> V.Vector Float
dCrossEntropy correctClass probabilityVector = V.generate (V.length probabilityVector) (\index -> if index == (correctClass-1) then -1 / (probabilityVector V.! (correctClass-1)) else 0::Float)

relu :: Float -> Float
relu = max 0

indicatorFunction :: (a -> Bool) -> a -> Float
indicatorFunction p x = if p x then 1.0 else 0.0

dRelu :: Float -> Float
dRelu = indicatorFunction (>= 0.0)