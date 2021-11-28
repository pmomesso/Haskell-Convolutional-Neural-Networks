module Examples where

import qualified Layers as L
import qualified Data.Matrix as M
import qualified Data.Vector as V
import Layers (TensorialLayer(MaxPoolingLayer))

matrix1 = M.matrix 6 6 (\(i, j) -> fromIntegral(i `min` j)::Float)
matrix2 = M.setElem 100.0 (2,1) matrix1

poolingLayer = MaxPoolingLayer 2 2