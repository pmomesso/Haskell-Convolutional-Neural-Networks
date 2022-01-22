module Examples where

import qualified Layers as L
import qualified Data.Matrix as M
import qualified Data.Vector as V
import TensorialLayer

matrix1 = M.matrix 6 6 (\(i, j) -> fromIntegral(i + j)::Float)
matrix2 = M.setElem 100.0 (2,1) matrix1
ones_6x6 = M.matrix 6 6 (const (1::Float))

m1 = V.fromList [matrix1]
m2 = V.fromList [matrix2]

singleChannelKernel_3x3 = M.matrix 3 3 (const (1::Float))

poolingLayer = MaxPoolingLayer 2 2

poolingLayerActivation = tensorialActivation poolingLayer (V.fromList [matrix2])

dE_dH_3x3 = V.fromList [M.matrix 3 3 (const (10::Float))]
dE_dH_4x4 = V.fromList [M.matrix 4 4 (const (10::Float))]