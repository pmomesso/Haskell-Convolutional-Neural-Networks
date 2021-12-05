module Main where

import qualified Layers as L
import qualified Data.Matrix as M
import qualified Data.Vector as V
import System.Random

main :: IO ()
main = do
    let rows = 2
    let cols = 2
    matrix <- randomRealMatrix rows cols
    print matrix

randomFloat :: IO Float
randomFloat = do
    randomIO :: IO Float

randomRealMatrix :: Int -> Int -> IO (M.Matrix Float)
randomRealMatrix rows cols = do
    let matrixValues = [ randomFloat | _ <- [1..rows*cols] ]
    matrixValues <- sequence matrixValues
    print matrixValues
    return (M.fromList rows cols matrixValues)

randomRealTensor :: Int -> Int -> Int -> IO (V.Vector (M.Matrix Float))
randomRealTensor depth rows cols = do
    let realMatrices = [ randomRealMatrix rows cols | _ <- [1..depth] ]
    matrices <- sequence realMatrices
    return $ V.fromList matrices