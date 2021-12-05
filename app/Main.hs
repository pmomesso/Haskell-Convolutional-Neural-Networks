module Main where

import qualified Layers as L
import qualified Data.Matrix as M
import qualified Data.Vector as V
import System.Random
import System.Environment (getArgs)
import Data.List.Split (splitOn)

main :: IO ()
main = do
    networkDescription <- readNetworkDescription
    print networkDescription

readNetworkDescription :: IO (String, [String], [String])
readNetworkDescription = do
    networkDescriptionString               <- getLine
    let             networkDescriptionList = splitOn "@" networkDescriptionString
    let inputPlusTensorialTowerDescription = map trim $ splitOn "," $ head networkDescriptionList
    let               imageInputDimensions = head inputPlusTensorialTowerDescription
    let          tensorialTowerDescription = tail inputPlusTensorialTowerDescription
    let              denseTowerDescription = map trim $ splitOn "," $ last networkDescriptionList
    return (imageInputDimensions, tensorialTowerDescription, denseTowerDescription)

trim :: String -> String
trim str = let frontStripped = dropWhile (==' ') str in reverse (dropWhile (==' ') $ reverse frontStripped)

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