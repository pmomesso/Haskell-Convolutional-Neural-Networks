module Main where

import qualified Layers as L
import qualified Data.Matrix as M
import qualified Data.Vector as V
import System.Random
import System.Environment (getArgs)
import Data.List.Split (splitOn)
import Functions    
import Layers

main :: IO ()
main = do
    networkDescription <- readNetworkDescription
    let (inputDimsString, tensorialTowerStringList, denseTowerStringList) = networkDescription
    let [channels, rows, cols] = parseInputDims inputDimsString

    print networkDescription

parseInputDims :: String -> [Int]
parseInputDims str = atoi <$> splitOn " " (trim $ tail str)

atoi :: String -> Int
atoi = read

readNetworkDescription :: IO (String, [String], [String])
readNetworkDescription = do
    networkDescriptionString              <- getLine
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

randomLayerFromDescr str =
    let layerType = head str in
    let body = trim $ tail str in
    case layerType of
    'C' -> randomConvolutionalLayerFromDescr body
    'P' -> maxPoolingLayerFromDescr body
    _ -> error "Not implemented"

randomConvolutionalLayerFromDescr :: String -> IO TensorialLayer
randomConvolutionalLayerFromDescr body = do
    let splitBody = splitOn "," body
    let [numFilters, channels, rows, cols] = fmap atoi splitBody
    kernelTensorList <- sequence [ randomRealTensor channels rows cols | _ <- [1..numFilters] ]
    let kernelTensorListAsVector = V.fromList kernelTensorList
    let biasVector = V.generate numFilters (const 0)
    return $ ConvolutionalLayer kernelTensorListAsVector biasVector relu dRelu

maxPoolingLayerFromDescr :: String -> IO TensorialLayer
maxPoolingLayerFromDescr body = do
    let splitBody = splitOn "," body
    let [suppRows, suppCols] = fmap atoi splitBody
    return $ MaxPoolingLayer suppRows suppCols