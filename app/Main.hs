module Main where

import qualified Layers as L
import qualified Data.Matrix as M
import qualified Data.Vector as V
import System.Random
import System.Environment (getArgs)
import Data.List.Split (splitOn)
import Functions
import Layers
import Control.Monad ( zipWithM )
import Codec.Picture (readImage, Pixel (pixelAt), convertRGB8, DynamicImage (ImageY8), Image (Image, imageHeight, imageWidth), PixelRGB8 (PixelRGB8))
import System.Directory (listDirectory)

type Category = Int
data CategoricalDataPoint a = CategoricalDataPoint a Category
type CategoricalDataset a = [ CategoricalDataPoint a ]

main :: IO ()
main = do
    tensor <- readGrayscaleFromPath "8-bit-256-x-256-Grayscale-Lena-Image.png"
    print $ M.nrows $ V.head tensor

pair :: [a -> b] -> [[a]] -> [[b]]
pair = zipWith fmap

readDataset :: FilePath -> IO [CategoricalDataPoint L.Image]
readDataset dir = do
    classesDirs <- listDirectory dir
    let classes = fmap atoi classesDirs
    imagesNoCategories <- mapM readImagesFromDir classesDirs
    let imagesWithCategories = concat $ pair (fmap (flip CategoricalDataPoint) classes) imagesNoCategories
    return imagesWithCategories

readImagesFromDir :: String -> IO [L.Image]
readImagesFromDir dirPath = do
    paths <- listDirectory dirPath
    mapM readGrayscaleFromPath paths

readNetworkFromStdin :: IO NeuralNetwork
readNetworkFromStdin = do
    networkDescription <- readNetworkDescription
    let (inputDimsString, tensorialTowerStringList, denseTowerStringList) = networkDescription
    let [channels, rows, cols] = parseInputDims inputDimsString
    tensorialNetwork <- mapM randomTensorialLayerFromDescr tensorialTowerStringList
    let (channelsAfter, rowsAfter, colsAfter) = resultingDimensionTensorialNetwork tensorialNetwork (channels, rows, cols)
    let numUnits = channelsAfter * rowsAfter * colsAfter
    let dimensions = numUnits : dimensionFromDescrList denseTowerStringList
    denseNetwork <- zipWithM randomDenseLayerFromDescr dimensions denseTowerStringList
    return $ ConvolutionalNetwork tensorialNetwork denseNetwork

readGrayscaleFromPath :: String -> IO L.Image
readGrayscaleFromPath imagePath = do
    img <- readImage imagePath
    case img of
        Left x -> error $ "Couldn't read " ++ imagePath
        Right x -> (return . imageToSingleChannelTensor) x

imageToSingleChannelTensor :: DynamicImage -> V.Vector RealMatrix
imageToSingleChannelTensor img =
    let toRGB = convertRGB8 img in
    V.fromList [ M.matrix (imageWidth toRGB) (imageHeight toRGB) (\(row, col) -> rgbToGray $ pixelAt toRGB row col) ]

rgbToGray :: PixelRGB8 -> Float
rgbToGray (PixelRGB8 r g b) = ((fromIntegral r :: Float) + (fromIntegral g :: Float) + (fromIntegral b :: Float)) / 3

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
randomFloat =
    randomIO :: IO Float

randomRealMatrix :: Int -> Int -> IO (M.Matrix Float)
randomRealMatrix rows cols = do
    let matrixValues = [ randomFloat | _ <- [1..rows*cols] ]
    matrixValues <- sequence matrixValues
    return (M.fromList rows cols matrixValues)

randomRealTensor :: Int -> Int -> Int -> IO (V.Vector (M.Matrix Float))
randomRealTensor depth rows cols = do
    let realMatrices = [ randomRealMatrix rows cols | _ <- [1..depth] ]
    matrices <- sequence realMatrices
    return $ V.fromList matrices

randomTensorialLayerFromDescr str =
    let layerType = head str in
    let body = trim $ tail str in
    case layerType of
    'C' -> randomConvolutionalLayerFromDescr body
    'P' -> maxPoolingLayerFromDescr body
    _ -> error "Invalid syntax"

randomConvolutionalLayerFromDescr :: String -> IO TensorialLayer
randomConvolutionalLayerFromDescr body = do
    let splitBody = splitOn " " body
    let [numFilters, channels, rows, cols] = fmap atoi splitBody
    kernelTensorList <- sequence [ randomRealTensor channels rows cols | _ <- [1..numFilters] ]
    let kernelTensorListAsVector = V.fromList kernelTensorList
    let biasVector = V.generate numFilters (const 0)
    return $ ConvolutionalLayer kernelTensorListAsVector biasVector relu dRelu

maxPoolingLayerFromDescr :: String -> IO TensorialLayer
maxPoolingLayerFromDescr body = do
    let splitBody = splitOn " " body
    let [suppRows, suppCols] = fmap atoi splitBody
    return $ MaxPoolingLayer suppRows suppCols

dimensionFromDescrList :: [[Char]] -> [Int]
dimensionFromDescrList = fmap (atoi . (head . tail) . splitOn " ")

randomDenseLayerFromDescr :: Int -> String -> IO DenseLayer
randomDenseLayerFromDescr numUnits str = do
    let layerType = head str
    let body = trim $ tail str
    case layerType of
        'D' -> randomFullyConnectedLayer numUnits body
        'S' -> randomSoftmaxLayer numUnits body
        _ -> error "Invalid syntax"

randomFullyConnectedLayer :: Int -> String -> IO DenseLayer
randomFullyConnectedLayer numUnits body = do
    let splitBody = splitOn "," body
    let [ outputDim ] = fmap atoi splitBody
    weightMatrix <- randomRealMatrix outputDim numUnits
    let biasVector = V.generate outputDim (const 0)
    return $ DenseLayer weightMatrix biasVector relu dRelu

randomSoftmaxLayer :: Int -> String -> IO DenseLayer
randomSoftmaxLayer numUnits body = do
    let splitBody = splitOn "," body
    let [ outputDim ] = fmap atoi splitBody
    weightMatrix <- randomRealMatrix outputDim numUnits
    let biasVector = V.generate outputDim (const 0)
    return $ SoftmaxLayer weightMatrix biasVector
