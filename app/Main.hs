module Main where

import qualified Data.Matrix as M
import qualified Data.Vector as V
import System.Random
import System.Environment (getArgs)
import Data.List.Split (splitOn)
import Functions
import Control.Monad ( zipWithM )
import Codec.Picture (readImage, Pixel (pixelAt), convertRGB8, DynamicImage (ImageY8), Image (Image, imageHeight, imageWidth), PixelRGB8 (PixelRGB8))
import System.Directory (listDirectory)
import Control.Applicative (Applicative(liftA2))

import Dataset
import Network
import DenseLayer
import TensorialLayer
import CommonTypes
import qualified CommonTypes as CT

sampleSize = 100
eta :: Float
eta = 1e-2

forwardNetworkWithState2 (ConvolutionalNetwork tensorialNetwork denseNetwork) image = let tensorialStates = forwardTensorialNetworkWithStates tensorialNetwork image in
                                                                                      tensorialStates

main :: IO ()
main = do
    network <- readNetworkFromStdin
    dataset <- readDataset "./dataset"
    sampledDataset <- sampleDataSet dataset sampleSize
    let trainingHistory = trainClassificationNetwork network eta sampledDataset crossEntropy dCrossEntropy
    print $ trainingErrorsList trainingHistory
    -- print $ map extractCategory sampledDataset
    putStr $ unlines (zipWith (curry show) (map extractCat dataset) (map (forwardNeuralNetwork ((snd . last) trainingHistory) . extractInput) dataset))
    return ()

trainingErrorsList :: [(a, b)] -> [a]
trainingErrorsList = map fst

sampleDataSet dataset sampleSize = shuffle sampleSize dataset

sampleDataSetFromPath :: FilePath -> Int -> IO [CategoricalDataPoint CT.Image]
sampleDataSetFromPath path sampleSize = readDataset path >>= shuffle sampleSize

pair :: [a -> b] -> [[a]] -> [[b]]
pair = zipWith fmap

readDataset :: FilePath -> IO [CategoricalDataPoint CT.Image]
readDataset dir = do
    let cleanDir = if last dir == '/' then dir else dir ++ "/"
    classesDirs <- listDirectory cleanDir
    let classes = fmap atoi classesDirs
    let classesPaths = fmap ((++"/") . (cleanDir++)) classesDirs
    imagesNoCategories <- mapM readImagesFromDir classesPaths
    let imagesWithCategories = concat $ pair (fmap (flip CategoricalDataPoint) classes) imagesNoCategories
    return imagesWithCategories

readImagesFromDir :: String -> IO [CT.Image]
readImagesFromDir dirPath = do
    paths <- listDirectory dirPath
    let cleanPaths = fmap (dirPath++) paths
    mapM readGrayscaleFromPath cleanPaths

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

readGrayscaleFromPath :: String -> IO CT.Image
readGrayscaleFromPath imagePath = do
    img <- readImage imagePath
    case img of
        Left x -> error $ "Couldn't read " ++ imagePath
        Right x -> (return . imageToSingleChannelTensor) x

imageToSingleChannelTensor :: DynamicImage -> V.Vector RealMatrix
imageToSingleChannelTensor img =
    let toRGB = convertRGB8 img in
    V.fromList [ M.matrix (imageHeight toRGB) (imageWidth toRGB) (\(row, col) -> rgbToGray (pixelAt toRGB (col-1) (row-1)) / 255.0 )]

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

shuffle :: Int -> [a] -> IO [a]
shuffle nums list =
    if nums == 0
    then return []
    else fmap singleton (pick list) `liftedConcat` shuffle (nums - 1) list

liftedConcat :: Monad m => m [a] -> m [a] -> m [a]
liftedConcat = liftA2 (++)

singleton :: a -> [a]
singleton x = [x]

pick :: [a] -> IO a
pick list = do
    index <- readRandomIndex list
    return $ list !! index

readRandomIndex :: [a] -> IO Int
readRandomIndex list = do
    randomInt <- randomIO :: IO Int
    return $ randomInt `mod` length list

randomRealMatrix :: Int -> Int -> IO (M.Matrix Float)
randomRealMatrix rows cols = do
    matrixValues <- readRandomFloatList $ rows * cols
    return (M.fromList rows cols matrixValues)

readRandomFloatList :: Int -> IO [Float]
readRandomFloatList n = randomListOf n randomFloat

randomListOf :: Int -> IO a -> IO [a]
randomListOf length generationAction = sequence [ generationAction | _ <- [1..length] ]

randomRealTensor :: Int -> Int -> Int -> IO (V.Vector (M.Matrix Float))
randomRealTensor depth rows cols = do
    matricesList <- randomListOf depth $ randomRealMatrix rows cols
    let scaled = fmap (fmap (/(fromIntegral (depth*rows*cols) :: Float))) matricesList
    return $ V.fromList scaled

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
    -- biasFloats <- readRandomFloatList numFilters
    let biasVector = V.generate numFilters (const 0) --V.fromList (fmap (\b -> b*0.2 - 0.1) biasFloats)
    return $ ConvolutionalLayer kernelTensorListAsVector biasVector relu dRelu

maxPoolingLayerFromDescr :: String -> IO TensorialLayer
maxPoolingLayerFromDescr body = do
    let splitBody = splitOn " " body
    let [suppRows, suppCols] = fmap atoi splitBody
    return $ MaxPoolingLayer suppRows suppCols

dimensionFromDescrList :: [String] -> [Int]
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
    let xavierConstant = sqrt 6 / (sqrt (fromIntegral outputDim :: Float) + sqrt (fromIntegral numUnits :: Float))
    let scaled = fmap (\w -> (2*w - 1)*xavierConstant) weightMatrix
    biasFloats <- readRandomFloatList outputDim
    let biasVector = V.generate outputDim (const 0)
    return $ DenseLayer scaled biasVector relu dRelu

randomSoftmaxLayer :: Int -> String -> IO DenseLayer
randomSoftmaxLayer numUnits body = do
    let splitBody = splitOn "," body
    let [ outputDim ] = fmap atoi splitBody
    weightMatrix <- randomRealMatrix outputDim numUnits
    let xavierConstant = sqrt 6 / (sqrt (fromIntegral outputDim :: Float) + sqrt (fromIntegral numUnits :: Float))
    let scaled = fmap (\w -> (2*w - 1)*xavierConstant) weightMatrix
    biasFloats <- readRandomFloatList outputDim
    let biasVector =  V.generate outputDim (const 0) --V.fromList (fmap (\b -> b*0.2 - 0.1) biasFloats)
    return $ SoftmaxLayer scaled biasVector
