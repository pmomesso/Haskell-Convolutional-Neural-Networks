module Main where

import qualified Layers as L
import qualified Data.Matrix as M
import System.Random

main :: IO ()
main = do {
    putStrLn "Hola"
}

randomFloat :: IO Float
randomFloat = do
    randomIO :: IO Float
