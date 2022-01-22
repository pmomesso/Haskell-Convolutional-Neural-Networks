module Dataset where

type Category = Int
data CategoricalDataPoint a = CategoricalDataPoint a Category

type CategoricalDataset a = [ CategoricalDataPoint a ]

extractCat :: CategoricalDataPoint a -> Category
extractCat (CategoricalDataPoint img cat) = cat

extractInput :: CategoricalDataPoint a -> a
extractInput (CategoricalDataPoint img cat) = img