module Dataset where

type Category = Int
data CategoricalDataPoint a = CategoricalDataPoint a Category
type CategoricalDataset a = [ CategoricalDataPoint a ]