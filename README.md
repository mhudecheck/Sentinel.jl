# Sentinel.jl

This is an early package for working with Sentinel 2 safe files in Julia. It includes functions for reading Sentinel 2 bands, creating cloud cover screens using multiple captures, interpolating 20m and 60m bands to 10m, and writing the results as a GeoTiff with associated metadata. It also interfaces with CuArrays.jl for GPU processing.

This is the first version of the library. At the moment, it only works with Level 2a files. The SAFE files must also be unzipped.

Sentinel 2 SAFE files are loaded with loadSentinel(). You can specify GPU = true to load the individual bands as CuArrays. The output is a dictionary with the bands specified as "BandName-Resolution" (e.g., Band One loads as B1-60m). The band values MetaArrays with relevant band and file metadata from gdalinfo() loaded to file["BandName-Resolution"].metadata and .filedata.

```
using Sentinel 

# Read Sentinel SAFE Directory
file = "S2A_MSIL2A_20210205T033931_N0214_R061_T48QTF_20210205T071719.SAFE"

# Load to GPU (Requires ~2GB for each capture) 
gpuOne = loadSentinel(file; GPU=true)

# Load to CPU (Requires ~2GB for each capture) 
cpuOne = loadSentinel(file; GPU=false)
```
Cloud screens are created with generateScreens(). It requires two or more SAFE files from the same UTM that preferably have low cloud cover levels. generateScreens() merges the Sen2Cor cloud screens and scene classification matrix provided in the SAFE file by the ESA with band capture deltas (Band 1, Band 2, and Band 4). Sceen thresholds can be provided for Bands 1, 2, and 4, as well as the existing cloud screen mask. If the input variable has the bands saved as CuArrays, GPU should be set to true. Note that you will see better results if you use three or more input SAFE files. 

```
secondFile = "S2A_MSIL2A_20210206T033931_N0214_R061_T48QTF_20210205T071719.SAFE"
gpuTwo = loadSentinel(secondFile; GPU=true)
fileList = [gpuOne, gpuTwo] 
generateScreens(fileList, GPU=true) 
```

The cloud screens are saved to the input variables as file["CloudScreen"]. The screens can then be applied to the input bands by running applyScreens(). The output bands are saved to the input file as file["Band-Screened"]. Note that all 20m and 60m bands are first interpolated to 10m before the cloud screen is applied. You will also see slightly different results if you run applyScreens() on the CPU or GPU (GPU = true). 

```
applyScreens(fileList, GPU=true) 
```

Cloudless composites are created by running generateCloudless(). 

```
cloudlessFile = applyScreens(fileList, GPU=true) 
```

You can then write the results of applyScreens() to disk as a GeoTiff with named bands with saveScreenedRasters(). saveScreenedRasters can accept multiple inputs (e.g., saveScreenedRasters(cloudlessA, cloudlessB; names=["a.tif", "b.tif"]). 

```
saveScreenedRasters(cloudlessFile; names=["UTM42.tif"])
```
