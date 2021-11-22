# Sentinel.jl

Note: Installation via Pkg.add("Sentinel") should work on Monday or Tuesday. For now, please use Pkg.add(url="https://github.com/mhudecheck/Sentinel.jl").

Sentinel.jl is a Julia library for working with ESA Sentinel 2 data. It includes functions for finding, downloading, reading, interpolating, and merging Sentinel 2 bands. It also includes functions for creating cloud cover screens by comparing multiple band captures and applying user-specified threshold values. 

This library interfaces extensively with ArchGDAL.jl, GoogleCloud.jl, and - where applicable - CUDA.jl. 

This is the first version of the library. At the moment, it only works with Level 2a files. Sentinel.jl lets you search for and download L2a SAFE file directories from Google Cloud's Sentinel 2 repository, and you can also work with SAFE file directories downloaded outside fo Sentinel.jl from the ESA or Amazon AWS.

Sentinel.jl is currently under development, so expect more features - as well as breaking changes - in the near future. 

A standard workflow looks something like: 
* ID UTM Tile Grid (e.g., with https://eatlas.org.au/data/uuid/f7468d15-12be-4e3f-a246-b2882a324f59)
* Get a list of current L2a products from Google Cloud's Sentinel 2 repository with safeList()
* Screen for your UTM tile, along with a capture date range and - if relevant - a maximum cloud cover threshold, with filterList()
* Download one or more L2a SAFE files with downloadTiles()
* Load one or more SAFE files with loadSentinel()

If you want to create a custom cloud screen composite, you would then:
* Generate and attach individual cloud screens with generateScreens([captureOne, captureTwo, ...])
* Interpolate 20m and 60m bands to 10m, and apply your cloud screens with applyScreens([captureOne, captureTwo, ...])
* Create a composite product by running compositeProduct = Sentinel.generateCloudless(captureOne, captureTwo, ...)
* Save the composite as a GeoTiff with saveScreenedRasters(compositeProduct, names=["composite.tif"])

While most people will perform these functions natively, if you have a relatively modern GPU, you can set GPU = true in many of these functions to receive a 10-15x speedup. Note that only Nvidia GPUs are supported, and you must install CUDA, as well as CUDA.jl, for this to work. Each L2a product takes up ~2gb of RAM, so you'll need a GPU with at least 6gb of memory to load and generate cloud screens for two L2a products. Interpolating and applying the cloud screens will consume significantly more memory, so most users who generate cloud screens with generateScreens(; GPU=true) will want to move their products off of the GPU before running applyScreens(; GPU=false) or generateCloudless(; GPU=false).     

You will need to have a working Google account, as well as credentials for Google Cloud, before you can search for and download SAFE files programmatically. For now, please follow the Google Cloud Prerequisites tutorial from GoogleCloud.jl, which you can read at https://juliacloud.github.io/GoogleCloud.jl/latest/. 

If you want to access the Google Cloud repository, first initialize your credentials by running:
```
using Sentinel.jl
using GoogleCloud.jl 

creds = JSONCredentials("path_to_credentials.json")
session = GoogleSession(creds, ["devstorage.full_control"])
set_session!(storage, session)

# Note that you can also run cloudInit("path_to_credentials.json")
```
 
Then, get a list of Sentinel L2A products with safeList(). Note that you can set a custom cache directory by running safeList(; cacheLocation = "/your_cache_directory"):
```
l2aList = safeList() # Outputs a DataFrame Array
```
You can then run filterList(l2aList, UTM_Tile, startDate, endDate; cloud) to subset the L2A capture list for your targeted UTM tile and date range. You can specify a maximum cloud cover level with filterList(...; cover=10).

```
l2aSubSet = filterList(l2aList, "32TNT", Date(2021,10,01), Date(2021,10,31); cover=5); # Returns a DataFrame with all captures of St. Gallen, CH and the surrounding region in October 2021 that have CLOUD_COVER set to 5 or less.
```

The files in l2aList can be downloaded by running downloadTiles(l2aSubSet, #_of_files; cache="...").
```
captureDirectory = downloadTiles(l2aSubSet, 4)
```

Sentinel 2 SAFE files are loaded with loadSentinel(). You can specify GPU = true to load the individual bands as CuArrays. The output is a dictionary with the bands specified as "BandName-Resolution" (e.g., Band One loads as B1-60m). The band values MetaArrays with relevant band and file metadata from gdalinfo() loaded to file["BandName-Resolution"].metadata and .filedata.

```
cd(captureDirectory)
downloadedFiles = readdir()

# Load to CPU 
cpuOne = loadSentinel(downloadedFiles[1]; GPU=false)
cpuTwo= loadSentinel(downloadedFiles[2]; GPU=false)

```
Cloud screens are created with generateScreens(). It requires two or more SAFE files from the same UTM that preferably have low cloud cover levels. generateScreens() merges the Sen2Cor cloud screens and scene classification matrix provided in the SAFE file by the ESA with band capture deltas (Band 1, Band 2, and Band 4). Screen thresholds can be provided for Bands 1, 2, and 4, as well as the existing cloud screen mask. If the input variable has the bands saved as CuArrays, GPU should be set to true. Note that you will see better results if you use three or more input SAFE files. 

```
fileList = [cpuOne, cpuTwo] 
generateScreens(fileList, GPU=false) 
```

The cloud screens are saved to the input variables as file["CloudScreen"]. The screens can then be applied to the input bands by running applyScreens(). The output bands are saved to the input file as file["Band-Screened"]. Note that all 20m and 60m bands are first interpolated to 10m before the cloud screen is applied. You will also see slightly different results if you run applyScreens() on the CPU or GPU (GPU = true). 

```
applyScreens(fileList, GPU=false) 
```

Cloudless composites are created by running generateCloudless(). 

```
cloudlessFile = applyScreens(fileList, GPU=false) 
```

You can then write the results of applyScreens() to disk as a GeoTiff with named bands with saveScreenedRasters(). saveScreenedRasters can accept multiple inputs (e.g., saveScreenedRasters(cloudlessA, cloudlessB; names=["a.tif", "b.tif"]). 

```
saveScreenedRasters(cloudlessFile; names=["32TNT_Cloudless_October_2021.tif"])
```
