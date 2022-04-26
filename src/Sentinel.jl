
module Sentinel
using CUDA
using ArchGDAL
using Base
using NamedArrays
using Metadata
using Interpolations
using Graphics
using Images
using GoogleCloud
using CSV
using CSVFiles
using DataFrames
using JSON
using Printf
using LightXML
using Base64
using Glob
using Dates
using LibGEOS
using Adapt
using KernelDensity
using Turing
using TiledIteration
using HTTP
using Statistics
using NCDatasets

export extractSentinelFive, buildR, buildS, processSentinelFiveTifs, createSentinelFiveTif, modifyVRT, sentinelFiveList, resizeCuda, resizeRaster, removeNaN, rastNormMean, normalizeRasters, scanRastMean, rastMean, generateArea, compareAreas, extractNoData, safeCoverage, mergeSAFE, migrateSafe, resizeCuda, flushSAFE, linearKernel, loadSentinel, scanInvert, cudaScan, cudaRevScan, cudaCirrusScan, sentinelCloudScreen, generateScreens, applyScreens, saveScreenedRasters, cloudInit, safeList, filterList, loadSentinel, loadRaster, extractSAFEGeometries, generateSAFEPath, sortSAFE, qc, generateCloudless

    function buildR(i; sStart=1, sEnd = 8)
        date = SubString.(i, sStart, sEnd)
        rString = "$date:./$i"
        return rString
    end

    function buildS(i; sStart=1, sEnd = 8)
        date = SubString.(i, sStart, sEnd)
        return date
    end

    function extractSentinelFive(targetDir, shapeFileDirectory, outputDirectory; globString = "20*", oRange = -1)
        cwd = pwd()
        cd(targetDir)
        origList = glob(globString)
        @info length(origList)

        if oRange != -1
            origList = origList[1:oRange]
        end

        rList = buildR.(origList);
        sList = buildS.(origList);

        l0Extract = `exactextract -r $rList -p $shapeFileDirectory/gadm36_0.shp -s mean"("$sList")" -o $outputDirectory/l0_means.csv -f NAME_0 --progress`
        l1Extract = `exactextract -r $rList -p $shapeFileDirectory/gadm36_1.shp -s mean"("$sList")" -o $outputDirectory/l1_means.csv -f NAME_1 --progress`
        l2Extract = `exactextract -r $rList -p $shapeFileDirectory/gadm36_2.shp -s mean"("$sList")" -o $outputDirectory/l2_means.csv -f NAME_2 --progress`
        l3Extract = `exactextract -r $rList -p $shapeFileDirectory/gadm36_3.shp -s mean"("$sList")" -o $outputDirectory/l3_means.csv -f NAME_3 --progress`
        @info 1
        try 
            run(l0Extract); 
        catch
        end
        @info 2
        try 
            run(l1Extract); 
        catch
        end       
        @info 3
        try 
            run(l2Extract); 
        catch
        end        
        @info 4
        try 
            run(l3Extract); 
        catch
        end        
        cd(cwd)
    end

    function createSentinelFiveTif(input, output; myId = 1, cacheDir=pwd() * "/.cache/", productName = "nitrogendioxide_tropospheric_column", qcVal = 50)
        # Create Cache
        mkpath(cacheDir)
        j = string(myId, "_temp.nc") # Sets ID for multithreaded use 
        jWd = string(cacheDir, j)
        @info jWd
        try
            # Copy File to Temp Folder
            cp(input, jWd, force=true)
            iSize = filesize(input)
            jSize = filesize(jWd)
            println("$iSize, $jSize")
        
            # Build Filenames & ID Strings
            lonString = string("HDF5:\"", jWd, "\"://PRODUCT/longitude")
            latString = string("HDF5:\"", jWd, "\"://PRODUCT/latitude")
            #dataString = string("HDF5:\"", jWd, "\"://PRODUCT/qaNTC")
            dataString = string("HDF5:\"", jWd, "\"://PRODUCT/", productName)

            # Get X and Y Dimension Lengths
            @time datasetTemp = Dataset(jWd, "a")
            ySize = length(datasetTemp.group["PRODUCT"]["scanline"].var)
            xSize = length(datasetTemp.group["PRODUCT"]["ground_pixel"].var)

            # Get Values
            dt = datasetTemp.group["PRODUCT"]
            qc =  datasetTemp.group["PRODUCT"]["qa_value"].var[:,:]
            vals = datasetTemp.group["PRODUCT"][productName].var[:,:]

            # Clean Values
            qcVec = qc .>= qcVal # Quality control screen
            @time qcVals = vals .* qcVec
            @time qaNTC = defVar(datasetTemp.group["PRODUCT"], "qaNTC", Float32, ("ground_pixel", "scanline"))
            qaNTC[:, :, 1] = qcVals

            # Close Dataset
            close(datasetTemp)
            datasetTemp = nothing
            qcVals = nothing
            qaNTC = nothing
            qcVals = nothing
            qcVec = nothing
            qc = nothing
            vals = nothing
            dt = nothing
            ### Gdal
            ## Build VRTs
            # Longitude
            #print("Building VRTs for $i", "\n")
            lonVrt = string(myId, "_lon.vrt")
            @info pkgdir(Sentinel)
            modifyVRT(pkgdir(Sentinel) * "/src/vrt/lonTemplate.vrt", lonVrt, lonString, xSize, ySize)

            # Latitude
            latVrt = string(myId, "_lat.vrt")
            modifyVRT(pkgdir(Sentinel) * "/src/vrt/latTemplate.vrt", latVrt, latString, xSize, ySize)

            # Data
            dataVrt = string(myId, "_data.vrt")
            modifyVRT(pkgdir(Sentinel) * "/src/vrt/dataTemplate.vrt", dataVrt, dataString, xSize, ySize, lonVrt, latVrt)

            ## Convert to GeoTIFF
            #print("Building TIF for $i", "\n")
            aggr = `gdalwarp -geoloc -t_srs EPSG:4326 -srcnodata 9.96921e+36f $dataVrt $output -tr 0.069 0.069 -tap -r average`
            @time run(aggr)
            rm(lonVrt)
            rm(latVrt)
            rm(dataVrt)
            rm(jWd)
            jWd = nothing
        catch err
            @info err
            #print("Error file $j, original file $i, $err", "\n")
            rm(jWd)
        end
    end

    function processSentinelFiveTifs(date, inputDirectory, outputDirectory; naVal = 9000, throttleVal = 1, searchString = "S5P_*_L2_*")
        cwd = pwd()
        cd(inputDirectory)
        tifList = glob(searchString * "$date*")
        merge = `gdalbuildvrt -srcnodata 9.969209968386869e+36 raster_$date.vrt $tifList`
        run(merge)
        @info 2

        # Create Results TIF
        outputTif = outputDirectory * "/" * date * ".tif"
        outputThrottledTif = outputDirectory * "/throttled_" * date * ".tif"
        aggr = `gdal_translate -r average raster_$date.vrt $outputTif`
        run(aggr)
        @info 3
        # Clean Results and Save to Second Tif
        cd(outputDirectory)
        try
            @info pwd(), "$date.tif"
            ArchGDAL.read("$date.tif") do dataset
                band1 = ArchGDAL.getband(dataset,1)
                raster = ArchGDAL.create(outputThrottledTif, driver=ArchGDAL.getdriver("GTiff"),  width=ArchGDAL.width(band1), height=ArchGDAL.height(band1), nbands=1, dtype=Float32)
    
                # Process Dataset
                data = ArchGDAL.read(band1)
                data[data.>= naVal] .= 0.0 # Clear NA Values
                data[data.<=0.0] .= 0.0 # Clean Negative Values
                data[data.>= throttleVal] .= throttleVal # Throttle Extraneous Results
                ref = ArchGDAL.getproj(dataset)
                geotransform = ArchGDAL.getgeotransform(dataset)
                ArchGDAL.setgeotransform!(raster, geotransform)
                ArchGDAL.setproj!(raster, ref)
    
                ## write the raster
                ArchGDAL.write!(raster, data, 1)
                ArchGDAL.destroy(raster)
            end
            cd(cwd)
        catch err
            @error err
            cd(cwd)
        end
    end

    """
    resizeCuda(input::CuArray, width::Integer, height::Integer; returnGPU::Bool, interpolation::Bool, interpolationType::String)

    Resizes CuArrays to specificed input widths and heights. CuArrays are converted to CuTextureArrays, which allows for linear and nearest neighbor interpolations. 

    # Arguments
        - `input::CuArray`: Input CuArray to be resized
        - `width::Integer`: Output array width
        - `height::Integer`: Output array height
        - `returnGPU::Bool`: Specify whether the function should return an Array (false) or CuArray (true)
        - `interpolation::Bool`: Specify whether the resized output array should be smoothed 
        - `smoothing::String`: If interpolation == true, you can select whether to apply a linear smoothing function (default) with "linear" or a nearest neighbor smoothing function if smoothing != "linear".
    """
    function resizeCuda(input::CuArray, width::Integer, height::Integer; returnGPU::Bool = false, interpolation::Bool = true, smoothing::String = "linear")
        textureArray = CuTextureArray(input)
        if interpolation == true
            if smoothing != "linear"
                textureType = CUDA.NearestNeighbour()
            else 
                textureType = CUDA.LinearInterpolation()
            end 
            texture = CuTexture(textureArray; normalized_coordinates=true, interpolation=textureType)
        else 
            texture = CuTexture(textureArray; normalized_coordinates=true)
        end
        outputCUDAArray = CuArray{eltype(input)}(undef, width, height)
        k = @cuda launch=false linearKernel(outputCUDAArray, texture)
        config = launch_configuration(k.fun)
        threads = Base.min(length(outputCUDAArray), config.threads)
        blocks = cld(length(outputCUDAArray), threads)
        @cuda threads=threads blocks=blocks linearKernel(outputCUDAArray, texture)
        if returnGPU == false
            outputArray = Array{eltype(input)}(outputCUDAArray)
            return outputArray
        else 
            return outputCUDAArray
        end
    end

    function linearKernel(output, input)
        tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        I = CartesianIndices(output)
        @inbounds if tid <= length(I)
            i,j = Tuple(I[tid])
            u = Float64(i-1) / Float64(size(output, 1)-1)
            v = Float64(j-1) / Float64(size(output, 2)-1)
            x = u
            y = v
            output[i,j] = input[x,y]
        end
        return
    end
    
    """
    resizeRaster(raster::AbstractArray, targetWidth::Integer, targetHeight::Integer; gpu::Bool)
    Resizes input arrays. When gpu == false or when the input raster is not a CuArray, resizeRaster() acts as wrapper for ImageTransformations.imresize().
    When gpu == true and the input raster is a CuArray, resizeRaster() acts as a memory-safe front end for resizeCuda(). 

    # Arguments
    - `raster::AbstractArray`: Input array
    - `targetWidth::Integer`: Output array width
    - `targetHeight::Integer`: Output array height
    - `gpu::Bool`: Specifies whether raster should be resized with ImageTransformations.imresize() (false) or Sentinel.resizeCuda() true
    - `interpolation::Bool`: Specify whether the resized output array should be smoothed 
    - `smoothing::String`: If interpolation == true, you can select whether to apply a linear smoothing function (default) with "linear" or a nearest neighbor smoothing function if smoothing != "linear".
    """
    function resizeRaster(raster::AbstractArray, targetWidth::Integer, targetHeight::Integer; gpu::Bool=false, device::Int = 0, interpolation = true, smoothing="linear")
        resizedImageArray = Array{eltype(raster)}(undef, targetWidth, targetHeight) # Output Array for Resized Image

        if isa(raster, CuArray) == false | gpu == false
            # Uses ImageTransformations.jl
            if smoothing == "linear"
                resizedImageArray = imresize(raster, (targetWidth, targetHeight), method=Linear());
            else
                resizedImageArray = imresize(raster, (targetWidth, targetHeight));
            end
        else
            # Output Image Size
            originalLength = height(raster)
            originalHeight = width(raster)
            imgType = eltype(raster)

            # Get Free vRam
            if device == 0
                dev = first(NVML.devices())
            else 
                dev = device!(device)
            end
            
            availMem = NVML.memory_info(dev).free
            availMem = availMem * .9
            
            # Get Max Img Size that Fits in vRAM
            maxLengthRam = floor(Int, sqrt(floor(Int, availMem / sizeof(imgType))))
            
            estUsage = originalHeight + targetHeight
            if estUsage < maxLengthRam
                numberIterations = 1
            else
                numberIterations = ceil(Int, estUsage / maxLengthRam)
            end
            
            ### Resize Input 
            inputDiv = (originalHeight + targetHeight) / maxLengthRam # Ratio for IDing Input and Output GPU Chunk Lengths
            outputChunkLength = floor(Int, targetWidth / inputDiv)
            outputChunkHeight = floor(Int, targetHeight / inputDiv)
            inputChunkLength = ceil(Int, originalLength / inputDiv)
            inputChunkHeight = ceil(Int, originalHeight / inputDiv)
            
            coordinatesOriginalImageArray = collect(TileIterator(axes(raster), (inputChunkLength, inputChunkHeight))) # Equals Input Length to GPU
            coordinatesResizedImageArray = collect(TileIterator(axes(resizedImageArray), (outputChunkLength, outputChunkHeight))) # Equals Output Length from GPU
            for i in 1:length(coordinatesOriginalImageArray)
                @info i
                chunkCoords = coordinatesOriginalImageArray[i]
                chunkToGPU = raster[chunkCoords...]
                resizedArrayCoords = coordinatesResizedImageArray[i]
                inputWidth = (maximum(resizedArrayCoords[1]) - minimum(resizedArrayCoords[1])) + 1
                inputHeight = (maximum(resizedArrayCoords[2]) - minimum(resizedArrayCoords[2])) + 1
                outputChunk = resizeCuda(chunkToGPU, inputWidth, inputHeight; interpolation=interpolation, smoothing = smoothing, returnGPU = false)
                resizedImageArray[resizedArrayCoords...] = outputChunk
                outputChunk = nothing
            end
        end
        return resizedImageArray
    end

    function pn(x) 
        @printf "%f" maximum(x)
    end

    # Downloads SAFE metadata xml and extracts geoms and nodata value

    function safeCoverageData(location; target="L2A")
        downloadStrings = replace(location, r"gs://gcp-public-data-sentinel-2/" => "")
        collData = GoogleCloud.storage(:Object, :get, "gcp-public-data-sentinel-2", downloadStrings * "/MTD_MSI$target.xml") 
        io = IOBuffer();
        write(io, collData);
        fileList = String(take!(io));
        safeMeta = LightXML.parse_string(fileList)
        a = root(safeMeta)
        b = child_nodes(a)
        n = ""
        m = ""
        for (i, elm) in enumerate(b)
            if LightXML.name(elm) == "Geometric_Info"
                for (j, cElm) in enumerate(child_nodes(elm))
                    if LightXML.name(cElm) == "Product_Footprint"
                        for (k, dElm) in enumerate(child_nodes(cElm))
                            if LightXML.name(dElm) == "Product_Footprint"
                                for (l, eElm) in enumerate(child_nodes(dElm))
                                    if LightXML.name(eElm) == "Global_Footprint"
                                        n = strip(content(eElm))
                                    end
                                end
                            end
                        end
                    end
                end
            elseif LightXML.name(elm) == "Quality_Indicators_Info"
                for (j, cElm) in enumerate(child_nodes(elm))
                    if LightXML.name(cElm) == "Image_Content_QI"      
                        for (k, dElm) in enumerate(child_nodes(cElm))
                            if LightXML.name(dElm) == "NODATA_PIXEL_PERCENTAGE"
                                m = strip(content(dElm))
                            end
                        end
                    end
                end
            end
        end
        s = split(n, " ")
        s[(1:length(s)) .% 2 .!= 0] .* " " .* s[(1:length(s)) .% 2 .!= 1] .* ","
        h = hcat(s[(1:length(s)) .% 2 .!= 0], s[(1:length(s)) .% 2 .!= 1])
        i = 0
        cArray = Vector{Float64}[]
        for i in 1:size(h)[1]
            l, r =  parse(Float64, String(h[i, 1])), parse(Float64, String(h[i, 2]))
            push!(cArray, [l, r])
        end
        polygon = LibGEOS.Polygon(LibGEOS.coordinates([cArray]))
        if m != ""
            noData = parse(Float64, m)
        else
            noData = 0.0 # L1C No Data Value 
        end        
        return noData, polygon
    end

    function scanNorm(x, maxVal)
        nullVal = convert(eltype(x), 0)
        if x != nullVal
            x = x / maxVal
        end
        return x
    end

    function loadSentinel(tif; GPU=false, normalize=false, target="L2A")
        tifArray = Dict()
        origString = tif * "/MTD_MSI$target.xml"
        safeMeta = ArchGDAL.read(origString)
        safeBasicMetaData = ArchGDAL.metadata(safeMeta)
        fileNames = ArchGDAL.metadata(safeMeta; domain="SUBDATASETS")
        fileList = split.(fileNames[occursin.("NAME", fileNames)], "=")
        cnt = 1
        for i in fileList
            fileBand = i[2]
            resolution = split(fileBand, ":")[3]
            if resolution != "TCI"
                ArchGDAL.read(fileBand) do dataset
                    number_rasters = (ArchGDAL.nraster(dataset))
                    ref = ArchGDAL.getproj(dataset)
                    geotransform = ArchGDAL.getgeotransform(dataset)
                    for bandCounter in 1:number_rasters
                        bandInfo = ArchGDAL.getband(dataset, bandCounter)
                        metaDataItem = ArchGDAL.metadataitem(bandInfo, "BANDNAME", domain="")
                        GPU == true ? band = CuArray{Float16}(ArchGDAL.read(bandInfo)) : band = Array{Int32}(ArchGDAL.read(bandInfo))
                        
                        # Assumes that minimum band value is always == or ~= 0.0
                        if normalize == true && metaDataItem[1] == 'B'
                            maxPixel = CUDA.findmax(band)
                            band = broadcast(scanNorm, band, maxPixel[1])
                        end
                        @info 1
                        bandMetaData = Dict("band_data" => ArchGDAL.metadata(bandInfo), 
                                            "location" =>  ArchGDAL.metadata(bandInfo; domain="LocationInfo"), 
                                            "image_structure" => ArchGDAL.metadata(bandInfo; domain="IMAGE_STRUCTURE"#, 
                                            ))  
                        @info 2
                        fileMetaData = Dict("filedata" => safeBasicMetaData, "ref" => ref, "geotransform" => geotransform)
                        @info 3
                        band = attach_metadata(band, Dict("band" => bandMetaData, "file" => fileMetaData));
                        sentRaster = Dict("$metaDataItem-$resolution" => band)
                        @info 4
                        tifArray = merge!(tifArray, sentRaster);
                        cnt = cnt + 1
                        band = nothing
                    end
                end
            end
        end
        return tifArray;
    end

    # Scan Code for CUDA Screens
    scanInvert(x, z) = (x > convert(typeof(x), z) ? x = convert(typeof(x), 1) : x = convert(typeof(x), 0))

    function cudaScan(x, y, z)
        z = convert(eltype(x), z)
        yNull = convert(typeof(y), 0)

        if x > z
            if y != yNull
                y = yNull
            end
        end
        return y
    end

    function flushSAFE(files; GPU = false) 
        for file in files
            if haskey(file, "CloudScreen")
                for i in keys(file)
                    if i != "CloudScreen"
                        keyName = split(i, "-")
                        if keyName[2] != "Screened" && keyName[2] != "10m"
                            file[i] = nothing
                        end
                    end
                end
            end
            if GPU == true
                GC.gc()
                CUDA.reclaim()
            end
        end
        return
    end

    function cudaRevScan(x, y, z)
        z = convert(typeof(x){elmtype}, z)
        yNull = convert(typeof(y), 0)
        if x < z
            if y != yNull
                y = yNull
            end
        end
        return y
    end

    cudaInvert(x) = (x == 1.0 ? x = 0.0 : x = 1.0)

    cudaBitScan(x,y) = (x == convert(typeof(x), 0) || y == convert(typeof(x), 0) ? convert(typeof(x), 0) : convert(typeof(x), 1))

    #scanMean(x, y) = ((x+y)/ convert(typeof(y), 2))
    function scanMean(x, y)
        x = convert(typeof(y), x)
        z = Float64(0)
        z = (x + y) / convert(typeof(y), 2)
        return z
    end

    scanFind(x, y) = (x == convert(typeof(x), y) ? x : convert(typeof(x), 0))

    scanGreaterThan(x, y) = (x > convert(typeof(x), y) ? x : convert(typeof(x), 0))

    #Array(broadcast(scanMean, parent(gpuOne["B1-Screened"]), parent(gpuOne["B2-Screened"]))) Pansharpening Code

    function cudaCirrusScan(x, y, lower, upper)
        lower = convert(eltype(x), lower)
        upper = convert(eltype(x), upper)
        yNull = convert(eltype(y), 0)
        if y == yNull
            y = yNull
        elseif x > upper
            y = yNull
        elseif x < lower
            y = yNull
        end
        return y
    end

    # Create Screen (CUDA Only)

    function sentinelCloudScreen(targetFile, screenFile; type="L2", b1Screen = 750, b2Screen = 1500, b4Screen = 1500, b10Screen = 1500, cloudMaskScreen = 80, GPU=true, GPU_All = false)
        #println("Starting Cloud Screen")
        # Users can pass either a string to the SAFE file location or preloaded Abstract Arrays
        if typeof(targetFile) == String
            #println("Loading Files")
            imageBandsA = loadSentinel(targetFile) 
            imageBandsB = loadSentinel(screenFile) 
        else
            #println("Selecting Bands")
            imageBandsA = targetFile
            imageBandsB = screenFile
        end

        function processScreens(imageBands1, imageBands2; type = "L2", GPU=true, GPU_All = false)
            #println("Processing Cloud Screens")
            # Loads B1, B2, B4, B9, Cloud Screen, and Surface Classification (Cirrus Cloud Detection)
            b1 = parent(imageBands1["B1-60m"])
            b2 = parent(imageBands2["B1-60m"])
            #n1 = parent(imageBands1["B9-60m"])
            #n2 = parent(imageBands2["B9-60m"])
            bl1 = parent(imageBands1["B2-10m"])
            bl2 = parent(imageBands2["B2-10m"])
            r1 = parent(imageBands1["B4-10m"])
            r2 = parent(imageBands2["B4-10m"])

            if type == "L2"
                c2 = parent(imageBands2["CLD-60m"])
                s2 = parent(imageBands2["SCL-60m"])
            else
                n1 = parent(imageBands1["B9-60m"])
                n2 = parent(imageBands2["B9-60m"])
            end

            if GPU_All == false
                #println("CPU")
                # Apply B1 Time Screen
                t1 = b2 - b1
                t1a = BitArray(zeros(width(t1), height(t1)))
                tvec = findall(t1 .> b1Screen)
                t1a[tvec] .= 1

                if type == "L2"
                    # Apply Sen2Cor Cloud Mask Screen
                    cvec = findall(c2 .> cloudMaskScreen)
                    t1a[cvec] .= 1

                    # Apply Cirrus Cloud Screen
                    t1a = .!t1a
                    t1a = broadcast(cudaCirrusScan, s2, t1a, 1, 8)
                    
                    # Process Cloud Screen - Cloud Masks are always at 10m
                    GPU == true ? cloudScreen = resizeCuda(CuArray(Array{Float16}(t1a)), width(r1), height(r1); interpolation=false) : cloudScreen = Images.imresize(t1a, width(r1), height(r1));
                    testVec = findall(cloudScreen .< 1)
                    cloudScreen[testVec] .= 0
                    cloudScreen = BitArray(cloudScreen)
                else 
                    # Apply B10 Time Screen
                    nScreen = n2 - n1
                    nVec = findall(nScreen .> b10Screen)
                    t1a[nVec] .= 1
                    t1a = .!t1a
                    GPU == true ? cloudScreen = resizeCuda(CuArray(Array{Float16}(t1a)), width(r1), height(r1); interpolation=false) : cloudScreen = Images.imresize(t1a, width(r1), height(r1));
                    cloudScreen = BitArray(cloudScreen)
                end
        
                # Apply B2 Time Screen
                bScreen = bl2 - bl1
                blVec = findall(bScreen .> b2Screen)
                cloudScreen[blVec] .= 0

                # Apply B4 Time Screen
                rScreen = r2 - r1
                rVec = findall(rScreen .> b4Screen)
                cloudScreen[rVec] .= 0
                return cloudScreen
            else 
                #println("GPU")

                # Apply B1 Time Screen
                t1 = b2 - b1
                t1a = broadcast(cudaScan, t1, 1.0, b1Screen)

                # Apply B9 Time Screen
                #nScreen = n2 - n1
                #t1a = broadcast(cudaScan, nScreen, t1a, b9Screen)

                if type == "L2"
                    # Apply Sen2Cor Cloud Mask Screen
                    t1a = broadcast(cudaScan, c2, t1a, cloudMaskScreen)

                    # Apply Cirrus Cloud Screen
                    t1a = broadcast(cudaCirrusScan, s2, t1a, 1, 8)

                    # Process Cloud Screen - Cloud Masks are always at 10m
                    cloudScreen = resizeCuda(CuArray{Float16}(t1a), width(r1), height(r1); returnGPU = true);
                    cloudScreen = broadcast(cudaCirrusScan, cloudScreen, cloudScreen, 1, 1)
                else
                    # Apply B10 Screen
                    nScreen = n2 - n1
                    t1a = broadcast(cudaScan, nScreen, t1a, b10Screen)
                    cloudScreen = resizeCuda(CuArray{Float16}(t1a), width(r1), height(r1); returnGPU = true);
                end

                # Apply B2 Time Screen
                bScreen = bl2 - bl1
                cloudScreen = broadcast(cudaScan, bScreen, cloudScreen, b2Screen)

                # Apply B4 Time Screen
                rScreen = r2 - r1
                cloudScreen = broadcast(cudaScan, rScreen, cloudScreen, b4Screen)
                t1 = nothing
                t1a = nothing
                bScreen = nothing
                rScreen = nothing   
                b2 = nothing
                c2 = nothing
                s2 = nothing
                bl1 = nothing
                bl2 = nothing
                r1 = nothing
                r2 = nothing
                if type != "L2"
                    n1 = nothing
                    n2 = nothing
                    nscreen = nothing
                end
                return cloudScreen
            end
        end
        screenOne = processScreens(imageBandsA, imageBandsB; GPU=GPU, GPU_All=GPU_All, type=type)
        screenTwo = processScreens(imageBandsB, imageBandsA; GPU=GPU, GPU_All=GPU_All, type=type)
        if GPU_All == true
            imageBandsA = nothing
            imageBandsB = nothing
            GC.gc()
            CUDA.reclaim()
        end
        return screenOne, screenTwo 
    end

    function generateScreens(files; type="L2", b1Screen = 1000, b2Screen = 1000, b4Screen = 1000, b10Screen = 1500, cloudMaskScreen = 20, GPU=false)
        for i in 1:length(files)
            fileA = files[i]
            # Check Geometries
            if type != "L2"
                aGeom = String(SubString(fileA["B2-10m"].filedata[10], 11, length(fileA["B2-10m"].filedata[10])))
                aGeom_Ptr = LibGEOS._readgeom(aGeom)
            else
                aGeom = String(SubString(fileA["B2-10m"].filedata[17], 11, length(fileA["B2-10m"].filedata[17])))
                aGeom_Ptr = LibGEOS._readgeom(aGeom)
            end
            if i != length(files)
                for j in i+1:length(files) 
                    fileB = files[j]
                    if type != "L2"
                        bGeom = String(SubString(fileB["B2-10m"].filedata[10], 11, length(fileB["B2-10m"].filedata[10])))
                    else
                        bGeom = String(SubString(fileB["B2-10m"].filedata[17], 11, length(fileB["B2-10m"].filedata[17])))
                    end
                    bGeom_Ptr = LibGEOS._readgeom(bGeom)
                    compareGeoms = LibGEOS.geomArea(bGeom_Ptr) / LibGEOS.geomArea(aGeom_Ptr)
                    if compareGeoms > .8 && compareGeoms < 1.2
                        targetScreen, screenScreen = sentinelCloudScreen(fileA, fileB; type = type, b1Screen = b1Screen, b2Screen = b2Screen, b4Screen = b4Screen, b10Screen = b10Screen, cloudMaskScreen = cloudMaskScreen, GPU=GPU, GPU_All = GPU)
                        #if type != "L1C"
                        haskey(fileA, "CloudScreen") == true ? fileA["CloudScreen"] = broadcast(cudaBitScan, fileA["CloudScreen"], screenScreen) : fileA["CloudScreen"] = screenScreen
                        haskey(fileB, "CloudScreen") == true ? fileB["CloudScreen"] = broadcast(cudaBitScan, fileB["CloudScreen"], targetScreen) : fileB["CloudScreen"] = targetScreen
                        #end
                        targetScreen = nothing
                        screenScreen = nothing
                        fileB = nothing
                    else 
                        println("Not Comparable")
                    end
                end
                fileA = nothing
            end
            if length(files) == 1
                arrayType = eltype(parent(fileA["B2-10m"]))
                width = size(parent(fileA["B2-10m"]))[1]
                if GPU == true
                    files[1]["CloudScreen"] = CuArray{arrayType}(ones(width, width))
                else
                    files[1]["CloudScreen"] = ones(arrayType, width, width)
                end
                arrayType = nothing
                width = nothing
            end
        end
        if GPU == true
            GC.gc()
            CUDA.reclaim()
        end
    end

    function applyScreens(files; GPU = false, normalize=false, target="", merge=false) 
        for file in files
            if haskey(file, "CloudScreen")
                for i in keys(file)
                    if i[1] == 'B' 
                        keyName = split(i, "-")
                        if keyName[2] != "Screened"
                            tmpScreen = parent(file[i])
                            if normalize == true 
                                targetScreen = parent(target[i])
                                @info i
                                tmpScreen = normalizeRasters(tmpScreen, targetScreen; merge=merge)
                                targetScreen = nothing
                            end
                            if size(file[i]) != size(file["CloudScreen"])
                                tmpScreen = resizeCuda(tmpScreen, width(file["CloudScreen"]), height(file["CloudScreen"]); returnGPU = GPU);
                            end
                            bandName = keyName[1] * "-Screened"
                            file[bandName] = tmpScreen .* file["CloudScreen"]
                            tmpScreen = nothing
                            if GPU == true
                                GC.gc()
                                CUDA.reclaim()
                            end
                        end
                    end
                end
            end
        end
        return
    end

    function scanMerge(x...; screen = 1) 
        count = 0
        y = convert(eltype(x[1]), 0)
        screen = convert(eltype(x[1]), screen)
        for i in x
            if i > screen
                y = y + i
                count = count + 1
            end
        end
        if count != 0
            y = y / convert(typeof(y), count)
        end
        return y
    end

    function scanMaxMerge(x...) 
        y = convert(eltype(x[1]), x[1])
        #y = x[1]
        for i in x
            if i > y
                y = i
            end
        end
        return y
    end

    function saveScreenedRasters(x...; names=[], type=UInt16)
        for i in 1:length(x)
            file = x[i]
            name = names[i]
            bandCount = 0
            bandList = []

            for j in keys(file)
                if j != "CloudScreen"
                    keyName = split(j, "-")
                    #@show j, keyName = split(j, "-")
                    if keyName[2] == "Screened"
                        append!(bandList, [j])
                        bandCount = bandCount + 1
                    end
                end
            end
            geoTransform = file["B2-10m"].geotransform
            ref = file["B2-10m"].ref
            sourceFile = transpose(parent(file["B2-10m"]))
            sourceFile = parent(file["B2-10m"])

            bandWidth = width(sourceFile)
            bandHeight = height(sourceFile)
            @show "Writing tif for $name"
            ArchGDAL.create(name; driver=ArchGDAL.getdriver("GTiff"), width=bandWidth, height=bandHeight, nbands=bandCount, dtype=type, options = ["BIGTIFF=YES"]) do raster
                ArchGDAL.setgeotransform!(raster, geoTransform)
                ArchGDAL.setproj!(raster, ref)
                for k in 1:bandCount
                    rast = broadcast(trunc, parent(file[bandList[k]]))
                    #rast = broadcast(removeNaN, rast)
                    #rast = transpose(rast)
                    rast = Array{type}(rast)
                    ArchGDAL.write!(raster, rast, k)
                    rast = nothing
                    ArchGDAL.getband(raster, k) do band
                        ArchGDAL.setcategorynames!(band, [bandList[k]])                    
                    end
                end
            end
            CUDA.reclaim()
            GC.gc()
        end
    end

    function generateCloudless(files...; scan="max")
        file = copy(files[1])
        keyList = keys(file)
        for key in keyList
            if key != "CloudScreen"
                if last(key, 8) == "Screened"
                    if scan == "max"
                        file[key] = broadcast(scanMaxMerge, map((x) -> parent(x[key]), files)...);
                    elseif scan == "mean"
                        file[key] = broadcast(scanMerge, map((x) -> parent(x[key]), files)...);
                    end
            end
            end
        end
        return file
    end

    function cloudDownload(location; writeFile = true)
        collData = GoogleCloud.storage(:Object, :get, "gcp-public-data-sentinel-2", location) 
        if writeFile == true
            open(location, "w") do file
                write(file, collData)
            end
        else
            io = IOBuffer();
            write(io, collData);
            return io
        end 
    end

    function obtain(SAFE)
        mkpath(SAFE)
        rawFileList = GoogleCloud.storage(:Object, :list, "gcp-public-data-sentinel-2"; prefix=SAFE, deliminater="/")
        io = IOBuffer();
        write(io, rawFileList);
        fileList = String(take!(io));
        fileList = JSON.parse(fileList);
        for item in fileList["items"]
            if occursin("_\$folder\$", item["name"]) == true
                folderPath = chop(item["name"], head=0, tail=9)
                mkpath(folderPath)
            else
                filePath = SubString.(item["name"], 1, findlast(==('/'),item["name"]))
                mkpath(filePath)
                cloudDownload(item["name"]) 
            end
        end
    end

    function safeList(; cache="/.img_cache", update=true, level="L2")
        currentDirectory = pwd()
        mkpath(cache)
        cd(cache)
        #println(cache)
        isIt = isfile("cacheIndex.csv.gz")
        println("There is an existing cacheIndex csv: $isIt")
        if isfile("cacheIndex.csv.gz") == true && update == false
            println("Opening cached csv")
            df = DataFrame(load(File(format"CSV", "cacheIndex.csv.gz")))
        else 
            if level == "L2"
                @info "Downloading L2A CSV Index"
                addr = "L2/index.csv.gz"
            else
                @info "Downloading L1C CSV Index"
                addr = "index.csv.gz"
            end
            obs = GoogleCloud.storage(:Object, :get, "gcp-public-data-sentinel-2", addr)
            open("cacheIndex.csv.gz", "w") do file
                write(file, obs)
            end
            df = DataFrame(load(File(format"CSV", "cacheIndex.csv.gz")))
        end
        cd(currentDirectory)
        return df
    end

    function sentinelFiveList(startDay, endDay; platform = "Sentinel-5", user = "s5pguest", pass = "s5pguest", start = 0, rows = 100)
        # Format Dates
        sDate = string(DateTime(startDay, "yyyymmdd"), "Z")
        eDate = string(DateTime(endDay, "yyyymmdd"), "Z")
        # Get XML
        xmlRoot, next = processSentHTML(user, pass, platform, start, rows, sDate, eDate)
        println(next)
        files = cleanSentXML(xmlRoot)
        while next != 0
        start = start + rows
        xmlRoot, next = processSentHTML(user, pass, platform, start, rows, sDate, eDate)
        println(next)
        itrFiles = cleanSentXML(xmlRoot)
        files = [files; itrFiles]
        end
        return files
    end

    function cleanSentXML(xmlRoot)
        files = DataFrame(title = String[], href = String[], fileName = String[], abbr = String[], desc = String[], uuid = String[])
        entries = xmlRoot["entry"]
        title = ""
        href = ""
        fileName = ""
        abbr = ""
        desc = ""
        uuid = ""
        for i in entries
            title = content(i["title"][1])
            linkList = i["link"]
            strList = i["str"]
            href = LightXML.attribute(linkList[1], "href")
        
            for j in strList 
            println(j)
            if LightXML.attribute(j, "name") == "filename"
                fileName = content(j)
            elseif LightXML.attribute(j, "name") == "processingmodeabbreviation"
                abbr = content(j)
            elseif LightXML.attribute(j, "name") == "producttypedescription"
                desc = content(j)
            elseif LightXML.attribute(j, "name") == "uuid"
                uuid = content(j)
            end
            end
            push!(files, (title, href, fileName, abbr, desc, uuid))
        end
        return files
    end

    function processSentHTML(user, pass, platform, start, rows, sDate, eDate)
        # Create HTTP Get Request
        link = "https://$user:$pass@s5phub.copernicus.eu/search?start=$start&rows=$rows&q="
        queryString = string("(platformname:$platform%20AND%20beginposition:%5b$sDate%20TO%20$eDate%5d)")
        url = string(link, queryString)
        
        # Get File List
        println("Getting: $url")
        resp = HTTP.get(url)
        println(resp.request)
        x = String(resp.body)
        xmlExt = parse_string(x)
        xmlRoot = root(xmlExt)
        links = xmlRoot["link"]
        next = 0
        for i in links
            if LightXML.attribute(i, "rel") == "next"
            next = LightXML.attribute(i, "href")
            end
        end
        return xmlRoot, next
    end   

    function zulu()
        escaped_format = "yyyy-mm-dd\\THH:MM:SS.sss\\Z"
        Zulu = String
        Dates.CONVERSION_SPECIFIERS['Z'] = Zulu
        Dates.CONVERSION_DEFAULTS[Zulu] = ""
        df = Dates.DateFormat(escaped_format)
        return df
    end 

    function parseSentinelTime(time, format)
        parsedTime = DateTime(time, format)
        return parsedTime
    end

    function filterList(inputFile, mgrs, startDate, endDate; cover=10, size=6.5*10^8, skipMissing = true)
        format = zulu()
        subsetDF = subset(inputFile, 
                                    :MGRS_TILE => ByRow(x -> x == mgrs),
                                    :CLOUD_COVER => ByRow(x -> x < cover),
                                    :SENSING_TIME => ByRow(x -> parseSentinelTime(x, format) > startDate),
                                    :SENSING_TIME => ByRow(x -> parseSentinelTime(x, format) < endDate),
                                    :TOTAL_SIZE => ByRow(x -> x > size),
                                    skipmissing = skipMissing,
                                    )
        subsetDF = sort(subsetDF, (:CLOUD_COVER))
        return subsetDF
    end

    function downloadTiles(inputList, numberFiles; cache="~/.img_cache")
        currentDirectory = pwd()
        mkpath(cache)
        cd(cache)
        downloadStrings = replace.(inputList.BASE_URL[1:numberFiles], r"gs://gcp-public-data-sentinel-2/" => "")
        for safeUrl in downloadStrings
            Sentinel.obtain(safeUrl)
        end
        filePath = cache .* "/" .* SubString.(downloadStrings, 1, findlast.(==('/'), downloadStrings))
        cd(currentDirectory)
        return filePath[1]
    end

    function cloudInit(credentials) 
        creds = JSONCredentials(credentials)
        session = GoogleSession(creds, ["devstorage.full_control"])
        set_session!(storage, session) 
    end

    function downloadSentinelFive(urlList; cache=pwd())
        for i in urlList
        print(i)
        authEncode = "Basic " * base64encode("s5pguest" * ":" * "s5pguest")
        auth = Dict("Authorization" => authEncode)
        HTTP.download(i, cache; headers = auth)
        end 
    end

    function bashDownloadSentinelFive(urlList; cache=pwd())
        currDir = pwd()
        cd(cache)
        outFile = open("links", "w")
        for i in urlList
            println(outFile, i)
        end
        close(outFile)
        parDownload = pipeline(`cat links`, `xargs -d "\n" -P 15 -L 1 wget --content-disposition --continue --user=s5pguest --password=s5pguest`)
        run(parDownload)
        cd(currDir)
    end

    # Loads GEOTiffs
    function loadRaster(tif; GPU=false)
        finalImg = ArchGDAL.read(tif) do dataset
            number_rasters = (ArchGDAL.nraster(dataset))
            ref = ArchGDAL.getproj(dataset)
            geotransform = ArchGDAL.getgeotransform(dataset)
            firstBand = ArchGDAL.getband(dataset, 1)
            width=ArchGDAL.width(firstBand)
            height=ArchGDAL.height(firstBand)
            finalDataset = Array{UInt16}(undef, width, height, number_rasters)
            finalDataset[:, :, 1] = Array{UInt16}(ArchGDAL.read(firstBand))
            for bandCounter in 2:number_rasters
                bandInfo = ArchGDAL.getband(dataset, bandCounter)
                GPU == true ? band = CuArray{Float16}(ArchGDAL.read(bandInfo)) : band = Array{UInt16}(ArchGDAL.read(bandInfo))
                #band = Array{UInt16}(ArchGDAL.read(bandInfo))
                finalDataset[:, :, bandCounter] = band
            end
            return finalDataset
        end
        return finalImg
    end

    # Extracts Geometries from SAFE Files
    function extractSAFEGeometries(safe; target="L2A")
        origString = safe * "/MTD_MSI$target.xml"
        safeMeta = ArchGDAL.read(origString)
        safeBasicMetaData = ArchGDAL.metadata(safeMeta)
        pathGeom = ArchGDAL.metadataitem(safeMeta, "FOOTPRINT", domain="")
        noData = ArchGDAL.metadataitem(safeMeta, "NODATA_PIXEL_PERCENTAGE", domain="")
        cloudCover = ArchGDAL.metadataitem(safeMeta, "CLOUD_COVERAGE_ASSESSMENT", domain="")
        #@show noData
        noData = parse.(Float32, noData)
        return pathGeom, noData, cloudCover
    end

    # Returns the directory path for a SAFE file
    function generateSAFEPath(tile, directory; subDirectory="/L2/tiles")
        a = tile[1:2]
        b = tile[3]
        c = tile[4:5]
        path = directory * subDirectory * "/$a/$b/$c/"
        return path
    end

    # Takes a list of SAFE directories, groups them by geometries, and returns them in sorted order (most -> lest recent)
    function sortSAFE(tile, directory; subDirectory="/L2/tiles") 
        path = generateSAFEPath(tile, directory; subDirectory=target)
        safeList = readdir(path)
        safeDF = DataFrame([safeList], :auto)
        safeDF[!, :sort] = SubString.(safeList, 12, 19)
        sort!(safeDF, :sort; rev=true)
        safeList = safeDF[!, 1]
        areaList = []
        pass = false
        # Run through the SAFE file geometries and push results to a DataFrame
        cDF = DataFrame(x1 = Any[], geom = Any[], noData = Any[], cloudCover = Any[])
        for safe in safeList
            geom, noData, cloudCover = extractSAFEGeometries(path * safe);
            cloudCover = parse(Float64, cloudCover)
            push!(cDF, [safe, geom, noData, cloudCover])
        end

        # Subset DataFrame for all obs that = full tile and have less than 20% est. cloud cover
        cDF2 = subset(cDF, 
            :noData => ByRow(x -> x < 2),
            :cloudCover => ByRow(x -> x < 20),
        )

        # Return full obs if there are enough to push to aggregation code (3 or more captures)
        if nrow(cDF2) > 1
            cDF2 = sort!(cDF2, :cloudCover)
            a = cDF2[1:2, :x1]
            b = []
            return a, nothing, nothing

        # If not, extract all obs where cloud cover is less than est 20%
        else
            tileCounter = 0
            tileMergeList = DataFrame(x1 = Any[], group=Any[])
            cDF2 = subset(cDF, 
                :cloudCover => ByRow(x -> x < 20),
            )
            cDF2 = sort!(cDF2, :noData)

            if nrow(cDF2) > 1
            # Map over remaining observations, merge geometries, and check if resulting merge == 100 % of tile area
                geomPtr = LibGEOS._readgeom(cDF2[1, :geom])
                for i in 2:nrow(cDF2)
                    geomPtr1 = LibGEOS._readgeom(cDF2[i, :geom])
                    tempUnion = LibGEOS.union(geomPtr, geomPtr1)
                    geomArea = LibGEOS.geomArea(tempUnion)
                    if geomArea > .98
                        push!(tileMergeList, [cDF2[1, :x1], 1])
                        push!(tileMergeList, [cDF2[1, :x1], 1])

                        if cDF2[1, :cloudCover] > 5 
                            delete!(cDF2, 1)
                        else
                            delete!(cDF2, i)
                        end
                        break
                    end
                end

                # Repeat for loop a second time (Yes, this isn't how it should be done)
                geomPtr = LibGEOS._readgeom(cDF2[1, :geom])
                for i in 2:nrow(cDF2)
                    geomPtr1 = LibGEOS._readgeom(cDF2[i, :geom])
                    tempUnion = LibGEOS.union(geomPtr, geomPtr1)
                    geomArea = LibGEOS.geomArea(tempUnion)
                    if geomArea > .98
                        push!(tileMergeList, [cDF2[1, :x1], 2])
                        push!(tileMergeList, [cDF2[i, :x1], 2])
                        #delete!(cDF2, 1)
                        #delete!(cDF2, i)
                        break
                    end
                end
                if nrow(tileMergeList) > 3
                    return nothing, nothing, tileMergeList
                else
                    pass = true
                end
            else 
                pass = true
            end
        end
        # Check if length of files with 98%+ coverage is greater than three
        #if length(areaList) > 3
        #    a = areaList[1:4]
        #    b = []
        #    return a, b
        if pass == true  
            fileAGeom, fileANoData, fileACloudCover = extractSAFEGeometries(path * safeList[1]);
            areaList = []
            if fileANoData < 1
                safeDF[!, :group] .= 1 
                a = safeDF 
                b = similar(a, 0)
            else 
                geomPtr = LibGEOS._readgeom(fileAGeom)
                push!(areaList, LibGEOS.geomArea(geomPtr))
                for i in 2:length(safeList)
                    fileBGeom, fileBNoData = extractSAFEGeometries(path * safeList[i])
                    geomPtr = LibGEOS._readgeom(fileBGeom)
                    push!(areaList, LibGEOS.geomArea(geomPtr))
                end
                zed = KernelDensity.kde(Vector{Float64}(areaList), npoints=4, boundary=(minimum(areaList),maximum(areaList)))
                group = []
                for i in areaList
                    if i <= zed.x[2]
                        a = 1
                    elseif zed.x[2] < i <= zed.x[3]
                        a = 2
                    else
                        a = 3
                    end
                    push!(group, a)
                end
                safeDF[!, :group] = group
                safeDF[!, :area] = areaList
                grouped_df  = groupby(safeDF, "group")
                safeDF2 = combine(grouped_df, x -> nrow(x) < 3 ? DataFrame() : x)
                #nGroups = unique(y[!, :group])
                nGroups = combine(groupby(safeDF2, [:group]), nrow => :count)
                nGroups = sort(nGroups, :count)
                if nrow(nGroups) > 1
                    for j in 1:2
                        g = nGroups[j, :group]
                        if j == 1
                            a = subset(safeDF2, :group => (x -> x .== (g)))
                        elseif j == 2
                            b = subset(safeDF2, :group => (x -> x .== (g)))
                        end
                    end
                elseif nrow(safeDF2) == 0
                    a = safeDF
                    b = similar(a, 0)
                else
                    a = subset(safeDF2, :group => (x -> x .== (1)))
                    #a = safeDF2
                    b = similar(a, 0)
                end
            end
            return a[!, :x1], b[!, :x1], nothing
        end
        #return a, b, safeDF, safeDF2
    end

    # QC Function (% of no data obs) 

    function qc(i)
        y = 0
        z = 0
        for x in i
            if floor(x) < 1
                y = y+1
            end
            z = z + 1
        end
        return y/z
    end

    function mergeSAFE(files...; normalize=false, merge=true)
        CUDA.allowscalar(true)
        file = copy(files[1])
        keyList = keys(file)
        for key in keyList
            meta = metadata(file[key]);
            if normalize == true
                #@info key
                cArray = []
                for i in 2:length(files)
                    tmpA = parent(files[i][key])
                    tmpB = parent(file[key])
                    push!(cArray, normalizeRasters(tmpB, tmpA; merge=merge))
                    tmpA = nothing
                    tmpB = nothing
                end
                if length(cArray) == 1
                    file[key] = cArray[1]
                else
                    file[key] = broadcast(scanMaxMerge, cArray...);
                end
                for i in cArray
                    i = nothing
                end
                cArray = nothing
            else
                file[key] = broadcast(rastNormMean, map((x) -> parent(x[key]), files)...);
            end
            file[key] = attach_metadata(file[key], meta); 
        end
        return file
    end

    function normalizeRasters(x, y; nSample = 20000, merge=true)
        CUDA.@allowscalar xSample = Turing.sample(x, nSample);
        CUDA.@allowscalar ySample = Turing.sample(y, nSample);
        if minimum(xSample) < maximum(xSample)
            dist = KernelDensity.kde(xSample, npoints=4, boundary=(minimum(xSample), maximum(xSample)));
            a = rastNormMean(xSample, dist.x[1], dist.x[2]);
            b = rastNormMean(ySample, dist.x[1], dist.x[2]);
            throttle = Float16(b/a);
            #@info throttle, a, b, minimum(xSample), maximum(xSample), minimum(ySample), maximum(ySample)
            retRaster = broadcast(*, x, throttle);
            if merge == true
                #retRaster = broadcast(scanRastMean, retRaster, y);
                #retRaster = broadcast(rastNormMean, retRaster, y);
                retRaster = broadcast(scanMerge, retRaster, y);
            end
        else 
            retRaster = x
        end
        x = nothing
        a = nothing
        b = nothing
        throttle = nothing
        xSample = nothing
        ySample = nothing
        return retRaster
    end

    function migrateSafe(composite, target="CPU")
        for (i, key) in enumerate(keys(composite))
            if composite[key] != nothing
                #@show i, key
                if has_metadata(composite[key]) == false
                    if target == "CPU"
                        #composite[key] = Array{type}(parent(composite[key]))
                        tmp = adapt(Array, parent(composite[key]))
                        #CUDA.unsafe_free!(composite[key])
                        composite[key] = tmp
                    else
                        composite[key] = adapt(CuArray, parent(composite[key]))
                        #composite[key] = CuArray{type}(parent(composite[key]))
                    end
                else
                    meta = metadata(composite[key])
                    #@show meta
                    if target == "CPU"
                        #composite[key] = Array{type}(parent(composite[key]))

                        composite[key] = adapt(Array, parent(composite[key]))
                        tmp = adapt(Array, parent(composite[key]))
                        #CUDA.unsafe_free!(composite[key])
                        composite[key] = tmp
                    else
                        composite[key] = adapt(CuArray, parent(composite[key]))
                        #composite[key] = CuArray{type}(parent(composite[key]))
                    end
                    composite[key] = attach_metadata(composite[key], meta)
                    meta = nothing
                end
            end
        end
        GC.gc()
        CUDA.reclaim()
        #return composite
    end


    function scanRastMean(x, y)
        c = Float64(x)
        z = convert(typeof(y), 0)
        if x != z && y != z
            c = (x+y) / convert(typeof(y), 2)
        end
        return c
    end

    function rastNormMean(x, minVal = 0, maxVal=99990)
        #c = convert(eltype(x), 0.1)
        #itr = convert(eltype(x), 0.1)
        #minVal = convert(eltype(x), minVal)
        #maxVal = convert(eltype(x), maxVal)
        c = 0.1
        itr = 0.1
        #@info length(x)
        for i in x
            if maxVal >= i > minVal
                c += i
                itr += 1.0
            end
        end
        f = Float16(c/itr)
        #f = convert(eltype(x), c/itr)
        return f
    end

    function rastMean(x...; minVal = 0, maxVal=9999)
        c = Float64(0)
        itr = Float64(0)
        minVal = convert(eltype(x), minVal)
        maxVal = convert(eltype(x), maxVal)
        #@info length(x)
        for j in x
            for i in j
                if maxVal >= i > minVal
                    c += i
                    itr += 1.0
                end
            end
        end
        f = convert(eltype(x), c/itr)
        return f
    end

    function removeNaN(x)
        if isnan(x)
            x = convert(eltype(x), 0)
        end
        return x
    end


    function generateArea(inputVector, tile)
        ret = DataFrame(a = [], b = [], c = [])
        for (i, enum) in enumerate(inputVector)
            nRet = DataFrame(a = [], b = [], c = [])
            for j in i+1:length(inputVector)
                altEnum = inputVector[j]
                tempUnion = LibGEOS.union(enum, altEnum)
                geomArea = LibGEOS.geomArea(tempUnion.ptr)
                push!(nRet, (i, j, geomArea))
            end
            if nrow(nRet) > 0
                sort(nRet, :c)
                #push!(ret, nRet)
                ret = vcat(ret, nRet)
            end
        end
        sort(ret, :c)
        return ret
    end

    function compareAreas(x, tile)
        returnDF = DataFrame(tile = [], a = [], b = [], c = [], id1 = [], id2 = [], cc1 = [], cc2 = [], d1 = [], d2 = [])
        inputVec = x[x.MGRS_TILE .== tile, :poly]
        a = generateArea(inputVec, tile);
        a[!, :id1] .= ""
        a[!, :id2] .= ""
        a[!, :cc1] .= 0.0
        a[!, :cc2] .= 0.0
        a[!, :d1] .= ""
        a[!, :d2] .= ""
        for i in 1:nrow(a)
            j = a[i, :]
            i1 = j.a
            i2 = j.b 
            id1 = x[i1, :PRODUCT_ID]
            id2 = x[i2, :PRODUCT_ID]
            cc1 = x[i1, :CLOUD_COVER]
            cc2 = x[i2, :CLOUD_COVER]
            d1 = x[i1, :SENSING_TIME]
            d2 = x[i2, :SENSING_TIME]
            a[i, :id1] = id1[1:3]
            a[i, :id2] = id2[1:3]
            a[i, :cc1] = cc1
            a[i, :cc2] = cc2
            a[i, :d1] = d1
            a[i, :d2] = d2
        end
        a[!, :tile] .= tile  
        append!(returnDF, a)
        return returnDF
    end

    function extractNoData(x)
        x[:, :nodata] .= 0
        noDataVec = []
        polyVec = []
        for i in 1:nrow(x)
            row = x[i, :]
            url = row.BASE_URL
            noData, poly = Sentinel.safeCoverageData(url; target="L2A");
            push!(noDataVec, noData)
            push!(polyVec, poly)
        end
        x[!, :nodata] = noDataVec
        x[!, :poly] = polyVec
        return x
    end

    # Modify VRTs (Sentinel 5 Function)
    function modifyVRT(input, output, inputString, xdim, ydim, lonVRT = C_NULL, latVRT = C_NULL)

        # Open VRT
        xdoc = parse_file(input)
        xroot = LightXML.root(xdoc)

        # Set Root Dimension Lengths
        LightXML.set_attribute(xroot, "rasterXSize", xdim)
        LightXML.set_attribute(xroot, "rasterYSize", ydim)

        # Set Lat and Lon VRTs
        if lonVRT != C_NULL
            metaData = xroot["metadata"]
            metaSub = metaData[1]
            metaExt = metaSub["mdi"]
            lon = metaExt[1]
            LightXML.set_content(lon, lonVRT)
            lat = metaExt[3]
            LightXML.set_content(lat, latVRT)
        end

        # Set Filename & ID Tag
        ces = xroot["VRTRasterBand"]
        e1 = ces[1]
        e2 = find_element(e1, "SimpleSource")
        e3 = find_element(e2, "SourceFilename")
        LightXML.set_content(e3, inputString)

        # Set RasterBand Dimension Lengths
        e4 = find_element(e2, "SrcRect")
        LightXML.set_attribute(e4, "xSize", xdim)
        LightXML.set_attribute(e4, "ySize", ydim)

        e5 = find_element(e2, "DstRect")
        LightXML.set_attribute(e5, "xSize", xdim)
        LightXML.set_attribute(e5, "ySize", ydim)

        # Set RasterBand SourceProperties Dimension and Block Lengths (Block = [x, y]/2)
        e6 = find_element(e2, "SourceProperties")
        LightXML.set_attribute(e6, "RasterXSize", xdim)
        LightXML.set_attribute(e6, "RasterYSize", ydim)

        ### Insert avg/mean script
        save_file(xdoc, output)
    end
end



