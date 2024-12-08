# purpose: shift (horizontal translation) a raster 
# dependance: gdal utilities
# condition: the raster coordinate system unit must be 1 metre (EPSG:9001)
# usage: sh shiftImage.sh inputimage dEast_in_meters dNorth_in_meters outputimage
# example: sh shiftImage.sh image.tif -0.8 +3.0 imageShifted.tif
# author: simon
module purge
module load gdal 
imgIn=$1
dEast=$(echo $2 | tr -d +)
dNorth=$(echo $3 | tr -d +)
imgOut=$4
ulx=`gdalinfo $imgIn | grep "Upper Left" | awk '{print $4}' | awk -F "," '{print $1}'`
uly=`gdalinfo $imgIn | grep "Upper Left" | awk '{print $5}' | awk -F ")" '{print $1}'`
lrx=`gdalinfo $imgIn | grep "Lower Right" | awk '{print $4}' | awk -F "," '{print $1}'`
lry=`gdalinfo $imgIn | grep "Lower Right" | awk '{print $5}' | awk -F ")" '{print $1}'`
echo "image corners ulx uly lrx lry: " $ulx $uly $lrx $lry
ulx2=`echo "$ulx+$dEast" | bc -l`
lrx2=`echo "$lrx+$dEast" | bc -l`
uly2=`echo "$uly+$dNorth" | bc -l`
lry2=`echo "$lry+$dNorth" | bc -l`
echo "new image corners ulx uly lrx lry: " $ulx2 $uly2 $lrx2 $lry2
gdal_translate -a_ullr $ulx2 $uly2 $lrx2 $lry2 $imgIn $imgOut
exit 0