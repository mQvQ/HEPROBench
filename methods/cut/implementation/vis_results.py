import pyvips

tif_path = ''

tiff_image = pyvips.Image.new_from_file(tiff_path, memory=True, access="sequential")

tiff_image = tiff_image.numpy().transpose(1, 2, 0)
# save image png



