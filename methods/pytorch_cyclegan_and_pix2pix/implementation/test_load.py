import pyvips

tar_path = '/path/to/pytorch-CycleGAN-and-pix2pix/results/crc-orion-pix2pix/test_100000/P37_S43_Full_A24_C59mX_E15_20220128_171510_544056-zlib.ome_16_52224_0_512_512.tiff'
img = pyvips.Image.new_from_file(tar_path, memory=True, access='sequential').numpy()
print(img.shape)

