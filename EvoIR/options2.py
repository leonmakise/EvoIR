import argparse

parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--mode', type=int, default=6,
                    help='0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for enhance, 5 for all-in-one (three tasks), 6 for all-in-one (five tasks)')

# normal
parser.add_argument('--gopro_path', type=str, default="/l/users/salman.khan/jiaqima/AIO/data/test/Deblur/", help='save path of test hazy images')
parser.add_argument('--enhance_path', type=str, default="/l/users/salman.khan/jiaqima/AIO/data/test/Low-light/", help='save path of test hazy images')
parser.add_argument('--denoise_path', type=str, default="/l/users/salman.khan/jiaqima/AIO/data/test/Denoise/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="/l/users/salman.khan/jiaqima/AIO/data/test/Derain/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="/l/users/salman.khan/jiaqima/AIO/data/test/Dehaze/", help='save path of test hazy images')



# # real
# parser.add_argument('--gopro_path', type=str, default="data/test/deblur/", help='save path of test hazy images')
# parser.add_argument('--enhance_path', type=str, default="data/test/enhance/", help='save path of test hazy images')
# parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
# parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
# parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')

parser.add_argument('--output_path', type=str, default="/l/users/jiaqi.ma/EvoAdaIR_results/", help='output save path')
parser.add_argument('--ckpt_name', type=str, default="ir5d.ckpt", help='checkpoint save path')
testoptions = parser.parse_args()
