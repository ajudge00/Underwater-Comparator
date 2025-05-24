import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.ancuti2018 import Ancuti2018
from src.methods.metrics import get_dcp, get_uciqe
from src.mohansimon2020 import MohanSimon2020
from src.yang2011 import Yang2011
from src.methods.white_balance import ancuti2018_precomp, CompChannel, gray_world
from src.methods.contrast import gamma_correction, clahe_with_lab, histogram_linearization
from src.methods.sharpening import normalized_unsharp_masking


def test_all(img_args: dict):
    for original_path, alpha in img_args.items():
        img = cv2.imread(original_path)
        assert img is not None

        path = original_path.split('/')[0:-1]
        name, extension = original_path.split('/')[-1].split('.')

        new_path = f'../images/testing/{name}'
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        yang_result = Yang2011(img)[-1]
        ancuti_result = Ancuti2018(img, precomp_red=alpha, msf_levels=10)[-1]
        mohan_result = MohanSimon2020(img, precomp_red=alpha, msf_levels=10)[-1]

        cv2.imwrite(f'{new_path}/00_{name}_original.jpg', img)
        cv2.imwrite(f'{new_path}/01_{name}_yang.jpg', yang_result)
        cv2.imwrite(f'{new_path}/02_{name}_ancuti.jpg', ancuti_result)
        cv2.imwrite(f'{new_path}/03_{name}_mohan.jpg', mohan_result)

        img_dcp = get_dcp(img)
        yang_dcp = get_dcp(yang_result)
        ancuti_dcp = get_dcp(ancuti_result)
        mohan_dcp = get_dcp(mohan_result)

        cv2.imwrite(f'{new_path}/04_{name}_original_dcp.jpg', (img_dcp * 255).astype(np.uint8))
        cv2.imwrite(f'{new_path}/05_{name}_yang_dcp.jpg', (yang_dcp * 255).astype(np.uint8))
        cv2.imwrite(f'{new_path}/06_{name}_ancuti_dcp.jpg', (ancuti_dcp * 255).astype(np.uint8))
        cv2.imwrite(f'{new_path}/07_{name}_mohan_dcp.jpg', (mohan_dcp * 255).astype(np.uint8))

        # METRICS
        img_uciqe = get_uciqe(img)
        yang_uciqe = get_uciqe(yang_result)
        ancuti_uciqe = get_uciqe(ancuti_result)
        mohan_uciqe = get_uciqe(mohan_result)

        img_dcp_median = np.median(img_dcp)
        yang_dcp_median = np.median(yang_dcp)
        ancuti_dcp_median = np.median(ancuti_dcp)
        mohan_dcp_median = np.median(mohan_dcp)

        with open(f'{new_path}/08_{name}_metrics.txt', 'w') as f:
            f.write(f'{name}.{extension}\n')
            f.write(f'UCIQE:\n')
            f.write(f'\tORIGINAL:\t{img_uciqe}\n')
            f.write(f'\tYANG:\t\t{yang_uciqe}\n')
            f.write(f'\tANCUTI:\t\t{ancuti_uciqe}\n')
            f.write(f'\tMOHAN:\t\t{mohan_uciqe}\n')
            f.write(f'\nDCP MEDIAN:\n')
            f.write(f'\tORIGINAL:\t{img_dcp_median}\n')
            f.write(f'\tYANG:\t\t{yang_dcp_median}\n')
            f.write(f'\tANCUTI:\t\t{ancuti_dcp_median}\n')
            f.write(f'\tMOHAN:\t\t{mohan_dcp_median}\n')


def test_histograms(img_path: str):
    img = cv2.imread(img_path)
    assert img is not None
    filename = img_path.split('/')[-1].split('.')[0]

    wb = ancuti2018_precomp(CompChannel.COMP_RED, img)
    wb = gray_world(wb)
    gamma = gamma_correction(wb, 2.2)
    sharp = normalized_unsharp_masking(wb)
    gamma_clahe = clahe_with_lab(gamma)
    sharp_histlin, best_r = histogram_linearization(sharp)

    cv2.imwrite(f'../images/testing/histograms/{filename}_gamma.jpg', gamma)
    cv2.imwrite(f'../images/testing/histograms/{filename}_sharp.jpg', sharp)
    cv2.imwrite(f'../images/testing/histograms/{filename}_clahe.jpg', gamma_clahe)
    cv2.imwrite(f'../images/testing/histograms/{filename}_histlin.jpg', sharp_histlin)

    plot_histogram(cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY),
                   'Eredeti hisztogram (gamma)',
                   '00_original_gamma'
                   )
    plot_histogram(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY),
                   'Eredeti hisztogram (élesített)',
                   '01_original_sharp')
    plot_histogram(cv2.cvtColor(gamma_clahe, cv2.COLOR_BGR2GRAY),
                   'CLAHE után (gamma)',
                   '02_clahe_gamma'
                   )
    plot_histogram(cv2.cvtColor(sharp_histlin, cv2.COLOR_BGR2GRAY),
                   f'Hiszt. kiegyenl. hatványozással (élesített) (r = {str(best_r)[:5]})',
                   '03_histlin_sharp'
                   )


def plot_histogram(img_gray: np.ndarray, title: str, filename: str):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()

    plt.figure(figsize=(10, 5))
    plt.bar(range(256), hist_normalized.flatten(), width=1.0, color='gray')
    plt.title(title)
    plt.xlabel("Intenzitás (0-255)")
    plt.ylabel("Normalizált előfordulás")
    plt.xlim([0, 255])
    plt.grid(axis='y', alpha=0.5)
    # plt.show()

    plt.savefig(f'../images/testing/histograms/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # test_all({
    #     '../images/testing/U-45_13.jpg': 1.0,
    #     '../images/testing/UIEB_163.png': 1.0,
    #     '../images/testing/UIEB_176.png': 1.0,
    #     '../images/testing/UIEB_184.png': 1.0,
    #     '../images/testing/UIEB_471.png': 1.0,
    #     '../images/testing/UIEB_508.png': 1.0,
    #     '../images/testing/UIEB_79.png': 2.8,
    #     '../images/testing/UIEB_526.png': 1.0
    # })

    test_histograms('../images/testing/UIEB_526.png')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
