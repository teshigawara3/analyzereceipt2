import cv2 as cv
import numpy as np
import os
import re
from matplotlib import pyplot as plt
from statistics import variance
from PIL import Image
from PIL.ExifTags import TAGS

def main_4():
    """
    四隅判定
    :return:
    """

    result_file_path = 'result.txt'
    image_path = './ngreceipt'
    results = []
    p = re.compile(r'.*.png$') # 画像ファイル読み込み用

    # original_image = cv.imread(image_path + "/16234398.png")
    # approx = get_maxarea_image(original_image, image_path + "/o_16234398.png")
    # results.append(image_path + "/o_16234398.png" + judgeOutOfArea(original_image, approx))
    # write_result(result_file_path, results)

    for file in os.listdir(image_path): # レシート画像の読み込み
        m = p.match(file)
        if m != None:
            # 画像ファイルを読み込む
            original_image = cv.imread(image_path + "/" + m.group())
            approx = get_maxarea_image(original_image, image_path + "/o_" + m.group())
            results.append(file.title() + judgeOutOfArea(original_image, approx))
    write_result(result_file_path, results)

def main_pinboke():
    image_path = './pinboke/renna.jpeg'

    # get_sharp_image(cv.imread(image_path))
    get_blur(cv.imread(image_path))
    print(get_laplacian(cv.imread(image_path)))
    print(get_laplacian(cv.imread("image_blurred_1.jpg")))
    print(get_laplacian(cv.imread("image_blurred_2.jpg")))
    print(get_laplacian(cv.imread("image_blurred_3.jpg")))
    print(get_laplacian(cv.imread("image_blurred_4.jpg")))
    print(get_laplacian(cv.imread("image_blurred_5.jpg")))

def get_blur(read_image):
    image_blurred_1 = cv.GaussianBlur(src=read_image, ksize=(3, 3), sigmaX=0, sigmaY=0)
    image_blurred_2 = cv.GaussianBlur(src=read_image, ksize=(5, 5), sigmaX=0, sigmaY=0)
    image_blurred_3 = cv.GaussianBlur(src=read_image, ksize=(7, 7), sigmaX=0, sigmaY=0)
    image_blurred_4 = cv.GaussianBlur(src=read_image, ksize=(9, 9), sigmaX=0, sigmaY=0)
    image_blurred_5 = cv.GaussianBlur(src=read_image, ksize=(11, 11), sigmaX=0, sigmaY=0)
    cv.imwrite("image_blurred_1.jpg", image_blurred_1)
    cv.imwrite("image_blurred_2.jpg", image_blurred_2)
    cv.imwrite("image_blurred_3.jpg", image_blurred_3)
    cv.imwrite("image_blurred_4.jpg", image_blurred_4)
    cv.imwrite("image_blurred_5.jpg", image_blurred_5)
    """
    結果
    1160.221451195952
    181.59207103346444
    76.88289956423715
    32.00924399339125
    20.075002140990705
    12.688829604657826
    """

def get_exifinfo():

    image_path = './exif'
    p = re.compile(r'.*.png$') # 画像ファイル読み込み用
    results = []
    for file in os.listdir(image_path): # レシート画像の読み込み
        m = p.match(file)
        if m != None:
            # 画像ファイルを読み込む
            with Image.open(image_path + "/" + file) as f:
                exif = f._getexif()
                for id, value in exif.items():
                    # print(TAGS.get(id), value)
                    if (TAGS.get(id) == "LensModel"):
                        results.append("応募ID:" + file.title() + " 端末名" + value)
    write_result("./result.txt", results)


def get_laplacian(read_image):
    gray = cv.cvtColor(read_image, cv.COLOR_BGR2GRAY)  # グレースケースに変換

    laplacian = cv.Laplacian(gray, cv.CV_64F)  # ラプラシアン値
    return laplacian.var()
def get_houghlines(read_image):
    """
    エッジから直線を引く
    :param read_image:
    :return:
    """
    # ガウシアン変換を行う
    image_blurred = cv.GaussianBlur(src=read_image, kernel=(5, 5), sigmaX=0, sigmaY=0)

    # グレースケールに変換
    gray_image = cv.cvtColor(image_blurred, cv.COLOR_RGB2GRAY)
    edge = cv.Canny(gray_image, 50, 150, apertureSize=3)
    cv.imwrite('edge.jpg', edge)
    th3 = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 41, 1)
    cv.imwrite('th3.jpg', th3)

    # opening処理を行う。縮小した後に拡大処理を行うので小さい点のようなノイズを取り除くことができる
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel, iterations=3)
    cv.imwrite('opening.jpg', opening)

    # lines = cv.HoughLines(opening, 1, np.pi/180, 100)
    lines = cv.HoughLinesP(opening, 1, np.pi/360, threshold=50, minLineLength=50, maxLineGap=1)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(read_image,(x1, y1), (x2, y2), (0, 0, 255), 2)

        # for rho, theta in line:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))

            # cv.line(read_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imwrite('houghlines.jpg', read_image)

def get_sharp_image(read_image):
    FILE_HIGH_PASS = "haghpassimage_3.jpg"
    # ハイパス フィルタ
    k = 1.0
    kernel = np.array([[-k, -k, -k], [-k, 1 + 8 * k, -k], [-k, -k, -k]])

    img_high_pass = cv.filter2D(read_image, -1, kernel)
    cv.imwrite(FILE_HIGH_PASS, img_high_pass)

def get_maxarea_image(read_image, output_path):
    """

    :param read_image:
    :return:
    """
    DARKGRAY = 50  # 暗すぎるグレー　TODO 要調整
    # 面積を取得
    height, width = read_image.shape[:2]
    image_area = height * width
    # ガウシアンフィルタで画像をぼかす(ノイズの除去)
    img_gaus_blur = cv.GaussianBlur(read_image, (15, 15), 0)
    # gbrに分解
    g, b, r = cv.split(img_gaus_blur)

    for i in range(len(g)):
        for j in range(len(g[i])):
            # 各画素の偏差を求める
            avg = (int(g[i][j]) + int(b[i][j]) + int(r[i][j])) / 3
            list = [g[i][j], b[i][j], r[i][j]]
            maxvalue = max(list)
            minvalue = min(list)

            # 色の偏差が10以上ある場合は色がついている部分とみなして、色を黒にする
            # if (g[i][j] - avg > VARIANT or b[i][j] - avg > VARIANT or r[i][j] - avg > VARIANT):
            # 最大値と最小値の差分が20以上ある場合は色がついている部分とみなして、黒色に塗りつぶす
            if ((maxvalue - minvalue) >= 30):
                g[i][j] = 0
                b[i][j] = 0
                r[i][j] = 0
            else:
                # グレーでも暗すぎる場合は黒とみなす TODO ここも要調整
                if (avg < DARKGRAY):
                    g[i][j] = 0
                    b[i][j] = 0
                    r[i][j] = 0
                else:
                    g[i][j] = 255
                    b[i][j] = 255
                    r[i][j] = 255

    # 分解した画像を元に戻す
    processed_image = cv.merge((g, b, r))
    cv.imwrite('processed_image.jpg', processed_image)
    # グレースケールに変換
    gray_image = cv.cvtColor(processed_image, cv.COLOR_RGB2GRAY)

    # 2値化を行う
    retval, otsu_binary = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)

    # 斑点を除去するために縮小拡大を行う
    # opening処理を行う。縮小した後に拡大処理を行うので小さい点のようなノイズを取り除くことができる
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(otsu_binary, cv.MORPH_OPEN, kernel, iterations=3)

    cv.imwrite('opening.jpg', opening)
    # 画像の矩形を調べる
    contours, hierarchy = cv.findContours(otsu_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 抽出された矩形の中で最も面積の大きいものを調べる
    max_cnt = 0  # 矩形のインデックス
    tmp_area = 0  # 矩形の面積
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            # 最も外側の矩形のみ対象に面積を求める
            # cv.drawContours(src, contours, i, (0, 255, 0), 3)
            if cv.contourArea(contours[i]) > tmp_area:
                tmp_area = cv.contourArea(contours[i])
                max_cnt = i

    # 求まった面積最大の矩形の形状を大まかなものに近似する
    arclen = cv.arcLength(contours[max_cnt], closed=True)
    approx = cv.approxPolyDP(contours[max_cnt], epsilon=arclen * 0.07, closed=True)

    cv.drawContours(read_image, [approx], -1, (0, 255, 0), 3)
    cv.imwrite(output_path, read_image)
    return approx


def judgeOutOfArea(origin_iamge, approx_image):

    if (len(approx_image)) != 4:
        return "レシートの形が正しく読み込めませんでした。(矩形判定できず)"


    org_height, org_width = origin_iamge.shape[:2]

    for pixel in approx_image:
        height_ = pixel[0][1]
        width_ = pixel[0][0]
        if height_ >= (org_height - 10) or height_ <= 10:
            return "縦がはみ出してるかも"

        if width_ >= (org_width - 10) or width_ <= 10:
            return "横がはみ出してるかも"

    return "問題ありませんでした。"

def adaptive_threshold(read_image):



    """
    result_file_path = 'area_result.txt'
    p = re.compile(r'.*.png')
    with open(result_file_path, mode='w', encoding='utf-8') as f:
        for file in os.listdir():
            m = p.match(file)
            if m != None:
                image_file = cv.imread(m.group())
                height, width = image_file.shape[:2]
                image_area = height * width
                img_gaus_blur = cv.GaussianBlur(image_file, (15, 15), 0)
                img_hsv = cv.cvtColor(img_gaus_blur, cv.COLOR_HSV2BGR)
                lower_white = np.array([0, 0, 100])
                upper_white = np.array([180, 200, 255])
                mask_white = cv.inRange(img_hsv, lower_white, upper_white)
                retval, otsu_binary = cv.threshold(mask_white, 0, 255, cv.THRESH_OTSU)
                contours, hierarchy = cv.findContours(otsu_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                tmp_area = 0
                for i in range(len(contours)):
                    if hierarchy[0][i][3] == -1:
                        if cv.contourArea(contours[i]) > tmp_area:
                            tmp_area = cv.contourArea(contours[i])
                            max_cnt = i

                arclen = cv.arcLength(contours[max_cnt], closed=False)
                approx = cv.approxPolyDP(contours[max_cnt], epsilon=arclen * 0.025, closed=False)
                cv.drawContours(image_file, [approx], -1, (0, 255, 0), 3)
                cv.imwrite(m.group() + '.jpg', image_file)
                f.write(m.group() + '最大面積:' + str(tmp_area) + ' 画像面積：' + str(image_area) + "|||")
    """
    # 影を消すテスト
    """
    read_image = cv.imread("13473587_5105830_1.png")
    rgb_planes = cv.split(read_image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        cv.normalize(diff_img, diff_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        # result_planes.append(diff_img)
        result_norm_planes.append(diff_img)

    # result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)

    # cv.imwrite('shadows_out.png', result)
    cv.imwrite('shadows_out_norm.png', result_norm)
    """

def write_result(file_path, results):
    with open(file_path, mode='w', encoding='utf-8') as f:
        for result in results:
            f.write(result)
            f.write("\n")


def get_center_pxcel(image):
    # 面積を取得
    height, width = image.shape[:2]
    return height / 2, width / 2

def calc_area(contours, hierarchy):
    """
    抽出された矩形の中で最も面積の大きいものを調べてインデックスを返す
    :param contours:
    :param hierarchy:
    :return: int
    """
    max_cnt = 0  # 矩形のインデックス
    tmp_area = 0  # 矩形の面積
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            # 矩形はhierarchyを持っていて、最も外側の矩形の面積のみを求める
            if cv.contourArea(contours[i]) > tmp_area:
                tmp_area = cv.contourArea(contours[i])
                max_cnt = i
    return max_cnt

def calc_variance(list):
    return variance(list)



    """ メインで使用する関数
    VARIANT = 12 # 許容する偏差の値 TODO 要調整
    DARKGRAY = 120 # 暗すぎるグレー　TODO 要調整


    result_file_path = 'area_result_3.txt'
    p = re.compile(r'.*.png$')
    with open(result_file_path, mode='w', encoding='utf-8') as f:
        for file in os.listdir():
            m = p.match(file)
            if m != None:
                # 画像ファイルを読み込む
                read_image = cv.imread(m.group())
                # 面積を取得
                height, width = read_image.shape[:2]
                image_area = height * width
                # ガウシアンフィルタで画像をぼかす
                img_gaus_blur = cv.GaussianBlur(read_image, (15, 15), 0)
                # gbrに分解
                g, b, r = cv.split(img_gaus_blur)
                print(len(g))
                for i in range(len(g)):
                    for j in range(len(g[i])):
                        # 各画素の偏差を求める
                        avg = (int(g[i][j]) + int(b[i][j]) + int(r[i][j])) / 3
                        list = [g[i][j], b[i][j], r[i][j]]
                        maxvalue = max(list)
                        minvalue = min(list)

                        # 色の偏差が10以上ある場合は色がついている部分とみなして、色を黒にする
                        #if (g[i][j] - avg > VARIANT or b[i][j] - avg > VARIANT or r[i][j] - avg > VARIANT):
                        # 最大値と最小値の差分が20以上ある場合は色がついている部分とみなして、黒色に塗りつぶす
                        if ((maxvalue - minvalue) >= 20):
                            g[i][j] = 0
                            b[i][j] = 0
                            r[i][j] = 0
                        else:
                            # グレーでも暗すぎる場合は黒とみなす TODO ここも要調整
                            if (avg < DARKGRAY):
                                g[i][j] = 0
                                b[i][j] = 0
                                r[i][j] = 0

                # 分解した画像を元に戻す
                processed_image = cv.merge((g, b, r))

                # cv.imwrite('processed_image.png', processed_image)
                cv.imwrite(m.group() + '_process.jpg', processed_image)

                # グレースケールに変換
                gray_image = cv.cvtColor(processed_image, cv.COLOR_RGB2GRAY)

                # 2値化を行う
                retval, otsu_binary = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
                # cv.imwrite('otsu_binary.png', otsu_binary)

                # 斑点を除去するために縮小拡大を行う
                # 縮小処理を行う
                kernel = np.ones((5, 5), np.uint8)
                erosion = cv.erode(otsu_binary, kernel, iterations=2)
                # 膨張処理を行う。こうすることで斑点を削除することができる
                dilate = cv.dilate(erosion, kernel, iterations=2)
                cv.imwrite(m.group() + 'dilate.jpg', dilate)


                # 画像の矩形を調べる
                contours, hierarchy = cv.findContours(otsu_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                # 抽出された矩形の中で最も面積の大きいものを調べる
                max_cnt = 0  # 矩形のインデックス
                tmp_area = 0  # 矩形の面積
                for i in range(len(contours)):
                    if hierarchy[0][i][3] == -1:
                        # 最も外側の矩形のみ対象に面積を求める
                        # cv.drawContours(src, contours, i, (0, 255, 0), 3)
                        if cv.contourArea(contours[i]) > tmp_area:
                            tmp_area = cv.contourArea(contours[i])
                            max_cnt = i
                # 求まった最大面積の矩形を保存
                cv.drawContours(read_image, contours, max_cnt, (0, 255, 0), 3)
                cv.imwrite(m.group() + 'contours.jpg', read_image)

                # 求まった面積最大の矩形の形状を大まかなものに近似する
                arclen = cv.arcLength(contours[max_cnt], closed=False)
                approx = cv.approxPolyDP(contours[max_cnt], epsilon=arclen * 0.025, closed=False)
                cv.drawContours(read_image, [approx], -1, (0, 255, 0), 3)
                cv.imwrite(m.group() + 'contours_apr.jpg', read_image)
                print(m.group() + '最大面積:' + str(tmp_area) + ' 画像面積：' + str(image_area))
                f.write(m.group() + '最大面積:' + str(tmp_area) + ' 画像面積：' + str(image_area) + "|||")
    """


    """ 最大値と最小値で検証
    VARIANT = 12 # 許容する偏差の値 TODO 要調整
    DARKGRAY = 120 # 暗すぎるグレー　TODO 要調整
    # read_img = cv.imread("13527690_5105761_1.png_process.jpg")
    read_img = cv.imread("13506218_5105739_1.png")


    gray_image = cv.cvtColor(read_img, cv.COLOR_RGB2GRAY)
    cv.imwrite('gray.jpg', gray_image)
    th3 = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 1)
    cv.imwrite('test.jpg', th3)
    #ヒストグラム平滑化を実施してみる
    eq_hist_img = cv.equalizeHist(gray_image)
    cv.imwrite('eq_hist_img.jpg', eq_hist_img)
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    cll = clahe.apply(gray_image)
    cv.imwrite('eq_hist_clahe.jpg', cll)
    hsv_img = cv.cvtColor(read_img, cv.COLOR_HSV2BGR)
    h, s, v = cv.split(hsv_img)
    cv.imwrite('h.jpg', h)
    cv.imwrite('s.jpg', s)
    cv.imwrite('v.jpg', v)
    # ガウシアンフィルタで画像をぼかす
    img_gaus_blur = cv.GaussianBlur(read_img, (15, 15), 0)
    # gbrに分解
    g, b, r = cv.split(img_gaus_blur)

    for i in range(len(g)):
        for j in range(len(g[i])):
            # 各画素の偏差を求める
            avg = (int(g[i][j]) + int(b[i][j]) + int(r[i][j])) / 3
            list = [g[i][j], b[i][j], r[i][j]]
            maxval = max(list)
            minval = min(list)
            # 色の偏差が10以上ある場合は色がついている部分とみなして、色を黒にする
            #if (g[i][j] - avg > VARIANT or b[i][j] - avg > VARIANT or r[i][j] - avg > VARIANT):
            # 色の最小値と最大値の差が20より大きい場合は色がついている部分とみなして背景色を黒に変更する。
            if ((maxval - minval) >= 20):
                g[i][j] = 0
                b[i][j] = 0
                r[i][j] = 0
            else:
                # グレーでも暗すぎる場合は黒とみなす TODO ここも要調整
                if (avg < DARKGRAY):
                    g[i][j] = 0
                    b[i][j] = 0
                    r[i][j] = 0

    # 分解した画像を元に戻す
    processed_image = cv.merge((g, b, r))
    cv.imwrite('omori_process.jpg', processed_image)
    # gray_image_ = cv.cvtColor(read_img, cv.COLOR_RGB2GRAY)
    # edges = cv.Canny(gray_image_, 100, 200)

    # グレースケールに変換
    gray_image = cv.cvtColor(processed_image, cv.COLOR_RGB2GRAY)

    # 2値化を行う
    retval, otsu_binary = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    cv.imwrite('otsu_binary.png', otsu_binary)

    # 縮小処理を行う
    kernel = np.ones((8, 8), np.uint8)
    erosion = cv.erode(otsu_binary, kernel, iterations=1)
    # 膨張処理を行う。こうすることで斑点を削除することができる
    dilate = cv.dilate(erosion, kernel, iterations=1)

    erosion = cv.erode(dilate, kernel, iterations=1)
    dilate = cv.dilate(erosion, kernel, iterations=1)
    cv.imwrite('dilate.jpg', dilate)
    # 画像の矩形を調べる
    contours, hierarchy = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 抽出された矩形の中で最も面積の大きいものを調べる
    max_cnt = 0  # 矩形のインデックス
    tmp_area = 0  # 矩形の面積
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            # 最も外側の矩形のみ対象に面積を求める
            # cv.drawContours(src, contours, i, (0, 255, 0), 3)
            if cv.contourArea(contours[i]) > tmp_area:
                tmp_area = cv.contourArea(contours[i])
                max_cnt = i

    cv.drawContours(read_img, contours, max_cnt, (0, 255, 0), 3)
    cv.imwrite('max_area_contours_noapr.jpg', read_img)
    # 求まった面積最大の矩形の形状を大まかなものに近似する
    arclen = cv.arcLength(contours[max_cnt], closed=False)
    approx = cv.approxPolyDP(contours[max_cnt], epsilon=arclen * 0.025, closed=False)
    cv.drawContours(read_img, [approx], -1, (0, 255, 0), 3)
    cv.imwrite('max_area_contours.jpg', read_img)
    """

    # 使ってるとこ　ここから
    # image_file = 'sample.png'
    # src = cv.imread(image_file, cv.IMREAD_COLOR)
    # src_ = cv.imread(image_file, cv.IMREAD_COLOR)
    # src__ = cv.imread(image_file, cv.IMREAD_COLOR)
    # src___ = cv.imread(image_file, cv.IMREAD_COLOR)
    # src____ = cv.imread(image_file, cv.IMREAD_COLOR)
    #
    # img_gaus_blur = cv.GaussianBlur(src, (15, 15), 0)
    # img_gray = cv.cvtColor(img_gaus_blur, cv.COLOR_RGB2GRAY)
    # img_gray_ = cv.cvtColor(img_gaus_blur, cv.COLOR_RGB2GRAY)
    # cv.imwrite('img_gaus_blur.jpg', img_gaus_blur)
    # img_hsv = cv.cvtColor(src, cv.COLOR_HSV2BGR)
    # img_hsv_gaused = cv.cvtColor(img_gaus_blur, cv.COLOR_HSV2BGR)
    #
    # cv.imwrite('hsv.jpg', img_hsv)
    # cv.imwrite('hsv_gaused.jpg', img_hsv_gaused)
    #
    # # 白の範囲のHSV(背景茶色で有効だった。消さない)
    # lower_white = np.array([0, 0, 100])
    # upper_white = np.array([180, 200, 255])
    #
    # # 白以外にマスク
    # mask_white = cv.inRange(img_hsv, lower_white, upper_white)
    # res_white = cv.bitwise_and(src, src, mask=mask_white)
    # cv.imwrite('mask_white.jpg', mask_white)
    # cv.imwrite('res_white.jpg', res_white)
    #
    # mask_white_gaused = cv.inRange(img_hsv_gaused, lower_white, upper_white)
    # res_white_gaused = cv.bitwise_and(src, src, mask=mask_white_gaused)
    # cv.imwrite('mask_white_gaused.jpg', mask_white_gaused)
    # cv.imwrite('res_white_gaused.jpg', res_white_gaused)
    #
    # # 大津の2値化
    # retval, otsu_binary = cv.threshold(mask_white_gaused, 0, 255, cv.THRESH_OTSU)
    # cv.imwrite('otsu_binary.jpg', otsu_binary)
    #
    # contours, hierarchy = cv.findContours(otsu_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    # src_with_shapes = cv.drawContours(src, contours, -1, (0, 255, 0), 3)
    # cv.imwrite('srcwithshape.jpg', src_with_shapes)
    # max_cnt = 0
    # tmp_area = 0
    # for i in range(len(contours)):
    #     if hierarchy[0][i][3] == -1:
    #         # cv.drawContours(src, contours, i, (0, 255, 0), 3)
    #         if cv.contourArea(contours[i]) > tmp_area:
    #             tmp_area = cv.contourArea(contours[i])
    #             max_cnt = i
    #
    # cv.drawContours(img_gray, contours, max_cnt, (0, 255, 0), 3)
    # arclen = cv.arcLength(contours[max_cnt], closed=False)
    # approx = cv.approxPolyDP(contours[max_cnt], epsilon=arclen * 0.025, closed=False)
    # cv.drawContours(src_, [approx], -1, (0, 255, 0), 3)
    # cv.imwrite('max_area_contours.jpg', img_gray)
    # cv.imwrite('approx_img.jpg', src_)
    #
    # height, width = src__.shape[:2]
    # height = height - 10
    # width = width - 10
    # print(height)
    # print(width)
    #
    # for pixel in approx:
    #     height_ = pixel[0][1]
    #     width_ = pixel[0][0]
    #     print("縦：" + str(height_))
    #     print("横：" + str(width_))
    #     if height_ >= height or height_ <= 10:
    #         print("縦がはみ出してるかもよ")
    #
    #     if width_ >= width or width_ <= 10:
    #         print("横がはみ出してるかもよ")
    #
    # th3 = cv.adaptiveThreshold(img_gray_, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 41, 1)
    # cv.imwrite('adaptive.jpg', th3)
    # cv.bitwise_not(th3, th3)
    # cv.imwrite('adaptive_.jpg', th3)
    # contours, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(src___, contours, -1, (0, 255, 0), 3)
    # cv.imwrite('white_receipt_contour.jpg', src___)
    # #
    # for i in range(len(contours)):
    #     if hierarchy[0][i][3] == -1:
    #         # cv.drawContours(src, contours, i, (0, 255, 0), 3)
    #         if cv.contourArea(contours[i]) > tmp_area:
    #             tmp_area = cv.contourArea(contours[i])
    #             max_cnt = i
    #
    # arclen = cv.arcLength(contours[max_cnt], closed=False)
    # approx = cv.approxPolyDP(contours[max_cnt], epsilon=arclen * 0.1, closed=False)
    # cv.drawContours(src____, [approx], -1, (0, 255, 0), 3)
    # cv.imwrite('adaptive_contour.jpg', src____)
    # 使ってるとこここまで

    # cv.imwrite('img_gray.jpg', img_gray)
    # image_blurred = cv.GaussianBlur(img_gray, (15, 15), 0)
    #
    # image_blurred_org = cv.GaussianBlur(src, (15, 15), 0)
    #
    # img_gray_ = cv.cvtColor(image_blurred_org, cv.COLOR_RGB2GRAY)

    # ハイパス フィルタ
    # kernel_high_pass = np.array([
    #     [-1, -1, -1],
    #     [-1, 8, -1],
    #     [-1, -1, -1]
    # ], np.float32)
    # img_high_pass = cv.filter2D(img_gray, -1, kernel_high_pass)
    # cv.imwrite('haipass.jpg', img_high_pass)


    # 陰影除去(original)
    # blur = cv.blur(src, (51, 51))
    # rij = src/blur
    # index_0 = np.where(rij < 0.98)
    # index_1 = np.where(rij >= 0.98)
    # rij[index_0] = 0
    # rij[index_1] = 1
    # cv.imwrite('ineidel_org.jpg', rij * 255)

    # 陰影除去(gray)
    # blur = cv.blur(img_gray, (51, 51))
    # rij = img_gray/blur
    # index_1 = np.where(rij >= 0.98)
    # index_0 = np.where(rij < 0.98)
    # rij[index_0] = 0
    # rij[index_1] = 1
    # cv.imwrite('ineidel_gray.jpg', rij * 255)

    # 陰影除去(gray_gaussian)
    # blur = cv.blur(image_blurred, (51, 51))
    # rij = img_gray / blur
    # index_1 = np.where(rij >= 0.98)
    # index_0 = np.where(rij < 0.98)
    # rij[index_0] = 0
    # rij[index_1] = 1
    # ineidel_gau = rij * 255
    # cv.imwrite('ineidel_gau.jpg', ineidel_gau)

    # rij_int = np.array(rij * 255, np.uint8)
    #
    # cedges = cv.Canny(rij_int, 50, 200, apertureSize=3)
    # cv.imwrite('cedges_ineidel_gau.jpg', cedges)
    #
    # cedges = cv.Canny(img_gray, 50, 200, apertureSize=3)
    # cv.imwrite('cedges.jpg', cedges)
    #
    #
    # cv.imwrite('result_1.jpg', image_blurred)
    # retval, dst = cv.threshold(image_blurred, 200, 255, cv.THRESH_TOZERO_INV)
    # # image_blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
    # th3 = cv.adaptiveThreshold(image_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 29, 2)
    # cv.imwrite('result_3.jpg', th3)



    # contours, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # dst = cv.drawContours(th3, contours, -1, (0, 0, 255, 255), 2, cv.LINE_AA)

    # image_blurred = cv.GaussianBlur(img_gray, (5, 5), 0)

    # edges = cv.Canny(image_blurred, 50, 200, apertureSize=3)
    #
    # cv.imwrite('result_4.jpg', edges)
    # contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cont = cv.drawContours(src, contours, -1, (0, 255, 0), 3)
    # cv.imwrite('result_5.jpg', cont)
    #
    # contours, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # test3 = cv.drawContours(src, contours, -1, (0, 255, 0), 3)
    # cv.imwrite('result_7.jpg', test3)
    # max_cnt = max(contours, key=lambda x: cv.contourArea(x))
    # test4 = cv.drawContours(th3, [max_cnt], -1, color=255, thickness=-1)
    # cv.imwrite('likely_large_figure.jpg', test4)


if __name__ == '__main__':
    # main_4() # 四隅判定
    # main_pinboke() # シャープ画像取得
    get_exifinfo()